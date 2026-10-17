"""
s3_ingest.py - Automated S3 Data Ingestion for Consumer Electronics Market Intelligence
Supports:
1. Real AWS S3 partitioned ingestion (s3://<bucket>/raw/date=YYYY-MM-DD/)
2. Local simulated S3 storage (for testing and offline development)
3. AWS Lambda handler compatibility (EventBridge scheduled ingestion)
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def validate_dataset(csv_path: str) -> pd.DataFrame:
    """Validate dataset schema, size, and critical non-null constraints."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    print(f"Validating dataset from: {csv_path}...")
    df = pd.read_csv(csv_path)

    required_cols = [
        "product_id", "product_name", "category",
        "discounted_price", "actual_price", "discount_percentage",
        "rating", "rating_count", "market_channel"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset: {col}")

    # Check non-null constraints
    null_counts = df[required_cols].isna().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Null values detected in critical columns: {null_counts.to_dict()}")

    print(f"Validation passed: {len(df):,} rows, {len(df.columns)} columns.")
    return df


def ingest_to_s3(
    file_path: str = "consumer_electronics_market_20k.csv",
    bucket_name: str = None,
    partition_date: str = None,
    local_mock: bool = False
) -> dict:
    """
    Ingest dataset to S3 under raw/date=YYYY-MM-DD/ partitioned prefix.
    If local_mock=True or AWS credentials are unavailable, stores into ./mock_s3_storage/.
    """
    df = validate_dataset(file_path)

    if not partition_date:
        partition_date = datetime.utcnow().strftime("%Y-%m-%d")

    s3_key = f"raw/date={partition_date}/consumer_electronics_market_20k.csv"
    meta_key = f"raw/date={partition_date}/metadata.json"

    channel_breakdown = df["market_channel"].value_counts().to_dict()
    metadata = {
        "dataset_name": "consumer_electronics_market_20k",
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "partition_date": partition_date,
        "ingested_at_utc": datetime.utcnow().isoformat(),
        "market_channels": channel_breakdown
    }

    # Decide between AWS S3 or Local Mock S3
    use_aws = False
    if not local_mock and BOTO3_AVAILABLE and bucket_name:
        try:
            s3_client = boto3.client("s3")
            # Quick check if credentials work
            s3_client.list_buckets()
            use_aws = True
        except (NoCredentialsError, ClientError, Exception) as e:
            print(f"AWS credentials not active ({e}). Falling back to local simulated S3 storage.")
            use_aws = False

    if use_aws:
        print(f"\n[AWS S3] Uploading to s3://{bucket_name}/{s3_key}...")
        s3_client = boto3.client("s3")
        
        # Ensure bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except ClientError:
            print(f"Creating S3 bucket: {bucket_name}...")
            s3_client.create_bucket(Bucket=bucket_name)

        s3_client.upload_file(file_path, bucket_name, s3_key)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=meta_key,
            Body=json.dumps(metadata, indent=2),
            ContentType="application/json"
        )
        dest_uri = f"s3://{bucket_name}/{s3_key}"
        print(f"[AWS S3] Successfully ingested dataset to: {dest_uri}")
    else:
        # Local simulated S3 directory structure
        local_dir = os.path.join("mock_s3_storage", "raw", f"date={partition_date}")
        os.makedirs(local_dir, exist_ok=True)
        local_data_path = os.path.join(local_dir, "consumer_electronics_market_20k.csv")
        local_meta_path = os.path.join(local_dir, "metadata.json")

        df.to_csv(local_data_path, index=False, encoding="utf-8")
        with open(local_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        dest_uri = os.path.abspath(local_data_path)
        print(f"\n[Simulated S3 Storage] Ingested dataset to: {dest_uri}")
        print(f"[Simulated S3 Storage] Partition Key: raw/date={partition_date}/")

    return {
        "status": "SUCCESS",
        "destination_uri": dest_uri,
        "partition_date": partition_date,
        "total_rows": len(df),
        "metadata": metadata
    }


def lambda_handler(event, context):
    """
    AWS Lambda entrypoint triggered by EventBridge cron rule (PROJECT.md Section 4).
    """
    bucket = os.environ.get("S3_BUCKET_NAME", "consumer-electronics-mlops")
    file_path = event.get("file_path", "consumer_electronics_market_20k.csv")
    date_str = event.get("date", datetime.utcnow().strftime("%Y-%m-%d"))

    result = ingest_to_s3(
        file_path=file_path,
        bucket_name=bucket,
        partition_date=date_str
    )
    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest consumer electronics dataset into S3 storage.")
    parser.add_argument("--file", default="consumer_electronics_market_20k.csv", help="Path to source CSV file")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET_NAME", ""), help="AWS S3 Bucket name")
    parser.add_argument("--date", default=None, help="Partition date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--local", action="store_true", help="Force local simulated S3 mode")

    args = parser.parse_args()
    res = ingest_to_s3(
        file_path=args.file,
        bucket_name=args.bucket,
        partition_date=args.date,
        local_mock=args.local or not bool(args.bucket)
    )
    print(f"\nIngestion Result: {res['status']} | Total records: {res['total_rows']:,}")
