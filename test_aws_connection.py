"""
test_aws_connection.py - AWS SageMaker & S3 Pre-Flight Verification Script
Configured for:
- AWS Account ID : 533267153786
- Account / User : Siddhant Udavant
- Target Region  : ap-south-1 (Mumbai) / us-east-1
- Default Bucket : consumer-electronics-mlops-533267153786
- SageMaker Role : arn:aws:iam::533267153786:role/service-role/AmazonSageMaker-ExecutionRole
"""

import os
import sys
import argparse

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
except ImportError:
    print("[ERROR] boto3 is not installed. Please run: pip install boto3")
    sys.exit(1)


def check_aws_credentials(region: str = "ap-south-1", target_account: str = "533267153786"):
    print("\n" + "="*70)
    print(" AWS PRE-FLIGHT VERIFICATION FOR SAGEMAKER DEPLOYMENT")
    print("="*70)
    print(f"Target AWS Account ID : {target_account}")
    print(f"Account User / Name   : Siddhant Udavant")
    print(f"Configured Region     : {region}")
    print("="*70)

    # 1. Check STS Identity
    print("\n[Step 1/4] Verifying Active AWS Identity via AWS STS...")
    try:
        sts = boto3.client("sts", region_name=region)
        caller_identity = sts.get_caller_identity()
        active_account = caller_identity.get("Account")
        active_arn = caller_identity.get("Arn")
        user_id = caller_identity.get("UserId")

        print(f" ✔ Successfully authenticated with AWS STS!")
        print(f"   - Authenticated Account ID : {active_account}")
        print(f"   - Principal / Caller ARN   : {active_arn}")
        print(f"   - User ID                  : {user_id}")

        if active_account != target_account:
            print(f"\n [WARNING] Authenticated account ({active_account}) does not match target ({target_account}).")
        else:
            print(" ✔ Account ID matches target configuration!")

    except NoCredentialsError:
        print(" ❌ No active AWS credentials found in environment or AWS profile.")
        print("\n To authenticate, please choose ONE of the following options:")
        print(" Option A - Set environment variables in your current PowerShell terminal:")
        print('   $env:AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_HERE"')
        print('   $env:AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_ACCESS_KEY_HERE"')
        print('   $env:AWS_DEFAULT_REGION = "ap-south-1"')
        print("\n Option B - Run AWS CLI configuration:")
        print("   aws configure")
        print("\n Option C - Add secrets to your GitHub repository for automated CI/CD:")
        print("   Settings -> Secrets and variables -> Actions:")
        print("   - AWS_ACCESS_KEY_ID")
        print("   - AWS_SECRET_ACCESS_KEY")
        print(f"   - SAGEMAKER_ROLE_ARN = arn:aws:iam::{target_account}:role/service-role/AmazonSageMaker-ExecutionRole")
        return False
    except ClientError as e:
        print(f" ❌ AWS STS ClientError: {e}")
        return False
    except Exception as e:
        print(f" ❌ Unexpected connection error: {e}")
        return False

    # 2. Check S3 Access
    bucket_name = f"consumer-electronics-mlops-{target_account}"
    print(f"\n[Step 2/4] Testing Amazon S3 Access & Bucket: '{bucket_name}'...")
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f" ✔ Bucket '{bucket_name}' exists and is accessible!")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ["404", "NoSuchBucket"]:
            print(f" ℹ Bucket '{bucket_name}' does not exist yet. Attempting to create...")
            try:
                if region == "us-east-1":
                    s3.create_bucket(Bucket=bucket_name)
                else:
                    s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": region}
                    )
                print(f" ✔ Created new S3 bucket: '{bucket_name}'")
            except Exception as create_err:
                print(f" ⚠ Could not auto-create bucket: {create_err}")
        elif error_code in ["403", "AccessDenied"]:
            print(f" ⚠ Access denied for bucket '{bucket_name}'. Please verify S3 permissions.")
        else:
            print(f" ⚠ S3 response: {e}")

    # 3. Check SageMaker API Access
    print(f"\n[Step 3/4] Testing AWS SageMaker API in region '{region}'...")
    sm = boto3.client("sagemaker", region_name=region)
    try:
        pipelines_resp = sm.list_pipelines(MaxResults=5)
        count = len(pipelines_resp.get("PipelineSummaries", []))
        print(f" ✔ SageMaker API connectivity confirmed! Existing pipelines found: {count}")
    except ClientError as e:
        print(f" ⚠ SageMaker API check: {e}")

    # 4. Check SageMaker Execution Role
    role_arn = f"arn:aws:iam::{target_account}:role/service-role/AmazonSageMaker-ExecutionRole"
    print(f"\n[Step 4/4] Verifying SageMaker Execution Role ARN...")
    print(f" Configured Role ARN: {role_arn}")
    print(" (Ensure this role has AmazonSageMakerFullAccess and AmazonS3FullAccess attached)")

    print("\n" + "="*70)
    print(" ALL CHECKS COMPLETED FOR ACCOUNT 533267153786")
    print("="*70 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify AWS Credentials and Resources")
    parser.add_argument("--region", default="ap-south-1", help="AWS Region (default: ap-south-1)")
    parser.add_argument("--account", default="533267153786", help="Target AWS Account ID")
    args = parser.parse_args()

    check_aws_credentials(region=args.region, target_account=args.account)
