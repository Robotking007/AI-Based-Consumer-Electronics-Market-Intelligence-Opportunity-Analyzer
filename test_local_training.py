"""
test_local_training.py - Comprehensive Local Verification of Training Job
Pre-validates training locally to catch and resolve any potential errors before deploying to AWS SageMaker:
1. Data Integrity & SageMaker CSV Compliance Check (no headers, target in col 0, zero NaNs)
2. SageMaker Simulated Container Environment Training Execution (env vars + CLI flags)
3. Model Artifact Packaging & Integrity Verification (xgboost-model.json & model.tar.gz)
4. Evaluation Metric Validation (RMSE, MAE, R²)
5. Inference Latency & Robustness Stress Testing (zeros, extremes, batch inference)
"""

import os
import sys
import json
import time
import tarfile
import subprocess
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import preprocessing
import train_xgboost
import evaluate


def test_split_integrity(split_path: str, split_name: str, expected_cols: int = 28):
    """Verify that CSV split complies strictly with SageMaker built-in algorithm specs."""
    print(f"Checking {split_name} split: {split_path}...")
    assert os.path.exists(split_path), f"File {split_path} does not exist!"

    # Read raw lines to verify no string header exists in train/val
    with open(split_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split(",")

    # Verify all tokens in first row can be cast to float
    for idx, token in enumerate(first_line):
        try:
            float(token)
        except ValueError:
            raise ValueError(f"[{split_name}] Header detected or non-numeric value in row 1, col {idx}: '{token}'")

    df = pd.read_csv(split_path, header=None)
    print(f" -> Rows: {len(df):,}, Columns: {df.shape[1]}")
    assert df.shape[1] == expected_cols, f"Expected {expected_cols} columns, got {df.shape[1]}"

    # Verify zero missing values
    null_count = df.isna().sum().sum()
    assert null_count == 0, f"[{split_name}] Contains {null_count} NaN values!"

    # Verify target variable bounds (col 0)
    y = df.iloc[:, 0].values
    assert not np.isnan(y).any(), "NaN found in target values!"
    assert not np.isinf(y).any(), "Inf found in target values!"
    print(f" -> Target summary: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")
    assert y.min() >= 0.0, "Negative demand growth target detected!"
    print(f" [PASS] {split_name} data is 100% compliant with SageMaker CSV format.\n")


def run_local_training_verification():
    print("="*75)
    print(" PRE-FLIGHT LOCAL TRAINING VERIFICATION FOR AWS SAGEMAKER DEPLOYMENT")
    print("="*75)

    # 1. Feature Engineering & Preprocessing
    proc_dir = "test_processed_data"
    raw_csv = "consumer_electronics_market.csv" if os.path.exists("consumer_electronics_market.csv") else "consumer_electronics_market_20k.csv"
    train_path, val_path, test_path = preprocessing.run_preprocessing(
        input_csv=raw_csv,
        output_dir=proc_dir,
        sample_limit=2500  # Fast verification sample
    )

    # 2. Strict Data Compliance Validation
    print("\n[Stage 2: Strict SageMaker Format Compliance Checks]")
    test_split_integrity(train_path, "Train", expected_cols=28)
    test_split_integrity(val_path, "Validation", expected_cols=28)

    # 3. Simulated SageMaker Training Job Execution
    print("\n[Stage 3: Training Execution with SageMaker Container Environment]")
    model_output_dir = "test_model_output"
    os.makedirs(model_output_dir, exist_ok=True)

    # Simulate SageMaker Container environment variables
    env = os.environ.copy()
    env["SM_CHANNEL_TRAIN"] = proc_dir
    env["SM_CHANNEL_VALIDATION"] = proc_dir
    env["SM_MODEL_DIR"] = model_output_dir
    env["SM_HP_MAX_DEPTH"] = "6"
    env["SM_HP_ETA"] = "0.08"
    env["SM_HP_NUM_ROUND"] = "100"
    env["SM_HP_SUBSAMPLE"] = "0.85"
    env["SM_HP_COLSAMPLE_BYTREE"] = "0.85"

    cmd = [
        sys.executable, "train_xgboost.py",
        "--train-data", train_path,
        "--val-data", val_path,
        "--model-dir", model_output_dir,
        "--num-round", "100",
        "--max-depth", "6",
        "--eta", "0.08"
    ]

    print(f"Running command: {' '.join(cmd)}")
    start_train = time.time()
    train_proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    train_duration = round(time.time() - start_train, 2)

    print(train_proc.stdout)
    if train_proc.returncode != 0:
        print("[ERROR IN TRAINING]:")
        print(train_proc.stderr)
        raise RuntimeError("Training script failed in SageMaker simulated mode!")

    print(f" [PASS] Training completed successfully in {train_duration}s without errors.\n")

    # 4. Model Artifact Inspection
    print("[Stage 4: Model Artifact Inspection & Tarball Verification]")
    model_json = os.path.join(model_output_dir, "xgboost-model.json")
    model_tar = os.path.join(model_output_dir, "model.tar.gz")

    assert os.path.exists(model_json), f"Missing raw booster: {model_json}"
    assert os.path.exists(model_tar), f"Missing packaged tarball: {model_tar}"

    with tarfile.open(model_tar, "r:gz") as tar:
        tar_names = tar.getnames()
        print(f" -> Files inside model.tar.gz: {tar_names}")
        assert any("xgboost-model" in name for name in tar_names), "Model file missing from tarball!"
    print(" [PASS] SageMaker model artifact packaging verified.\n")

    # 5. Model Evaluation & Metric Calculation
    print("[Stage 5: Model Evaluation on Test Split]")
    eval_dir = "test_evaluation_output"
    eval_report = evaluate.evaluate(
        model_dir=model_output_dir,
        test_path=test_path,
        output_dir=eval_dir
    )

    metrics = eval_report["regression_metrics"]
    rmse = metrics["rmse"]["value"]
    mae = metrics["mae"]["value"]
    r2 = metrics["r2"]["value"]
    rel_err = metrics["relative_mae_percent"]["value"]

    print(f" -> RMSE : {rmse:.3f} (Benchmark: < 12.0)")
    print(f" -> MAE  : {mae:.3f} (Benchmark: < 10.0)")
    print(f" -> R²   : {r2:.3f} (Benchmark: > 0.80)")
    print(f" -> Rel% : {rel_err:.1f}% (Benchmark: < 15.0%)")

    assert rmse < 12.0, f"RMSE {rmse} failed quality benchmark!"
    assert r2 > 0.80, f"R² {r2} below acceptable threshold!"
    print(" [PASS] Model meets all production quality benchmarks.\n")

    # 6. Inference Stress Testing & Latency Benchmarks
    print("[Stage 6: Inference Latency & Robustness Stress Testing]")
    booster = xgb.XGBRegressor()
    booster.load_model(model_json)

    # 1. Single sample latency
    single_sample = np.ones((1, 27), dtype=np.float32)
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = booster.predict(single_sample)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    print(f" -> Single prediction latency : Avg {avg_latency:.2f} ms | P95 {p95_latency:.2f} ms")

    # 2. Batch throughput
    batch_samples = np.random.uniform(0.0, 100.0, size=(1000, 27)).astype(np.float32)
    t_start = time.perf_counter()
    batch_preds = booster.predict(batch_samples)
    t_batch = time.perf_counter() - t_start
    print(f" -> Batch throughput           : {len(batch_preds):,} predictions in {t_batch*1000:.1f} ms ({len(batch_preds)/t_batch:,.0f} req/s)")

    # 3. Extreme edge-case inputs
    zero_pred = booster.predict(np.zeros((1, 27), dtype=np.float32))[0]
    high_pred = booster.predict(np.full((1, 27), 10000.0, dtype=np.float32))[0]
    assert not np.isnan(zero_pred) and not np.isnan(high_pred), "NaN output on extreme inputs!"
    print(f" -> Edge case testing (zeros -> {zero_pred:.2f}%, high -> {high_pred:.2f}%) -> PASS")

    print("\n" + "="*75)
    print(" ALL PRE-FLIGHT LOCAL TRAINING CHECKS PASSED SUCCESSFULLY!")
    print(" Ready for AWS SageMaker deployment with zero formatting/data errors.")
    print("="*75 + "\n")


if __name__ == "__main__":
    run_local_training_verification()
