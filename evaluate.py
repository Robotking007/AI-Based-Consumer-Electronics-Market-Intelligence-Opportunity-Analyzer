"""
evaluate.py - Model Evaluation Step for SageMaker Pipeline Quality Gate
Evaluates challenger model on the fixed test set and outputs evaluation.json
matching SageMaker PropertyFile specification (Section 4 & 5 of PROJECT.md).
"""

import os
import sys
import json
import tarfile
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate(model_dir: str, test_path: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading test dataset from: {test_path}...")
    test_df = pd.read_csv(test_path)

    # First column is target_demand_growth_pct, remaining are features
    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values

    # Locate model file (extracted or inside tar)
    raw_model_path = os.path.join(model_dir, "xgboost-model")
    tar_model_path = os.path.join(model_dir, "model.tar.gz")

    if not os.path.exists(raw_model_path) and os.path.exists(tar_model_path):
        print(f"Extracting model from: {tar_model_path}...")
        with tarfile.open(tar_model_path, "r:gz") as tar:
            tar.extractall(path=model_dir)

    model = xgb.XGBRegressor()
    model.load_model(raw_model_path)
    print(f"Loaded trained XGBoost model from: {raw_model_path}")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Compute key MLOps evaluation metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    mean_target = float(np.mean(y_test))
    rel_error = float((mae / mean_target) * 100.0)

    eval_report = {
        "regression_metrics": {
            "rmse": {
                "value": round(rmse, 4),
                "standard_deviation": round(float(np.std(np.abs(y_test - y_pred))), 4)
            },
            "mae": {
                "value": round(mae, 4)
            },
            "r2": {
                "value": round(r2, 4)
            },
            "mean_demand_growth": {
                "value": round(mean_target, 2)
            },
            "relative_mae_percent": {
                "value": round(rel_error, 2)
            }
        },
        "evaluation_dataset": {
            "test_sample_count": len(y_test),
            "features_evaluated": X_test.shape[1]
        }
    }

    eval_output_file = os.path.join(output_dir, "evaluation.json")
    with open(eval_output_file, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print("\n================ EVALUATION RESULTS ================")
    print(f" Test Set Size     : {len(y_test):,} samples")
    print(f" RMSE              : {rmse:.3f} (Lower is better)")
    print(f" MAE               : {mae:.3f} (Average prediction error)")
    print(f" R² Score          : {r2:.3f} (Variance explained)")
    print(f" Relative Error    : {rel_error:.1f}%")
    print(f" Report Saved To   : {eval_output_file}")
    print("====================================================\n")

    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="model_output", help="Directory containing model artifact")
    parser.add_argument("--test-data", default="processed_data/test.csv", help="Test dataset path")
    parser.add_argument("--output-dir", default="evaluation_output", help="Evaluation output directory")

    args = parser.parse_args()
    evaluate(args.model_dir, args.test_data, args.output_dir)
