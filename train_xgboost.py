"""
train_xgboost.py - Model Training Step for SageMaker Training Job and Local Training
Trains an XGBoost Regressor to predict 30-day demand growth (Section 3 & 4 of PROJECT.md).
"""

import os
import sys
import json
import tarfile
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb


def train(
    train_path: str,
    val_path: str,
    output_dir: str,
    max_depth: int = 6,
    eta: float = 0.08,
    n_estimators: int = 150,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85
):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading training and validation datasets...")

    # Load without header (standard SageMaker XGBoost format: column 0 is target)
    train_data = pd.read_csv(train_path, header=None)
    val_data = pd.read_csv(val_path, header=None)

    y_train = train_data.iloc[:, 0].values
    X_train = train_data.iloc[:, 1:].values

    y_val = val_data.iloc[:, 0].values
    X_val = val_data.iloc[:, 1:].values

    print(f"Train samples: {len(X_train):,}, Validation samples: {len(X_val):,}")
    print(f"Features: {X_train.shape[1]}")

    model = xgb.XGBRegressor(
        max_depth=max_depth,
        learning_rate=eta,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        objective="reg:squarederror",
        eval_metric=["rmse", "mae"]
    )

    print("\nTraining XGBoost Regressor...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=30
    )

    # Save model
    model_file = os.path.join(output_dir, "xgboost-model.json")
    model.save_model(model_file)
    print(f"\nSaved raw XGBoost model to: {model_file}")

    # Also keep legacy xgboost-model link/copy for backward compatibility
    legacy_file = os.path.join(output_dir, "xgboost-model")
    try:
        import shutil
        shutil.copy2(model_file, legacy_file)
    except Exception:
        pass

    # Also save as SageMaker compatible model.tar.gz
    tar_path = os.path.join(output_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_file, arcname="xgboost-model.json")
        if os.path.exists(legacy_file):
            tar.add(legacy_file, arcname="xgboost-model")
    print(f"Packaged SageMaker model artifact: {tar_path}")

    # Evaluate validation metrics
    val_preds = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
    val_mae = np.mean(np.abs(val_preds - y_val))
    print(f"\nValidation Performance -> RMSE: {val_rmse:.3f} | MAE: {val_mae:.3f}")

    return model, val_rmse, val_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Support SageMaker environment variables or CLI flags
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "processed_data")
    val_dir = os.environ.get("SM_CHANNEL_VALIDATION", "processed_data")
    model_dir = os.environ.get("SM_MODEL_DIR", "model_output")

    parser.add_argument("--train-data", default=os.path.join(train_dir, "train.csv"))
    parser.add_argument("--val-data", default=os.path.join(val_dir, "validation.csv"))
    parser.add_argument("--model-dir", default=model_dir)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.08)
    parser.add_argument("--n-estimators", type=int, default=150)

    args = parser.parse_args()
    train(
        train_path=args.train_data,
        val_path=args.val_data,
        output_dir=args.model_dir,
        max_depth=args.max_depth,
        eta=args.eta,
        n_estimators=args.n_estimators
    )
