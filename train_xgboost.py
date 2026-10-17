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


def resolve_channel_path(path_or_dir: str, default_filename: str) -> str:
    """Resolve directory or file path from SageMaker channel configuration."""
    if not path_or_dir:
        return default_filename
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        candidate = os.path.join(path_or_dir, default_filename)
        if os.path.exists(candidate):
            return candidate
        # Search for any CSV file in the channel directory
        csvs = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir) if f.endswith(".csv")]
        if csvs:
            return csvs[0]
    return path_or_dir


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
    resolved_train = resolve_channel_path(train_path, "train.csv")
    resolved_val = resolve_channel_path(val_path, "validation.csv")

    print(f"Loading training dataset from   : {resolved_train}")
    print(f"Loading validation dataset from : {resolved_val}")

    if not os.path.exists(resolved_train):
        raise FileNotFoundError(f"Training dataset does not exist at: {resolved_train}")
    if not os.path.exists(resolved_val):
        raise FileNotFoundError(f"Validation dataset does not exist at: {resolved_val}")

    # Load without header (standard SageMaker XGBoost format: column 0 is target)
    train_data = pd.read_csv(resolved_train, header=None)
    val_data = pd.read_csv(resolved_val, header=None)

    # Validate data integrity
    train_nulls = train_data.isna().sum().sum()
    val_nulls = val_data.isna().sum().sum()
    if train_nulls > 0 or val_nulls > 0:
        raise ValueError(f"Corrupt data detected! NaN count: train={train_nulls}, val={val_nulls}")

    y_train = train_data.iloc[:, 0].values.astype(np.float32)
    X_train = train_data.iloc[:, 1:].values.astype(np.float32)

    y_val = val_data.iloc[:, 0].values.astype(np.float32)
    X_val = val_data.iloc[:, 1:].values.astype(np.float32)

    print(f"Train samples: {len(X_train):,}, Validation samples: {len(X_val):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Target Range (Train): min={y_train.min():.2f}, max={y_train.max():.2f}, mean={y_train.mean():.2f}")

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

    import pickle
    import joblib

    # 1. Save standard Scikit-Learn Joblib format (.joblib)
    joblib_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, joblib_path)
    print(f"\nSaved Scikit-Learn Joblib model to: {joblib_path}")

    # 2. Save Python Pickle format (.pkl)
    pkl_path = os.path.join(output_dir, "model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved Python Pickle model to: {pkl_path}")

    # 3. Save raw XGBoost JSON model for SageMaker container
    model_file = os.path.join(output_dir, "xgboost-model.json")
    model.save_model(model_file)
    print(f"Saved raw XGBoost JSON model to: {model_file}")

    # Also keep legacy xgboost-model link/copy for backward compatibility
    legacy_file = os.path.join(output_dir, "xgboost-model")
    try:
        import shutil
        shutil.copy2(model_file, legacy_file)
    except Exception:
        pass

    # Package all formats into SageMaker compatible model.tar.gz
    tar_path = os.path.join(output_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(joblib_path, arcname="model.joblib")
        tar.add(pkl_path, arcname="model.pkl")
        tar.add(model_file, arcname="xgboost-model.json")
        if os.path.exists(legacy_file):
            tar.add(legacy_file, arcname="xgboost-model")
    print(f"Packaged SageMaker model artifact (.joblib, .pkl, .json): {tar_path}")

    # Evaluate validation metrics
    val_preds = model.predict(X_val)
    val_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
    val_mae = float(np.mean(np.abs(val_preds - y_val)))
    print(f"\nValidation Performance -> RMSE: {val_rmse:.3f} | MAE: {val_mae:.3f}")

    return model, val_rmse, val_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SageMaker XGBoost Training Step")
    # Support SageMaker environment variables or CLI flags
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "processed_data")
    val_dir = os.environ.get("SM_CHANNEL_VALIDATION", "processed_data")
    model_dir = os.environ.get("SM_MODEL_DIR", "model_output")

    parser.add_argument("--train-data", default=train_dir)
    parser.add_argument("--val-data", default=val_dir)
    parser.add_argument("--model-dir", default=model_dir)

    # Accept standard or SageMaker hyperparameter flags
    hp_max_depth = int(os.environ.get("SM_HP_MAX_DEPTH", 6))
    hp_eta = float(os.environ.get("SM_HP_ETA", 0.08))
    hp_num_round = int(os.environ.get("SM_HP_NUM_ROUND", 150))
    hp_subsample = float(os.environ.get("SM_HP_SUBSAMPLE", 0.85))
    hp_colsample = float(os.environ.get("SM_HP_COLSAMPLE_BYTREE", 0.85))

    parser.add_argument("--max-depth", type=int, default=hp_max_depth)
    parser.add_argument("--eta", type=float, default=hp_eta)
    parser.add_argument("--n-estimators", "--num-round", "--num_round", dest="n_estimators", type=int, default=hp_num_round)
    parser.add_argument("--subsample", type=float, default=hp_subsample)
    parser.add_argument("--colsample-bytree", "--colsample_bytree", dest="colsample_bytree", type=float, default=hp_colsample)

    args = parser.parse_args()
    train(
        train_path=args.train_data,
        val_path=args.val_data,
        output_dir=args.model_dir,
        max_depth=args.max_depth,
        eta=args.eta,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree
    )
