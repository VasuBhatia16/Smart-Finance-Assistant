# backend/ml/train_xgb.py
import json
import pickle
import os
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from ml.preprocess import build_feature_matrix


def load_history_from_json(path):
    with open(path, "r") as f:
        return json.load(f)


def make_supervised_dataset(M, window, target_idx):
    X, y = [], []
    T, F = M.shape

    for i in range(window, T):
        X.append(M[i-window:i].flatten())   # Flatten window
        y.append(M[i, target_idx])          # Next-step value

    return np.array(X), np.array(y)


def train_xgb(history, window=6):
    print("Using Low-Level XGBoost API (compatible mode)")
    print("XGBoost version:", xgb.__version__)

    print("Building feature matrix...")
    M = build_feature_matrix(history)

    if M.shape[0] < 20:
        raise ValueError("Not enough samples to train XGBoost.")

    # SCALE FEATURES
    scaler = StandardScaler()
    M_scaled = scaler.fit_transform(M)

    print("Creating supervised dataset...")
    X, y = make_supervised_dataset(M_scaled, window, target_idx=2)

    # train/val split
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Convert to DMatrix (low-level)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Training parameters
    params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "rmse",
    }

    print("\nTraining XGBoost model with built-in early stopping...")

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=True
    )

    print("\nSaving model and scaler...")
    output_dir = os.path.join(os.path.dirname(__file__), "xgb")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "xgb_model.json")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    print(f"Saving model to {model_path}")
    model.save_model(model_path)

    print(f"Saving scaler to {scaler_path}")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("\nTraining complete.")
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, "training_data.json")
    history = load_history_from_json(json_path)
    train_xgb(history,window=6)
