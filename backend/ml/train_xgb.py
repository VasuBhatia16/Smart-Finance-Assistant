# backend/ml/train_xgb.py
"""
XGBoost trainer with manual early stopping (LSTM-style).

Key behavior:
- Uses low-level xgb.train loop, one boosting round at a time.
- After each round, evaluate on validation set and track best val RMSE.
- If val RMSE doesn't improve for `patience` rounds, early stop.
- Saves the best model (as JSON) and scaler (pickle).
"""

import os
import json
import pickle
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
        X.append(M[i - window:i].flatten())
        y.append(M[i, target_idx])
    return np.array(X), np.array(y)


def train_xgb_manual_early_stopping(
    X_train, y_train, X_val, y_val,
    params,
    num_rounds=500,
    patience=20,
    verbose=True
):
    """
    Train XGBoost using manual early stopping.

    Returns:
      best_booster (xgb.Booster), best_val_rmse (float), best_round (int)
    """

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    booster = None
    best_val = float("inf")
    best_model_raw = None
    es_counter = 0
    best_round = 0

    # For optional tracking of train RMSE per round (not required)
    for round_idx in range(1, num_rounds + 1):
        # Train one boosting round (continue from previous booster if any)
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1,
            xgb_model=booster,
            verbose_eval=False
        )

        # Predict on train & val to compute RMSE (for logging)
        preds_train = booster.predict(dtrain)
        preds_val = booster.predict(dval)

        train_rmse = float(np.sqrt(np.mean((preds_train - y_train) ** 2)))
        val_rmse = float(np.sqrt(np.mean((preds_val - y_val) ** 2)))

        if verbose:
            print(f"Round {round_idx}/{num_rounds}  |  Train RMSE = {train_rmse:.4f}  |  Val RMSE = {val_rmse:.4f}")

        # Early stopping check (LSTM-style)
        if val_rmse < best_val:
            best_val = val_rmse
            # Save raw bytes of current booster state as best snapshot
            best_model_raw = booster.save_raw()  # bytes
            es_counter = 0
            best_round = round_idx
        else:
            es_counter += 1

        if es_counter >= patience:
            if verbose:
                print("\nEarly stopping triggered!")
                print(f"Stopping at round {round_idx}. Best round was {best_round} with Val RMSE = {best_val:.4f}")
            break

    # Restore best booster
    if best_model_raw is None:
        # No improvement ever (rare) — use final booster
        final_booster = booster
    else:
        final_booster = xgb.Booster()
        # load_model accepts bytes-like (bytearray)
        final_booster.load_model(bytearray(best_model_raw))

    return final_booster, best_val, best_round


def train_xgb(history, window=6, num_rounds=500, patience=20):
    print("Using manual early stopping XGBoost training")
    print("XGBoost version:", xgb.__version__)

    # Build feature matrix (T x F)
    M = build_feature_matrix(history)
    M = np.asarray(M)
    if M.shape[0] < window + 5:
        raise ValueError("Not enough samples to train XGBoost. Increase history length or lower window.")

    # Scale features (fit on entire matrix; supervised split is later)
    scaler = StandardScaler()
    M_scaled = scaler.fit_transform(M)

    # Supervised dataset
    X, y = make_supervised_dataset(M_scaled, window=window, target_idx=2)  # target_idx=2 per your LSTM code

    # Train/val split (time-order)
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"Dataset: total samples={n}, train={len(X_train)}, val={len(X_val)}")

    # Prepare params (tune as needed)
    params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "rmse",
        "verbosity": 0
    }

    # Train with manual early stopping
    booster, best_val, best_round = train_xgb_manual_early_stopping(
        X_train, y_train, X_val, y_val,
        params=params,
        num_rounds=num_rounds,
        patience=patience,
        verbose=True
    )

    # Save model and scaler
    output_dir = os.path.join(os.path.dirname(__file__), "xgb")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "xgb_model.json")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    # Save booster as JSON (preferred; human-readable)
    booster.save_model(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("\nTraining complete.")
    print(f"Best Val RMSE: {best_val:.4f} at round {best_round}")
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, "training_data.json")
    history = load_history_from_json(json_path)
    train_xgb(history, window=6, num_rounds=500, patience=20)
