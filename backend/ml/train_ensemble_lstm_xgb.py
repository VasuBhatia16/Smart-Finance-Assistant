# backend/ml/train_ensemble_lstm_xgb.py
"""
Train an ensemble that stacks LSTM + XGBoost using a linear meta-learner.

Strategy:
- Build feature matrix M (T, F) via build_feature_matrix(history).
- Scale features with StandardScaler.
- Create supervised windows with window size W -> N samples.
- Split time-ordered into:
    base_train (first 60%),
    meta_train  (next 20%),
    test        (last 20%)
- Train LSTM on base_train (with early stopping as in your LSTM code).
- Train XGBoost on base_train with manual early stopping (round-by-round).
- Generate predictions of both base models on meta_train -> train a LinearRegression meta-learner.
- Evaluate ensemble on test set and save all artifacts.
"""

import os
import json
import pickle
from math import ceil
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import xgboost as xgb

from ml.preprocess import build_feature_matrix
from ml.dataset import TimeSeriesDataset
from ml.model import LSTMForecaster


# ----------------------------
# XGBoost manual early-stopping helper (as used before)
# ----------------------------
def train_xgb_manual_early_stopping(
    X_train, y_train, X_val, y_val,
    params,
    num_rounds=500,
    patience=20,
    verbose=True
) -> Tuple[xgb.Booster, float, int]:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    booster = None
    best_val = float("inf")
    best_model_raw = None
    es_counter = 0
    best_round = 0

    for round_idx in range(1, num_rounds + 1):
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1,
            xgb_model=booster,
            verbose_eval=False
        )

        preds_val = booster.predict(dval)
        val_rmse = float(np.sqrt(np.mean((preds_val - y_val) ** 2)))

        if verbose:
            preds_train = booster.predict(dtrain)
            train_rmse = float(np.sqrt(np.mean((preds_train - y_train) ** 2)))
            print(f"XGB Round {round_idx}/{num_rounds} | Train RMSE={train_rmse:.4f} | Val RMSE={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_model_raw = booster.save_raw()
            es_counter = 0
            best_round = round_idx
        else:
            es_counter += 1

        if es_counter >= patience:
            if verbose:
                print(f"\nXGB Early stopping triggered at round {round_idx}. Best round {best_round} with Val RMSE={best_val:.4f}")
            break

    # Restore best booster
    if best_model_raw is None:
        final_booster = booster
    else:
        final_booster = xgb.Booster()
        final_booster.load_model(bytearray(best_model_raw))

    return final_booster, best_val, best_round


# ----------------------------
# LSTM training helper (mirrors your train_lstm logic)
# ----------------------------
def train_lstm_model(
    dataset: TimeSeriesDataset,
    train_idx_range,
    val_idx_range,
    input_dim: int,
    window: int,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = None
) -> Tuple[LSTMForecaster, dict]:
    """
    Train LSTMForecaster on a subset of dataset using early stopping.
    dataset: TimeSeriesDataset built over M_scaled with window_size=window
    train_idx_range, val_idx_range: ranges or lists of indices in dataset to use
    Returns trained model (best restored) and metadata dict.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Subsets and loaders
    train_ds = torch.utils.data.Subset(dataset, train_idx_range)
    val_ds = torch.utils.data.Subset(dataset, val_idx_range)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMForecaster(input_dim=input_dim)
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    es_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_seq, y in train_loader:
            X_seq = X_seq.to(device).float()
            y = y.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            pred = model(X_seq)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_seq, y in val_loader:
                X_seq = X_seq.to(device).float()
                y = y.to(device).float().unsqueeze(1)
                pred = model(X_seq)
                val_loss += loss_fn(pred, y).item()

        train_avg = total_loss / (len(train_loader) if len(train_loader) > 0 else 1)
        val_avg = val_loss / (len(val_loader) if len(val_loader) > 0 else 1)
        print(f"LSTM Epoch {epoch+1}/{epochs} | Train Loss={train_avg:.4f} | Val Loss={val_avg:.4f}")

        if val_avg < best_val:
            best_val = val_avg
            best_state = model.state_dict()
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= patience:
                print("\nLSTM Early stopping triggered!")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {"best_val_loss": best_val}
    return model, metadata


# ----------------------------
# Supervised helpers
# ----------------------------
def make_supervised_dataset_flat(M_scaled: np.ndarray, window: int, target_idx: int = 2):
    """
    Create flattened-window supervised X and y.
    X[i] = flatten(M_scaled[i-window:i])
    y[i] = M_scaled[i, target_idx]  (use original scale target later)
    Returns X (N, window*F), y (N,)
    """
    T, F = M_scaled.shape
    X, y = [], []
    for i in range(window, T):
        X.append(M_scaled[i - window:i].flatten())
        y.append(M_scaled[i, target_idx])
    return np.array(X), np.array(y)


# ----------------------------
# Main ensemble training routine
# ----------------------------
def train_ensemble(history, window=6, device=None):
    # Build feature matrix and scale
    M = build_feature_matrix(history)   # (T, F)
    M = np.asarray(M, dtype=float)
    scaler = StandardScaler()
    M_scaled = scaler.fit_transform(M)

    # Supervised flattened dataset for XGBoost and for aligning indices
    X_all, y_all_scaled = make_supervised_dataset_flat(M_scaled, window=window, target_idx=2)
    N = len(X_all)
    if N < 10:
        raise ValueError(f"Not enough supervised samples ({N}). Need more history months.")

    # We'll keep a copy of the original (unscaled) target for final reporting
    # Recreate y in original scale (target index from original M)
    _, y_all = make_supervised_dataset_flat(M, window=window, target_idx=2)

    # Split indices: base_train 60%, meta_train 20%, test 20% (time order)
    base_frac = 0.6
    meta_frac = 0.2
    base_end = int(base_frac * N)
    meta_end = base_end + int(meta_frac * N)
    # ensure at least one sample in each
    if meta_end >= N:
        meta_end = base_end + 1
    if base_end < 3:
        raise ValueError("Not enough base training samples; reduce window or add history.")

    idx_base = list(range(0, base_end))
    idx_meta = list(range(base_end, meta_end))
    idx_test = list(range(meta_end, N))

    print(f"Total supervised samples: {N} | base_train={len(idx_base)} meta_train={len(idx_meta)} test={len(idx_test)}")

    # ----------------------------
    # LSTM: build TimeSeriesDataset over M_scaled with same window
    # dataset indices naturally map to supervised sample indices (0..N-1)
    # ----------------------------
    dataset = TimeSeriesDataset(M_scaled, window_size=window, target_idx=2)

    # Train LSTM on base indices, validate on meta indices (meta acts as val-> produce meta features later)
    lstm_epochs = 100
    lstm_batch = 16
    lstm_lr = 1e-3
    lstm_patience = 10

    lstm_model, lstm_meta = train_lstm_model(
        dataset=dataset,
        train_idx_range=idx_base,
        val_idx_range=idx_meta,
        input_dim=M_scaled.shape[1],
        window=window,
        epochs=lstm_epochs,
        batch_size=lstm_batch,
        lr=lstm_lr,
        patience=lstm_patience,
        device=device
    )

    # ----------------------------
    # XGBoost: prepare X/y for base training
    # X_all (flattened windows) and y_all (original scale target)
    # ----------------------------
    # Use original scale for target for XGB training
    X_base = X_all[idx_base]
    y_base = y_all[idx_base]
    X_meta = X_all[idx_meta]
    y_meta = y_all[idx_meta]
    X_test = X_all[idx_test]
    y_test = y_all[idx_test]

    # Further split base for XGB early stopping: use 80/20 within base
    inner_split = int(0.8 * len(X_base))
    X_base_train, y_base_train = X_base[:inner_split], y_base[:inner_split]
    X_base_val, y_base_val = X_base[inner_split:], y_base[inner_split:]

    xgb_params = {
        "objective": "reg:squarederror",
        "eta": 0.03,
        "max_depth": 3,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_lambda": 5.0,
        "verbosity": 0
    }

    xgb_num_rounds = 1000
    xgb_patience = 30

    xgb_model, xgb_best_val, xgb_best_round = train_xgb_manual_early_stopping(
        X_base_train, y_base_train,
        X_base_val, y_base_val,
        params=xgb_params,
        num_rounds=xgb_num_rounds,
        patience=xgb_patience,
        verbose=True
    )

    # ----------------------------
    # Create meta-training features:
    # For meta_train period produce predictions from both base models (LSTM and XGB)
    # ----------------------------
    # LSTM predictions on meta set
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lstm_model.to(device)
    lstm_model.eval()
    meta_preds_lstm = []
    with torch.no_grad():
        # iterate over meta subset from dataset (indices idx_meta)
        for idx in idx_meta:
            X_seq, _ = dataset[idx]  # X_seq shape (window, F) or (F, window) depending on dataset
            # Ensure shape: (1, window, F) or (1, seq_len, features)
            X_in = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(device)
            yhat = lstm_model(X_in).cpu().numpy().reshape(-1)[0]
            meta_preds_lstm.append(float(yhat))
    meta_preds_lstm = np.array(meta_preds_lstm)

    # XGB predictions on meta set
    dmeta = xgb.DMatrix(X_meta)
    meta_preds_xgb = xgb_model.predict(dmeta)

    # Stack into meta features [n_meta x 2]
    X_meta_stack = np.vstack([meta_preds_lstm, meta_preds_xgb]).T
    y_meta_true = y_meta  # original-scale targets

    # Train linear meta-learner
    meta_lr = LinearRegression()
    meta_lr.fit(X_meta_stack, y_meta_true)
    print("Meta-learner coefficients:", meta_lr.coef_, "intercept:", meta_lr.intercept_)

    # ----------------------------
    # Evaluate ensemble on TEST set
    # ----------------------------
    # LSTM predictions on test
    test_preds_lstm = []
    with torch.no_grad():
        for idx in idx_test:
            X_seq, _ = dataset[idx]
            X_in = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(device)
            yhat = lstm_model(X_in).cpu().numpy().reshape(-1)[0]
            test_preds_lstm.append(float(yhat))
    test_preds_lstm = np.array(test_preds_lstm)

    # XGB predictions on test
    dtest = xgb.DMatrix(X_test)
    test_preds_xgb = xgb_model.predict(dtest)

    X_test_stack = np.vstack([test_preds_lstm, test_preds_xgb]).T
    ensemble_preds = meta_lr.predict(X_test_stack)

    # Compute RMSEs (compatible with older sklearn)
    rmse_lstm = float(np.sqrt(mean_squared_error(y_test, test_preds_lstm)))
    rmse_xgb = float(np.sqrt(mean_squared_error(y_test, test_preds_xgb)))
    rmse_ensemble = float(np.sqrt(mean_squared_error(y_test, ensemble_preds)))

    print(f"Test RMSE LSTM: {rmse_lstm:.4f} | XGB: {rmse_xgb:.4f} | Ensemble: {rmse_ensemble:.4f}")

    # ----------------------------
    # Save artifacts
    # ----------------------------
    save_dir = os.path.join(os.path.dirname(__file__), "ensemble")
    os.makedirs(save_dir, exist_ok=True)

    # LSTM: save state_dict and scaler used to create dataset
    lstm_path = os.path.join(save_dir, "lstm_model.pt")
    torch.save(lstm_model.state_dict(), lstm_path)

    # Save XGBoost model
    xgb_path = os.path.join(save_dir, "xgb_model.json")
    xgb_model.save_model(xgb_path)

    # Save scaler and meta learner
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    meta_path = os.path.join(save_dir, "meta_lr.pkl")
    meta_meta = os.path.join(save_dir, "ensemble_metadata.json")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(meta_path, "wb") as f:
        pickle.dump(meta_lr, f)

    metadata = {
        "rmse_lstm": float(rmse_lstm),
        "rmse_xgb": float(rmse_xgb),
        "rmse_ensemble": float(rmse_ensemble),
        "xgb_best_round": int(xgb_best_round),
        "xgb_val_rmse": float(xgb_best_val),
        "lstm_best_val_loss": float(lstm_meta.get("best_val_loss", None)),
        "n_supervised_samples": int(N),
        "indices": {"base_end": base_end, "meta_end": meta_end}
    }

    with open(meta_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved ensemble artifacts:")
    print(" LSTM:", lstm_path)
    print(" XGB:", xgb_path)
    print(" Scaler:", scaler_path)
    print(" Meta LR:", meta_path)
    print(" Metadata:", meta_meta)

    return {
        "lstm_model_path": lstm_path,
        "xgb_model_path": xgb_path,
        "scaler_path": scaler_path,
        "meta_path": meta_path,
        "metadata": metadata
    }


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, "training_data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"training_data.json not found at {json_path}")
    with open(json_path, "r") as f:
        history = json.load(f)

    train_ensemble(history, window=6)
