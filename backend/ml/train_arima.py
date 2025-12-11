# backend/ml/train_arima.py
"""
Train a non-seasonal ARIMA model for monthly total_expense forecasting.

Features:
- Robust parsing of history JSON (accepts list or dict{'history': [...]})
- Small grid-search over (p,d,q) candidates with fallbacks for convergence issues
- If all ARIMA fits fail, returns a MeanPredictor fallback
- Saves either a pickled ARIMA model or a minimal JSON fallback (no pickling of local objects)
- Provides a loader helper to rehydrate either model type
"""

import json
import os
import pickle
from math import ceil
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# -------------------------
# Minimal fallback predictor (module-level so it can be used by loader)
# -------------------------
class MeanPredictor:
    """
    Simple mean-value predictor with a minimal statsmodels-like interface:
      - predict(steps) -> numpy array
      - get_forecast(steps) -> object with attribute .predicted_mean (numpy array)
    """
    def __init__(self, train_series):
        try:
            self.mean_value = float(pd.Series(train_series).astype(float).mean())
        except Exception:
            # fallback
            self.mean_value = float(train_series) if train_series is not None else 0.0

    def predict(self, steps=1):
        return np.full((steps,), self.mean_value, dtype=float)

    def get_forecast(self, steps=1):
        class FR:
            def __init__(self, arr):
                self.predicted_mean = arr
        return FR(np.full((steps,), self.mean_value, dtype=float))


# -------------------------
# Utilities: load history & build series
# -------------------------
def load_history_from_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def build_total_expense_series(history: Any) -> pd.Series:
    """
    Robustly convert the `history` input into a pandas Series of monthly totals.

    Accepts either:
      - a list of month-dicts: [ { "month": "2025-01", "categories": {...} }, ... ]
      - or a dict: { "history": [ ... ] }

    Each month dict may contain:
      - "total_expense" (preferred)
      - OR "categories" dict (will be summed)
    Returns a pandas Series indexed by PeriodIndex (freq='M').
    """
    # Normalize to a list
    if isinstance(history, list):
        hist_list = history
    elif isinstance(history, dict) and isinstance(history.get("history"), list):
        hist_list = history["history"]
    else:
        raise ValueError("Unsupported history format: expected list or dict with key 'history'.")

    rows = []
    for i, m in enumerate(hist_list):
        if not isinstance(m, dict):
            print(f"Warning: skipping non-dict entry at index {i}")
            continue

        month = m.get("month")
        if not month:
            print(f"Warning: skipping entry at index {i} because 'month' key is missing")
            continue

        total = None
        if "total_expense" in m and m["total_expense"] not in (None, ""):
            try:
                total = float(m["total_expense"])
            except Exception:
                print(f"Warning: invalid total_expense at index {i}; attempting to sum categories")
                total = None

        if total is None:
            cats = m.get("categories", {})
            if isinstance(cats, dict) and cats:
                try:
                    total = float(sum(float(v) for v in cats.values()))
                except Exception:
                    print(f"Warning: invalid category values at index {i}; setting total to 0.0")
                    total = 0.0
            else:
                print(f"Warning: month {month} has neither 'total_expense' nor 'categories'; using 0.0")
                total = 0.0

        rows.append((month, total))

    if not rows:
        raise ValueError("No valid monthly rows found in history input.")

    df = pd.DataFrame(rows, columns=["month", "total"])
    df = df.sort_values("month")

    # Convert to PeriodIndex (YYYY-MM)
    try:
        idx = pd.PeriodIndex(df["month"].astype(str), freq="M")
    except Exception:
        parsed = pd.to_datetime(df["month"].astype(str), errors="coerce")
        if parsed.isnull().any():
            raise ValueError("Unable to parse all month strings into dates.")
        idx = pd.PeriodIndex(parsed.dt.to_period("M"))

    series = pd.Series(df["total"].astype(float).values, index=idx)
    return series


# -------------------------
# Robust ARIMA trainer with fallbacks
# -------------------------
def train_arima_on_series(
    series: pd.Series,
    train_frac: float = 0.8,
    pdq_candidates: List[Tuple[int, int, int]] = None,
    enforce_stationarity: bool = False,
    enforce_invertible: bool = False,
    maxiter: int = 50
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train ARIMA on the provided monthly series with robust fallbacks.

    Returns:
      - model: statsmodels fitted model OR MeanPredictor (fallback)
      - metadata: dict describing chosen order or fallback details
    """
    n = len(series)
    if n < 6:
        raise ValueError(f"Not enough samples for ARIMA fitting: got {n}, recommend >= 6 (prefer >=12)")

    split = int(ceil(train_frac * n))
    train = series.iloc[:split].astype(float)
    val = series.iloc[split:].astype(float)

    if pdq_candidates is None:
        pdq_candidates = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 0)]

    tried = []
    results = []
    best_mse = float("inf")
    best_model = None
    best_cfg = None

    def try_fit(order, enforce_stat, enforce_inv, local_maxiter):
        nonlocal best_mse, best_model, best_cfg
        try:
            print(f"Attempting ARIMA order={order} stationarity={enforce_stat} invertible={enforce_inv} maxiter={local_maxiter} ...", end=" ")
            model = ARIMA(
                train,
                order=order,
                enforce_stationarity=enforce_stat,
                enforce_invertible=enforce_inv
            )
            fitted = model.fit(method_kwargs={"maxiter": local_maxiter}, disp=False)
            steps = len(val)
            if steps == 0:
                pred = fitted.fittedvalues
                mse = mean_squared_error(train, pred) if len(pred) == len(train) else float("inf")
            else:
                forecast_res = fitted.get_forecast(steps=steps)
                pred = forecast_res.predicted_mean
                mse = mean_squared_error(val, pred)
            print(f"OK (val_mse={mse:.4f})")
            results.append({"order": order, "stationary": enforce_stat, "invertible": enforce_inv, "val_mse": float(mse)})
            if mse < best_mse:
                best_mse = mse
                best_model = fitted
                best_cfg = {"order": order, "stationary": enforce_stat, "invertible": enforce_inv}
            return True
        except Exception as e:
            print(f"FAILED ({e})")
            results.append({"order": order, "stationary": enforce_stat, "invertible": enforce_inv, "error": str(e)})
            return False

    # Primary grid
    for order in pdq_candidates:
        tried.append(("primary", order))
        ok = try_fit(order, enforce_stationarity, enforce_invertible, maxiter)
        if not ok:
            combos = [(not enforce_stationarity, enforce_invertible), (enforce_stationarity, not enforce_invertible), (True, True)]
            for st, inv in combos:
                if (st, inv) == (enforce_stationarity, enforce_invertible):
                    continue
                tried.append(("fallback_flags", order, st, inv))
                ok2 = try_fit(order, st, inv, max(10, maxiter // 2))
                if ok2:
                    break

    # Ultra-conservative fallback orders
    if best_model is None:
        fallback_orders = [(0, 1, 0), (0, 0, 1), (1, 0, 0), (0, 1, 1)]
        for order in fallback_orders:
            tried.append(("ultra_fallback", order))
            ok = try_fit(order, True, True, max(20, maxiter // 2))
            if ok:
                break

    # Try differencing and fit a simple ARIMA on differenced data
    if best_model is None:
        try:
            print("Attempting one differencing on the full series and retrying ARIMA(0,0,1) on diffed data...")
            diffed = series.diff().dropna()
            if len(diffed) >= max(6, int(0.5 * n)):
                split_d = int(ceil(train_frac * len(diffed)))
                train_d = diffed.iloc[:split_d]
                val_d = diffed.iloc[split_d:]
                model = ARIMA(train_d, order=(0, 0, 1), enforce_stationarity=True, enforce_invertible=True)
                fitted = model.fit(method_kwargs={"maxiter": max(20, maxiter // 2)}, disp=False)
                steps = len(val_d)
                if steps == 0:
                    pred = fitted.fittedvalues
                    mse = mean_squared_error(train_d, pred)
                else:
                    forecast_res = fitted.get_forecast(steps=steps)
                    pred = forecast_res.predicted_mean
                    mse = mean_squared_error(val_d, pred)
                print(f"OK on differenced data (val_mse={mse:.4f})")
                best_model = fitted
                best_cfg = {"order": (0, 0, 1), "differenced": True}
                results.append({"order": (0, 0, 1), "differenced": True, "val_mse": float(mse)})
        except Exception as e:
            print(f"Differenced fit failed: {e}")
            results.append({"differenced_attempt_error": str(e)})

    # Final fallback: return MeanPredictor (do not attempt to pickle this)
    if best_model is None:
        print("All ARIMA fitting attempts failed. Falling back to MeanPredictor.")
        mean_model = MeanPredictor(train)
        metadata = {
            "best_order": None,
            "best_val_mse": None,
            "candidates_tried": results,
            "fallback": "mean_predictor",
            "n_samples": n,
            "train_size": int(split),
            "val_size": int(n - split)
        }
        return mean_model, metadata

    # Refit selected config on full series when possible
    try:
        if best_cfg.get("differenced"):
            final_model = best_model  # already fitted on differenced data
        else:
            print(f"Refitting selected ARIMA order={best_cfg.get('order')} on full series...")
            final_model = ARIMA(
                series,
                order=best_cfg.get("order"),
                enforce_stationarity=best_cfg.get("stationary", enforce_stationarity),
                enforce_invertible=best_cfg.get("invertible", enforce_invertible)
            ).fit(method_kwargs={"maxiter": maxiter}, disp=False)
    except Exception as e:
        print(f"Refit on full series failed; using the best model trained on train split. Error: {e}")
        final_model = best_model

    metadata = {
        "best_order": best_cfg.get("order"),
        "best_stationary": best_cfg.get("stationary", None),
        "best_invertible": best_cfg.get("invertible", None),
        "best_val_mse": float(best_mse) if best_mse != float("inf") else None,
        "candidates_tried": results,
        "n_samples": n,
        "train_size": int(split),
        "val_size": int(n - split)
    }

    return final_model, metadata


# -------------------------
# Save / Load helpers
# -------------------------
def save_model(model, metadata, save_dir):
    """
    Save trained model and metadata.

    - If model is a MeanPredictor (fallback), save arima_fallback.json with {"mean": <float>, "metadata": {...}}
    - Otherwise pickle the model to arima_model.pkl and save metadata to arima_metadata.json
    Returns tuple (model_path_or_json, meta_path).
    """
    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, "arima_metadata.json")

    # Always write metadata
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Detect fallback by duck-typing
    is_fallback = False
    try:
        if isinstance(model, MeanPredictor) or (hasattr(model, "mean_value") and hasattr(model, "get_forecast")):
            is_fallback = True
    except Exception:
        is_fallback = False

    if is_fallback:
        fallback_path = os.path.join(save_dir, "arima_fallback.json")
        payload = {
            "mean": float(getattr(model, "mean_value", 0.0)),
            "metadata": metadata
        }
        with open(fallback_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Fallback model saved to JSON: {fallback_path}")
        return fallback_path, meta_path

    # Try pickling the model
    model_path = os.path.join(save_dir, "arima_model.pkl")
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model pickled to: {model_path}")
        return model_path, meta_path
    except Exception as e:
        # Safety net: fallback to JSON metadata-only save
        print(f"Warning: failed to pickle model ({e}). Falling back to metadata-only JSON.")
        fallback_path = os.path.join(save_dir, "arima_fallback.json")
        payload = {
            "note": "pickle_failed",
            "error": str(e),
            "metadata": metadata
        }
        with open(fallback_path, "w") as f:
            json.dump(payload, f, indent=2)
        return fallback_path, meta_path


def load_saved_arima_model(save_dir):
    """
    Load saved ARIMA model or fallback JSON from save_dir.
    Returns object with get_forecast(steps) -> object with `.predicted_mean` (numpy array).
    """
    model_path = os.path.join(save_dir, "arima_model.pkl")
    fallback_path = os.path.join(save_dir, "arima_fallback.json")

    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                mdl = pickle.load(f)
            return mdl
        except Exception as e:
            print(f"Warning: could not unpickle model: {e}")

    if os.path.exists(fallback_path):
        with open(fallback_path, "r") as f:
            payload = json.load(f)
        mean_val = payload.get("mean")
        if mean_val is not None:
            class FallbackWrapper:
                def __init__(self, mean):
                    self.mean_value = float(mean)
                def predict(self, steps=1):
                    return np.full((steps,), self.mean_value, dtype=float)
                def get_forecast(self, steps=1):
                    class FR:
                        def __init__(self, arr):
                            self.predicted_mean = arr
                    return FR(np.full((steps,), self.mean_value, dtype=float))
            return FallbackWrapper(mean_val)
        else:
            raise RuntimeError(f"Saved fallback JSON does not contain 'mean' in {fallback_path}. Payload: {payload}")

    raise FileNotFoundError(f"No model or fallback found in {save_dir}")


# -------------------------
# CLI usage (main)
# -------------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, "training_data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"training_data.json not found at {json_path}")

    history = load_history_from_json(json_path)
    series = build_total_expense_series(history)

    print("Series preview:")
    print(series.to_string())

    # Candidate grid (conservative)
    pdq_candidates = [(0, 1, 1), (1, 1, 0), (1, 1, 1)]

    model, metadata = train_arima_on_series(
        series,
        train_frac=0.8,
        pdq_candidates=pdq_candidates,
        enforce_stationarity=False,
        enforce_invertible=False,
        maxiter=50
    )

    save_dir = os.path.join(os.path.dirname(__file__), "arima")
    model_path, meta_path = save_model(model, metadata, save_dir)

    print("\nTraining complete.")
    print(f"Model saved at: {model_path}")
    print(f"Metadata saved at: {meta_path}")
