# backend/ml/predict.py
import os
import pickle
import torch
from typing import List, Dict, Any
import numpy as np
from ml.model import LSTMForecaster
from ml.preprocess import build_feature_matrix

TARGET_IDX = 2  # total_expense is the 3rd feature per preprocess.py

class Predictor:
    def __init__(self, model_path: str | None = None, scaler_path: str | None = None):
        here = os.path.dirname(__file__)
        self.model_path = model_path or os.path.join(here,"lstm", "lstm_model.pt")
        self.scaler_path = scaler_path or os.path.join(here,"lstm", "scaler.pkl")

        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        input_dim = len(self.scaler.mean_)
        self.model = LSTMForecaster(input_dim=input_dim)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model.load_state_dict(
            torch.load(self.model_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

    def _dynamic_category_weights(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        category_sum: Dict[str, float] = {}
        total_spend = 0.0
        for entry in history:
            cats: Dict[str, float] = entry.get("categories", {}) or {}
            for c, v in cats.items():
                try:
                    val = float(v)
                except:
                    continue
                category_sum[c] = category_sum.get(c, 0.0) + val
                total_spend += val

        if total_spend <= 0 or not category_sum:
            n = max(len(category_sum), 1)
            return {c: 1.0 / n for c in (category_sum or {"other": 1.0}).keys()}
        return {c: s / total_spend for c, s in category_sum.items()}

    def predict_next_month(self, history: List[Dict[str, Any]], window: int = 6) -> Dict[str, Any]:
        M = build_feature_matrix(history)
        if len(M) < window:
            raise ValueError(f"Need at least {window} months")

        last_window = M[-window:]
        last_scaled = self.scaler.transform(last_window)
        X = torch.tensor(last_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = float(self.model(X).item())

        dummy = np.zeros((1, len(self.scaler.mean_)))
        dummy[0, TARGET_IDX] = pred_scaled
        unscaled = self.scaler.inverse_transform(dummy)
        real_total = float(unscaled[0, TARGET_IDX])

        weights = self._dynamic_category_weights(history)
        breakdown = {c: real_total * w for c, w in weights.items()}

        return {
            "predicted_total_expenses": real_total,
            "category_breakdown": breakdown,
            "category_weights": weights,
        }
