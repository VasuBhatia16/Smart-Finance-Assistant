# backend/ml/predict.py
import os
import pickle
import torch
import numpy as np
from typing import List, Dict, Any
from xgboost import XGBRegressor

from ml.model import LSTMForecaster
from ml.preprocess import build_feature_matrix

TARGET_IDX = 2
class Predictor:
    def __init__(self, model_type: str="lstm"):
        """
        model_type: "lstm" or "xgb"
        """

        self.model_type = model_type.lower()
        print("Initializing Predictor with model type:", self.model_type)
        here = os.path.dirname(__file__)

        if self.model_type == "lstm":
            self.model_path = os.path.join(here,"lstm", "lstm_model.pt")
            self.scaler_path = os.path.join(here,"lstm", "scaler.pkl")
        elif self.model_type == "xgb":
            self.model_path = os.path.join(here, "xgb", "xgb_model.json")
            self.scaler_path = os.path.join(here, "xgb", "scaler.pkl")
        else:
            raise ValueError("model_type must be 'lstm' or 'xgb'")

        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        if self.model_type == "lstm":
            input_dim = len(self.scaler.mean_)
            self.model = LSTMForecaster(input_dim=input_dim)
            self.model.load_state_dict(
                torch.load(self.model_path, map_location="cpu")
            )
            self.model.eval()

        elif self.model_type == "xgb":
            self.model = XGBRegressor()
            self.model.load_model(self.model_path)

    # --------- Shared category logic ----------
    def _dynamic_category_weights(self, history):
        category_sum = {}
        total_spend = 0
        for entry in history:
            for c, v in (entry.get("categories") or {}).items():
                try:
                    val = float(v)
                except:
                    continue
                category_sum[c] = category_sum.get(c, 0.0) + val
                total_spend += val

        if total_spend == 0 or not category_sum:
            n = max(len(category_sum), 1)
            return {c: 1 / n for c in (category_sum or {"other": 1})}

        return {c: s / total_spend for c, s in category_sum.items()}

    # ---------- Main prediction ----------
    def predict_next_month(self, history, window=6):
        M = build_feature_matrix(history)
        if len(M) < window:
            raise ValueError(f"Need at least {window} months")

        M_scaled = self.scaler.transform(M)

        # ----- LSTM -----
        if self.model_type == "lstm":
            last_window = M_scaled[-window:]
            X = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred_scaled = float(self.model(X).item())

        # ----- XGBoost -----
        else:
            last_window = M_scaled[-window:]
            X = last_window.flatten().reshape(1, -1)
            pred_scaled = float(self.model.predict(X)[0])

        dummy = np.zeros((1, len(self.scaler.mean_)))
        dummy[0, TARGET_IDX] = pred_scaled
        unscaled = self.scaler.inverse_transform(dummy)
        real_total = float(unscaled[0, TARGET_IDX])

        weights = self._dynamic_category_weights(history)
        breakdown = {c: real_total * w for c, w in weights.items()}
        # if(breakdown.get("Food",0)>15000):
        #     breakdown["Food"]-=7000
        #     real_total-=7000
        # if(breakdown.get("Misc",0)>25000):
        #     breakdown["Misc"]-=5000
        #     real_total-=5000
        # if(breakdown.get("food",0)>15000):
        #     breakdown["food"]-=7000
        #     real_total-=7000
        # if(breakdown.get("misc",0)>25000):
        #     breakdown["misc"]-=5000
        #     real_total-=5000
        return {
            "predicted_total_expenses": real_total,
            "category_breakdown": breakdown,
            "category_weights": weights,
        }
