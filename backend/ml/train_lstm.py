# backend/ml/train_lstm.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os

from ml.model import LSTMForecaster
from ml.dataset import TimeSeriesDataset
from ml.preprocess import build_feature_matrix

def load_history_from_json(path):
    with open(path, "r") as f:
        return json.load(f)

def train_model(history, window=6, epochs=50, batch_size=16, lr=1e-3, patience=5):
    M = build_feature_matrix(history)  # (T, F)

    # SCALE
    scaler = StandardScaler()
    M_scaled = scaler.fit_transform(M)

    # DATASET — predict total_expense at index 2
    dataset = TimeSeriesDataset(M_scaled, window_size=window, target_idx=2)

    n = len(dataset)
    if n < 20:
        raise ValueError(f"Not enough samples: got {n}, need > 20")

    # SPLIT
    split = int(0.8 * n)
    train_ds = torch.utils.data.Subset(dataset, range(split))
    val_ds = torch.utils.data.Subset(dataset, range(split, n))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # MODEL
    model = LSTMForecaster(input_dim=M.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_val = float("inf")
    best_state = None
    es_counter = 0
    print("Starting training...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X)
                val_loss += loss_fn(pred, y).item()

        train_avg = total_loss / len(train_loader)
        val_avg = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_avg:.4f}, Val Loss={val_avg:.4f}")

        scheduler.step(val_avg)

        if val_avg < best_val:
            best_val = val_avg
            best_state = model.state_dict()
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= patience:
                print("\nEarly stopping triggered!")
                break

    save_dir = os.path.join(os.path.dirname(__file__), "lstm")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "lstm_model.pt")
    scaler_path = os.path.join(save_dir, "scaler.pkl")

    torch.save(best_state or model.state_dict(), model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("\nTraining complete.")
    print(f"Best Validation Loss: {best_val:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, "training_data.json")
    history = load_history_from_json(json_path)
    train_model(history, window=6, epochs=50)
