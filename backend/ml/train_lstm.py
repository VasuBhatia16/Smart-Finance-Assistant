import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

from ml.model import LSTMForecaster
from ml.dataset import TimeSeriesDataset
from ml.preprocess import build_feature_matrix
import json
import os


def load_history_from_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
    
def train_model(history, window=6, epochs=50, batch_size=16, lr=1e-3, patience=5):
    """
    history: list of dicts (same structure as backend input)
    window: number of months used as input
    """
    M = build_feature_matrix(history)  # (T, F)

    # === SCALE FEATURES ===
    scaler = StandardScaler()
    M_scaled = scaler.fit_transform(M)

    # === CREATE DATASET ===
    dataset = TimeSeriesDataset(M_scaled, window_size=window)

    n = len(dataset)
    if n < 20:
        raise ValueError(f"Not enough samples: got {n}, need > 20")

    # === TRAIN/VAL SPLIT ===
    split = int(0.8 * n)
    train_ds = torch.utils.data.Subset(dataset, range(split))
    val_ds = torch.utils.data.Subset(dataset, range(split, n))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # === MODEL ===
    model = LSTMForecaster(input_dim=M.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # === EARLY STOPPING ===
    best_val_loss = float("inf")
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

            # === Prevent gradient explosion ===
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X)
                val_loss += loss_fn(pred, y).item()

        train_loss_avg = total_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss_avg:.4f}, Val Loss={val_loss_avg:.4f}")

        # === LR Scheduler ===
        scheduler.step(val_loss_avg)

        # === Early stopping logic ===
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_state = model.state_dict()
            es_counter = 0
        else:
            es_counter += 1

        if es_counter >= patience:
            print("\nEarly stopping triggered!")
            break

    # === Save final model (best) ===
    if best_state:
        torch.save(best_state, "lstm_model.pt")
    else:
        torch.save(model.state_dict(), "lstm_model.pt")

    # === Save scaler ===
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nTraining complete.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("Model saved to lstm_model.pt, scaler saved to scaler.pkl.")



if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, "training_data.json")
    history = load_history_from_json(json_path)
    train_model(history, window=6, epochs=50)
