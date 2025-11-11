import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (B, W, F)
        returns: (B,) predicted expense
        """
        out, _ = self.lstm(x)           # (B, W, H)
        last = out[:, -1, :]            # (B, H)
        out = self.fc(last)             # (B, 1)
        return out.squeeze(-1)
