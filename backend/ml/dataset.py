# backend/ml/dataset.py
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, M_scaled, window_size=6, target_idx=2):
        """
        M_scaled: numpy array (T, F)
        window_size: number of timesteps in each input window
        target_idx: which feature index to predict (2 = total_exp per preprocess.py)
        """
        self.M = M_scaled
        self.W = window_size
        self.target_idx = target_idx

        if self.M.shape[0] < self.W:
            raise ValueError(f"Not enough rows: {self.M.shape[0]} < window_size {self.W}")

    def __len__(self):
        return self.M.shape[0] - self.W + 1

    def __getitem__(self, i):
        # X window
        x = self.M[i:i+self.W, :]                 # (W, F)
        # y = target at the last timestep of the window
        y = self.M[i+self.W-1, self.target_idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y
