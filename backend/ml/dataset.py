import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    """
    Sequence dataset for LSTM:
    input: window of size W
    target: next-month expense (unscaled or scaled)
    """

    def __init__(self, feature_matrix, window_size, target_index=2):
        """
        feature_matrix: numpy array (T, F)
        window_size: number of timesteps used as input
        target_index: index of feature to predict (default = total_expense)
        """
        self.X = feature_matrix  # (T, F)
        self.window_size = window_size
        self.target_index = target_index

        self.data = []
        self.labels = []

        for i in range(len(feature_matrix) - window_size):
            window = feature_matrix[i:i + window_size]
            target = feature_matrix[i + window_size, target_index]

            self.data.append(window)
            self.labels.append(target)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])
