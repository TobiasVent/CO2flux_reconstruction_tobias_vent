
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple


Sample = namedtuple("Sample", ["features", "target", "meta"])


    
    
class PointwiseSampleDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.X = torch.tensor(np.array([s.features for s in samples]), dtype=torch.float32)
        self.y = torch.tensor(np.array([s.target for s in samples]), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class PointwiseSampleDatasetMonth(torch.utils.data.Dataset):
    def __init__(self, X, y, meta=None):
        """
        Args:
            X: numpy array of shape (N, window_size, num_features)
            y: numpy array of shape (N, 1)
            meta: optional list of dicts (length N) with metadata
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.meta = meta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.meta is not None:
            return self.X[idx], self.y[idx], self.meta[idx]
        else:
            return self.X[idx], self.y[idx]
        
class PointwiseSampleDatasetMonthMLP(torch.utils.data.Dataset):
    def __init__(self, X, y, meta=None):
        """
        Args:
            X: numpy array of shape (N, window_size, num_features)
            y: numpy array of shape (N, 1)
            meta: optional list of dicts (length N) with metadata
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # shape: (N, window_size, num_features)
        self.y = torch.tensor(y, dtype=torch.float32)  # shape: (N, 1)
        self.meta = meta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].flatten()  # flatten only the single sample
        y = self.y[idx]
        if self.meta is not None:
            return x, y, self.meta[idx]
        else:
            return x, y
        

class PointwiseSampleDatasetMonthMLPWithPos(torch.utils.data.Dataset):
    def __init__(self, X, y, meta=None):
        """
        Args:
            X: numpy array of shape (N, window_size, num_features)
            y: numpy array of shape (N, 1)
            meta: optional list of dicts (length N) with metadata
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # shape: (N, window_size, num_features)
        self.y = torch.tensor(y, dtype=torch.float32)  # shape: (N, 1)
        self.meta = meta
    def flatten_sample(self, x, n_dyn=10, n_static=4, window_size=4):
         # shape: (4, 14)
        x_dyn = x[:, :n_dyn].flatten()   # 4×10 = 40
        x_static = x[0, n_dyn:]          # 4 static features
        return np.concatenate([x_dyn, x_static])
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.flatten_sample(self.X[idx])
        #x = self.X[idx].flatten()  # flatten only the single sample
        y = self.y[idx]
        if self.meta is not None:
            return x, y, self.meta[idx]
        else:
            return x, y

class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.tensor(sample.features, dtype=torch.float32).flatten()
        y = torch.tensor(sample.target, dtype=torch.float32)
        meta = sample.meta  # e.g., {'nav_lat': ..., 'nav_lon': ..., 'time_counter': ...}
        return x, y


class MLPDatasetWithPos(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    def flatten_sample(self, sample, n_dyn=10, n_static=4, window_size=4):
        x = sample.features  # shape: (4, 14)
        x_dyn = x[:, :n_dyn].flatten()   # 4×10 = 40
        x_static = x[0, n_dyn:]          # 4 static features
        return np.concatenate([x_dyn, x_static])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.tensor(self.flatten_sample(sample), dtype=torch.float32)
        y = torch.tensor(sample.target, dtype=torch.float32)
        # x = torch.tensor(sample.features, dtype=torch.float32).flatten()
        # y = torch.tensor(sample.target, dtype=torch.float32)
        meta = sample.meta  # e.g., {'nav_lat': ..., 'nav_lon': ..., 'time_counter': ...}
        return x, y