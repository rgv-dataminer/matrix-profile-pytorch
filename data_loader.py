import numpy as np
import torch

def load_univariate_ts(filename, dir='dataset/'):
    """
    Load a time series from a text file.

    Input:
        filename: Name of the time-series file.
        dir: Directory containing the file.

    Output:
        ts: PyTorch tensor of shape (1, sequence_length).
    """
    ts = np.loadtxt(dir + filename)
    ts = torch.tensor(ts).unsqueeze(0).float()
    return ts

def extract_subsequence(ts, ls=100):
    """
    Extract all subsequences of a given length from a time series.

    Input:
        ts: PyTorch tensor of shape (C, sequence_length).
        ls: Length of each subsequence.

    Output:
        windows: PyTorch tensor of shape (num_subsequences, C, ls).
    """
    windows = ts.unfold(dimension=1, size=ls, step=1)
    windows = windows.permute(1, 0, 2)
    return windows


def subseqeunce_dataloader(ts, ls=100, batch_size=32, shuffle=False):
    """
    Create a DataLoader for time-series subsequences.

    Input:
        ts: PyTorch tensor of shape (C, sequence_length).
        ls: Length of each subsequence.
        batch_size: Number of subsequences per batch. (default 32)
        shuffle: Whether to shuffle the subsequences.

    Output:
        window_loader: DataLoader yielding batches of subsequences and
                       their corresponding indices.
    """
    windows = extract_subsequence(ts, ls)
    num_windows = windows.shape[0]

    window_indices = torch.arange(num_windows)

    dataset = TensorDataset(windows, window_indices)

    window_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    return window_loader
