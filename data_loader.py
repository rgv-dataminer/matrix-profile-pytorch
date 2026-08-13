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
