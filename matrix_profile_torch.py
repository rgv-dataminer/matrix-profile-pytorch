from data_loader import *
import torch
import numpy as np

def query_base_euclidean(ts, s):
    """
    Args:
        ts: (C, T)
        s:  (N, C, L)

    Returns:
        euclidean_dist: (N, num_windows)
    """
    # (num_windows, C, L)
    windows = ts.unfold(dimension=1, size=s.shape[-1], step=1)
    windows = windows.permute(1, 0, 2)

    # L2 normalize
    windows = windows / torch.norm(
        windows, p=2, dim=(1, 2), keepdim=True
    ).clamp_min(1e-8)

    s = s / torch.norm(
        s, p=2, dim=(1, 2), keepdim=True
    ).clamp_min(1e-8)

    # Compute squared Euclidean distance
    dot = torch.einsum("ice,nce->ni", windows, s)

    dist2 = (2 - 2 * dot).clamp_min(0)

    return dist2

def query_base(ts, s):
    # Input:
    # ts: time series data: c x l 
    # s: a set of subsequence: n x c x ls 
    # Output:
    # cosine_max: n  (highest cosine similarity value)
    # idx: n (index of highest cosine similarity)

    windows = ts.unfold(dimension=1, size=s.shape[-1], step=1)
    windows = windows.permute(1, 0, 2)

    numerator = torch.einsum("ice,nce->ni", windows, s)

    denominator = (
        torch.norm(windows, dim=(1, 2)).unsqueeze(0)
        * torch.norm(s, dim=(1, 2)).unsqueeze(1)
    ).clamp_min(1e-8)

    cosine_sim = numerator / denominator

    return cosine_sim
  
def create_batch_mask_v2(batch_idx, ts_len=1000, ls=10):
    if len(batch_idx.shape)==1:
      batch_idx.unsqueeze(0)
    idx = torch.arange(ts_len).unsqueeze(1)
    return torch.abs(idx-batch_idx).T < ls


def matrix_profile(ts, ls, batch_size=32, mode='c'):
    """
    Compute the matrix profile of a time series.

    Input:
        ts: PyTorch tensor of shape (1, sequence_length).
        ls: Length of each subsequence.
        batch_size: Number of subsequences processed per batch (default 32).
        mode: Distance/similarity measure to use:
              'c' for cosine similarity, 'e' for Euclidean distance.

    Output:
        all_score: PyTorch tensor containing the best match score for
                   each subsequence.
        all_idx: PyTorch tensor containing the index of the best match
                 for each subsequence.
    """
    
    all_score = []
    all_idx = []
    window_loader = subseqeunce_dataloader(ts, ls=ls, batch_size=batch_size)
    for window_batch, idx_batch in window_loader:
        window_batch = window_batch.to(ts.device)
        idx_batch = idx_batch.to(ts.device)
        if mode=='c':
          cos_sim = query_base(ts, window_batch)
          mask = create_batch_mask_v2(idx_batch,ts_len=cos_sim.shape[1],ls=window_batch.shape[2])
          cos_sim[mask] = -10000
          score, idx = cos_sim.max(dim=1)
        elif mode=='e':
          dist = query_base_euclidean(ts, window_batch)
          mask = create_batch_mask_v2(idx_batch,ts_len=dist.shape[1],ls=window_batch.shape[2])
          dist[mask] = 10000
          score, idx = dist.min(dim=1)
        
        # save all batch result
        all_score.append(score.cpu())
        all_idx.append(idx.cpu())

    # Concatenate results from all batches
    all_score = torch.cat(all_score, dim=0)
    all_idx = torch.cat(all_idx, dim=0)
    return all_score, all_idx
