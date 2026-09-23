import torch
import numpy as np
from matrix_profile_torch import query_base

def find_top_k_instances(dist,top_k=4, l=700, sample_subsequence_start=0):
  # Input: dist 1 x L size distance profile sequence. 
  #
  # Can we find top-4 patterns that is located in different region (non-overlapping)
  # return the indexes of the pattern
  #min_separation = ls
  if len(dist.shape) > 1:
    dist = dist[0, :]
  # Add infinity to both ends to ensure edges are considered in local minima
  inf = torch.tensor([float('inf')], device=dist.device)
  dist_padded = torch.cat([inf, dist, inf])


  # Find all local minima
  is_min = (dist_padded[1:-1] < dist_padded[:-2]) & (dist_padded[1:-1] < dist_padded[2:])
  dip_indices = torch.where(is_min)[0] + 1

  # Exclude dips too close to the target subsequence
  far_from_target = (
      torch.abs(dip_indices - sample_subsequence_start) >= l
  )

  dip_indices = dip_indices[far_from_target]
  # Sort remaining dips from lowest distance to highest
  order = torch.argsort(dist[dip_indices])
  sorted_dips = dip_indices[order]

  # Keep lowest dip within each min_separation neighborhood
  kept_dips = []

  for idx in sorted_dips:
      if all(torch.abs(idx - kept) >= l for kept in kept_dips):
          kept_dips.append(idx)

  kept_dips = torch.stack(kept_dips)

  if len(kept_dips) >= top_k:
    return kept_dips[:top_k]
  else:
    print(f"Not enough similarities, only {len(kept_dips)}")
    return kept_dips


def find_instances(dist_profile,l=100):
    dist_profile = dist_profile.squeeze().unsqueeze(dim=1)
    ts_unfold = dist_profile.unfold(0,2*l,1)
    a,b= ts_unfold.min(dim=2)
    linear = torch.arange(b.shape[0])
    c = b + linear.unsqueeze(dim=1)
    candidates = c.unique()

    A = torch.cat((a,c),dim=1)
    sorted_array = A[A[:, 0].argsort()]
    aa = sorted_array[:,1]
    index = aa[torch.cat((torch.tensor([True]), aa[1:] != aa[:-1]))]
    dist = dist_profile[index.long()]
    index = index.long()
    return dist, index

def find_motif_instance(ts, idx, l=100, qfunc=None):
    s = ts[:,idx:idx+l]
    if qfunc is None:
        qfunc = query_base
    dist_profile = qfunc(ts, s.unsqueeze(dim=0))
    dist_profile = 2- 2*dist_profile
    dist, loc = find_instances(dist_profile,l=l)
    return dist, loc 
