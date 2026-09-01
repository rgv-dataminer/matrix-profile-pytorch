import torch
import numpy as np
from matrix_profile_torch import query_base

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
