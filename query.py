import torch
import numpy as np

def get_motif_instances(dist_profile,l=100):
    ts_unfold = dist_profile.unfold(0,2*l,1)
    a,b= ts_unfold.min(dim=2)
    linear = torch.arange(b.shape[0])
    c = b + linear.unsqueeze(dim=1)
    candidates = c.unique()

    A = torch.cat((a,c),dim=1)
    sorted_array = A[A[:, 0].argsort()]
    aa = sorted_array[:,1]
    index = aa[torch.cat((torch.tensor([True]), aa[1:] != aa[:-1]))]
    dist = ts[index.long()]
    return dist, index
