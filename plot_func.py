import matplotlib.pyplot as plt
import torch

def plot_pair_motif(ts, score,idx_mp, ls=100, dim=0):
    idx = torch.argmax(score)
    plot_finding(ts, idx, idx_mp[idx], ls=ls, dim=dim)


def plot_finding(ts, q, a, ls=100, dim=0):
  plt.figure(figsize=(15, 4))
  target = ts[dim,:]
  if torch.is_tensor(ts):
    target = target.squeeze().cpu().detach().numpy()
  plt.plot(target, label="Reference signal", color="C0")
  query = target[q:q + ls]
  signal = target[a:a + ls]
  
  plt.plot(np.arange(q, q + ls), query, linewidth=2, color='red',label="Matched subsequence")
  plt.plot(np.arange(a, a + ls), signal, linewidth=2, color='green',label="Matched subsequence")
  plt.title("Reference Signal with Matched Subsequence")
  plt.xlabel("Time Index")
  plt.ylabel("Amplitude")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()
  
  offset = 0.8 * (signal.max() - signal.min())

  plt.figure(figsize=(4, 4))
  plt.plot(signal, label="Extracted segment", color="C0")
  plt.plot(query + offset, label="Template", color="red")

  plt.title("Template vs Extracted Segment")
  plt.xlabel("Sample Index")
  plt.ylabel("Amplitude")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()
