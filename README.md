# matrix-profile-pytorch
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SUWcAMumVsggoWrZ6Fv2ndF0sfMLTzVG?usp=sharing)

This is a fast and simplified implementation of matrix profile with pytorch back-end. 

The implementation does not follow any existing matrix profile algorithm. Instead, its implement a simple brute-force based solution that performing fast dot product that support both GPU and CPU setting. The implementation is easy to use, significant fast than existing package, and can controllable memory usage via batch configuration. 

The code can simply run matrix profile with pytorch backend via:

```python
from matrix_profile_torch import matrix_profile
import torch
ts = torch.randn(1,1000)
mp, mpi = matrix_profile(ts,ls=100)
```

Once matrix profile and index profile is obtained. The code support quick pair motif visualization

```python
plot_pair_motif(ts,mp,mpi,ls=100)
```

A demo example with TEK time series data is included in the provided Google Colab Link. The code will produce the pair motif visualization shown below.
