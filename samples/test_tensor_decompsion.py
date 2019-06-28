import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac,robust_pca   #CPdecomposition
import torch
tl.set_backend('pytorch')


## Numpy backend
# X = np.random.rand(3,4,5)
# factors = non_negative_parafac(X, rank=1)
# print(len(factors))
# print([f.shape for f in factors])
# print(factors[0])

## Pytorch backend
# X = torch.randn(2,3, 7,9, dtype=torch.double)
# factors = non_negative_parafac(X, rank=4)
# print(len(factors))
# AA=factors[0]
# print(AA.size())

X = torch.randn(3, 7,9, dtype=torch.double)
D,E = robust_pca(X)
print(D)
print(E)
