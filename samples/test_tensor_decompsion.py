import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac   #CPdecomposition

X = np.random.rand(3,4,5)
factors = non_negative_parafac(X, rank=1)
print(len(factors))
print([f.shape for f in factors])
print(factors[0])