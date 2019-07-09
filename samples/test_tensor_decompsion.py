import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac,robust_pca, tucker   #CPdecomposition
import torch
import torch.nn as nn
tl.set_backend('pytorch')


class DeSELayer(nn.Module):
    def __init__(self,channel,reduction=16,rank=4):
        super(DeSELayer, self).__init__()
        self.rank = rank
        self.rank_fc = nn.Sequential(
            nn.Linear(rank, rank, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rank,1,bias=False)
        )
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y=None
        b, c, _, _ = x.size()
        for i in range(0, x.size()[0]):
            x_item = x[i, :, :, :]
            factors = parafac(x_item, rank=self.rank)
            y = torch.cat((y,factors[0].unsqueeze(0)),dim=0) if not y is None else factors[0].unsqueeze(0)

        y = self.rank_fc(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

dese = DeSELayer(channel=64,rank = 3)
X=torch.rand(10,64,4,4)
y = dese(X)
print(y.size())



# X = torch.rand(10,3,4,5)
# for i in range(0,X.size()[0]):
#     x_item = X[i,:,:,:]
#     factors = parafac(X, rank=2)
#
# factors = parafac(X,rank=2)
# print(len(factors))
# print(factors[0].size())






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

# X = torch.randn( 4,7,9, dtype=torch.double)
# D=parafac(X,rank=2)
# print(len(D))
# print(D[0])


#
# X = torch.randn( 4,7,9, dtype=torch.double)
# core, factors = tucker(X, ranks=4)
# print(core.size())
# print(len(factors))