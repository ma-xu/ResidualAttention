import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Myconv(nn.Module):
    def __init__(self):
        super(Myconv, self).__init__()
        self.conv=nn.Conv2d(3, 64, 3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True)

    def forward(self, x):
        return self.conv(x)
    


myconv = Myconv()
model_parameters = filter(lambda p: p.requires_grad, myconv.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)