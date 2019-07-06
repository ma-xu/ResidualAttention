import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Myconv(nn.Module):
    def __init__(self):
        super(Myconv, self).__init__()
        self.conv=nn.Conv2d(64, 128, 3, stride=1,
                 padding=0, dilation=1, groups=4, bias=True)


    def forward(self, x):
        return self.conv(x)
    


myconv = Myconv()
model_parameters = filter(lambda p: p.requires_grad, myconv.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

print("---------------")
class Hisconv(nn.Module):
    def __init__(self):
        super(Hisconv, self).__init__()
        self.conv=nn.ConvTranspose2d(64,1,kernel_size=7,stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1)


    def forward(self, x):
        return self.conv(x)

hisconv= Hisconv()
y = hisconv((torch.randn(1,64,1,1)))
print(y.size())



print("---------------")


class Groupconv(nn.Module):
    def __init__(self):
        super(Groupconv, self).__init__()
        self.conv = nn.Conv2d(30, 10, 1, stride=1,
                              padding=0, dilation=1, groups=10, bias=False)

    def forward(self, x):
        return self.conv(x)


groupconv = Groupconv()
model_parameters = filter(lambda p: p.requires_grad, groupconv.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

for i in range(0,3):
    print(i)



x=torch.rand(2,64,4,5)
y0 = x[:,torch.arange(0, 64, step=3),:,:]
y1 = x[:,torch.arange(1, 64, step=3) ,:,:]
y2 = x[:,torch.arange(2, 64, step=3),:,:]
y = torch.cat((y0,y1,y2),1)

print(y)
print(y.size())
print(torch.equal(x,y))




