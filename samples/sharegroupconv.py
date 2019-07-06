
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ShareGroupConv(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size, stride=1, padding=0, dialation=1,bias=False):
        super(ShareGroupConv, self).__init__()
        self.in_channel=in_channel
        self.out_channel = out_channel
        self.oralconv = nn.Conv2d(in_channel//out_channel, 1, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dialation, groups=1, bias=bias)

    def forward(self, x):
        out = None
        for j in range(0, self.out_channel):
            term = x[:,torch.arange(j,self.in_channel,step=self.out_channel),:,:]
            out_term = self.oralconv(term)

            out = torch.cat((out,out_term),dim=1) if not out is None else out_term
        return out


shareGroupConv = ShareGroupConv(6,2,kernel_size=3)
mm=torch.ones((3,2,8,8))
mn=2*torch.ones((3,2,8,8))
mb = 3*torch.ones((3,2,8,8))
mmm = torch.cat((mm,mn,mb),dim=1)
out = shareGroupConv(mmm)
print(out.size())
model_parameters = filter(lambda p: p.requires_grad, shareGroupConv.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)


print('__________________________')
x= torch.rand(1,1,3,3)
yy = nn.functional.interpolate(x,scale_factor=2,mode='nearest')
# yy = nn.functional.interpolate(x,size=5)
# yy = nn.functional.upsample(x,size=5,mode='nearest')
print(yy.size())
print(yy)