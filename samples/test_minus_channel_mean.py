import torch
import torch.nn as nn

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.mean(x,1).unsqueeze(1)


x= torch.rand(3,4,5,5)
channelPool = ChannelPool()
x_channel = channelPool(x)
y=x-x_channel
print(y)
gap = nn.AdaptiveAvgPool2d(1)
Ay = gap(y)
Ax = gap(x)
print('--------------------------')
print(Ay)
print('--------------------------')
print(Ax)
print('--------------------------')
print(Ax-Ay)
