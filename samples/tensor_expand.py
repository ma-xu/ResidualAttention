import torch
import torch.nn.functional as F

x1 = torch.ones(2,1,4,4)
x2 = 2*torch.ones(2,1,4,4)
x3 = 3*torch.ones(2,1,4,4)
x = torch.cat((x1,x2,x3),dim=1)
print(x.size())
print(x)


xx = x.view(-1,1,1,1).repeat(1,2,1,1).view(2,6,4,4)
print('___________')
print(xx)
print(xx.size())