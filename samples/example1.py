import torch
import torch.nn.functional as F
import torch.nn as nn
nn.LayerNorm

x = torch.rand(1,1,2,2)
x1 = torch.sigmoid(x)
x2 = F.softmax(x)
print('______x________')
print(x)
print('______x1________')
print(x1)
print('______x2________')
print(x2)

# x= torch.round(10*torch.rand(2,3,4,5))
# print(x)
# print("x shape:"+str(x.size()))
# y = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
# print(y)
# print("y shape:"+str(y.size()))