import torch
import torch.nn.functional as F

x= torch.round(10*torch.rand(2,3,4,5))
print(x)
print("x shape:"+str(x.size()))
y = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
print(y)
print("y shape:"+str(y.size()))