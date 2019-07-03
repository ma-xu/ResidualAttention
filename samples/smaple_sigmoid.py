import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



xx=torch.range(-5,5,step=0.01)
alpha=2
y1= torch.sigmoid(xx)
y2 = torch.sigmoid(alpha*xx)
y3 = F.logsigmoid(xx)

# y3 = F.softmax(xx)

plt.close('all')

plt.scatter(xx, y1)
plt.show()

plt.scatter(xx, y2)
plt.show()

plt.scatter(xx, y3)
plt.show()