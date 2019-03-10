

import torch


pos = torch.rand(2,3)
neg = torch.rand(5,3)

# neg: 1 x 5 x 3
# pos: 2 x 1 x 3
c = torch.unsqueeze(pos, dim=1) * torch.unsqueeze(neg,dim=0)
c = c.sum(dim=2)
print(c.shape, c.sum())
