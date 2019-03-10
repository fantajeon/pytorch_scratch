
import pdb
import time
import numpy as np


import torch
from torch import nn
import torch.optim as optim
import torch.multiprocessing as mp

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

class MySubModel(nn.Module):
    def __init__(self):
        super(MySubModel, self).__init__()
        self.fc = nn.Linear(2, 1)
        self.fc.weight.data = norm_col_init( self.fc.weight.data, 0.01)
        self.fc.bias.data.fill_(0)

    def forward(self, inputs):
        return self.fc(inputs)

    def print_params(self):
        print("fc.weight", self.fc.weight.data )
        print("fc.bias", self.fc.bias.data )


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embed = MySubModel()

    def forward(self, inputs):
        return self.embed(inputs)

    def print_params(self):
        self.embed.print_params()

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

def test(q, shared_model):
    mymodel = MyModel()
    print ("test ========================= ")
    mymodel.print_params()
    data = q.get()
    print("recv_q", data)
    mymodel.load_state_dict(shared_model.state_dict())

    print ("load state ======================")
    mymodel.print_params()


if __name__ == "__main__":
    model = MyModel()
    model.share_memory()

    q = mp.Queue(maxsize=1)
    p = mp.Process(target=test, args=(q,model,))
    p.start()

    w1 = 1.0
    w2 = 0.8
    W = torch.from_numpy(np.array( [[w1,w2]], dtype=np.float32 ))
    b = 0.3

    optimizer = optim.Adam( model.parameters(), lr=0.01 )

    test_data = []
    for _ in range(10000):
        x = np.random.rand(1,2)
        x = torch.from_numpy(x).type(torch.float32)
        y = torch.matmul(W, torch.transpose(x, 0,1)) + b
        #pdb.set_trace()
        loss = (y - model(x)).pow(2).mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()
    print ("main process weight of model======================")
    model.print_params()
    print ("push model to queie")
    q.put( [W, b] )
    p.join()
