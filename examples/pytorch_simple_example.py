#!/usr/bin/env python
import torch
import torch.nn as nn
import torchex.nn as exnn
from tnng import Generator, MultiHeadLinkedListLayer

m = MultiHeadLinkedListLayer()
# all layers can be lazy evaluation.
m.append([exnn.Linear(64), exnn.Linear(128), exnn.Linear(256)])
m.append([nn.ReLU(), nn.ELU()])
m.append([exnn.Linear(16), exnn.Linear(32), exnn.Linear(64),])
m.append([nn.ReLU(), nn.ELU()])
m.append([exnn.Linear(10)])

g = Generator(m)

x = torch.randn(128, 256)

class Model(nn.Module):
    def __init__(self, idx=0):
        super(Model, self).__init__()
        self.model = nn.ModuleList([l[0] for l in g[idx]])

    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x

m = Model(0)
o = m(x)

'''
ModuleList(
  (0): Linear(in_features=256, out_features=64, bias=True)
  (1): ReLU()
  (2): Linear(in_features=64, out_features=16, bias=True)
  (3): ReLU()
  (4): Linear(in_features=16, out_features=10, bias=True)
)
'''
