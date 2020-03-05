![Github CI/CD](https://github.com/0h-n0/toy_neural_network_generator/workflows/Github%20CI/CD/badge.svg?branch=master)

# Toy Neural Network Generator

## Installation

```shell
$ pip install tnng
```

## Simple Model Generator

```python
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
```

## Multimodal Model Generator

```python
#!/usr/bin/env python
import torch
import torch.nn as nn
import torchex.nn as exnn
from tnng import Generator, MultiHeadLinkedListLayer

m = MultiHeadLinkedListLayer()
m1 = MultiHeadLinkedListLayer()
# all layers can be lazy evaluation.
m.append([exnn.Linear(64), exnn.Linear(128), exnn.Linear(256)])
m.append([nn.ReLU(), nn.ELU()])
m.append([exnn.Linear(16), exnn.Linear(32), exnn.Linear(64),])
m.append([nn.ReLU(), nn.ELU()])

m1.append([exnn.Conv2d(16, 1), exnn.Conv2d(32, 1), exnn.Conv2d(64, 1)])
m1.append([nn.MaxPool2d(2), nn.AvgPool2d(2)])
m1.append([nn.ReLU(), nn.ELU(), nn.Identity()])
m1.append([exnn.Conv2d(32, 1), exnn.Conv2d(64, 1), exnn.Conv2d(128, 1)])
m1.append([nn.MaxPool2d(2), nn.AvgPool2d(2)])
m1.append([exnn.Flatten(),])

m = m + m1
m.append([exnn.Linear(128)])
m.append([nn.ReLU(), nn.ELU(), nn.Identity()])
m.append([exnn.Linear(10)])


g = Generator(m)
class Model(nn.Module):
    def __init__(self, idx=0):
        super(Model, self).__init__()
        self.model = g[idx]
        for layers in self.model:
            for layer in layers:
                self.add_module(f'{layer}', layer)

    def forward(self, x, img):
        for m in self.model:
            if len(m) == 2:
                if m[0] is not None:
                    x = m[0](x)
                img = m[1](img)
            elif len(m) == 1 and m[0] is None:
                x = torch.cat((x, img), 1)
            else:
                x = m[0](x)
        return x

x = torch.randn(128, 256)
img = torch.randn(128, 3, 28, 28)
m = Model()
o = m(x, img)
print(o.shape)
```
