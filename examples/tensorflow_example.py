#!/usr/bin/env python
import tensorflow.keras.layers as L
from tensorflow.keras import Sequential
from tnng import Generator, MultiHeadLinkedListLayer

m = MultiHeadLinkedListLayer()
m.append([L.Dense(64), L.Dense(128), L.Dense(256)])
m.append([L.ReLU(), L.ELU()])
m.append([L.Dense(16), L.Dense(32), L.Dense(64),])
m.append([L.ReLU(), L.ELU()])
m.append([L.Dense(10)])

g = Generator(m)

print(f"the number of generated models: {len(g)}")

model = Sequential()
for l in g[0]:
    # returned layer is list type.
    model.add(l[0])

model = Sequential()
for l in g[2]:
    # returned layer is list type.
    model.add(l[0])
