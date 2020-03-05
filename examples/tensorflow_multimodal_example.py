#!/usr/bin/env python
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import Sequential
from tnng import Generator, MultiHeadLinkedListLayer

m = MultiHeadLinkedListLayer()
m.append([L.Dense(64), L.Dense(128), L.Dense(256), tf.identity])
m.append([L.ReLU(), L.ELU(), tf.identity])
m.append([L.Dense(16), L.Dense(32), L.Dense(64), tf.identity])
m.append([L.ReLU(), L.ELU(), tf.identity])

m1 = MultiHeadLinkedListLayer()
m1.append([L.Conv2D(64, 2), L.Conv2D(128, 2), tf.identity])
m1.append([L.ReLU(), L.ELU(), tf.identity])
m1.append([L.Conv2D(256, 2), L.Conv2D(256, 1), tf.identity])
m1.append([L.ReLU(), L.ELU(), tf.identity])
m1.append([L.Conv2D(256, 2), L.Conv2D(256, 1), tf.identity])
m1.append([L.ReLU(), L.ELU(), tf.identity])
m1.append([L.Conv2D(256, 2), L.Conv2D(256, 1), tf.identity])
m1.append([L.ReLU(), L.ELU(), tf.identity])
m1.append([L.Flatten()])

m = m + m1
m.append([L.Dense(10)])
g = Generator(m)

x1 = tf.compat.v1.placeholder(tf.float32, shape=(None, 32))
x2 = tf.compat.v1.placeholder(tf.float32, shape=(None, 28, 28, 3))

def build(x, img, idx):
    for m in g[idx]:
        if len(m) == 2:
            if m[0] is not None:
                x = m[0](x)
            img = m[1](img)
        elif len(m) == 1 and m[0] is None:
            x = tf.concat((x, img), 1)
        else:
            x = m[0](x)
    return x

y = build(x1, x2, 0)
