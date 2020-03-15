#!/usr/bin/env python
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import Sequential
from tnng import Generator, MultiHeadLinkedListLayer

m2 = MultiHeadLinkedListLayer()
m2.append_lazy(L.Flatten, [dict(input_shape=(28, 28)),])
m2.append_lazy(L.Dense, [dict(units=32), dict(units=64), dict(units=128)])
m2.append_lazy(L.ReLU, [dict(),])
m2.append_lazy(L.Dense, [dict(units=16), dict(units=32), dict(units=64)])
m2.append_lazy(L.ReLU, [dict(),])


m1 = MultiHeadLinkedListLayer()
m1.append_lazy(L.Flatten, [dict(input_shape=(28, 28)),])

m = m1 + m2

m.append_lazy(L.Dense, [dict(units=10),])
m.append_lazy(L.Dense, [dict(units=10),])
m.append_lazy(L.Dense, [dict(units=10),])

g = Generator(m)
g.draw_graph('/home/ono/Dropbox/tensorflow_multi_modal.png')

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
