import warnings
import unittest

import pytest

from tnng.layer import (BaseLayer,
                        Layer,
                        LazyLayer,
                        MultiHeadLinkedListLayer)


class Hoge:
    def __init__(self, args):
        self.args = args
    def __call__(self):
        return self.args


def test_base_layer_property():
    p = BaseLayer()
    c = BaseLayer(p)
    assert c.parent is p
    assert p.parent is None
    assert c.child is None
    assert p.child is None


def test_layer_property():
    p = Layer(layers=[1, 2])
    c = Layer(layers=[1, 2], parent=[p,])
    assert c.parent[0] is p
    assert p.parent is None
    assert c.child is None
    assert p.child is None


def test_lazylayer_property():

    p = LazyLayer()
    c = LazyLayer(Hoge, [{'args': 1}, {'args': 2}], parent=[p,])
    assert c.parent[0] is p
    assert p.parent is None
    assert c.child is None
    assert p.child is None


def test_multi_haed_linked_list_layer_depth():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([6, 7, 8])
    m.append([10])
    assert len(m) == 3

def test_multi_modal_multi_haed_linked_list_layer_depth():
    m = MultiHeadLinkedListLayer()
    m1 = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([6, 7, 8])
    m.append([10])
    m1.append([1, 2, 3, 4, 5])
    m1.append([6, 7, 8])
    m1.append([10])
    m = m1 + m # add concat layer
    m.append([10, 2])
    assert len(m) == 5

def test_multi_haed_linked_list_lazylayer_depth():
    m = MultiHeadLinkedListLayer()
    args = [{f'arg_{i}': i for i in range(10)},]
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    assert len(m) == 3

def test_multi_modal_multi_haed_linked_list_lazylayer_depth():
    m = MultiHeadLinkedListLayer()
    m1 = MultiHeadLinkedListLayer()
    args = [{f'arg_{i}': i for i in range(10)},]
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m1.append_lazy(Hoge, args)
    m1.append_lazy(Hoge, args)
    m1.append_lazy(Hoge, args)
    m = m1 + m # add concat layer
    m.append_lazy(Hoge, args)
    warnings.warn(f'{m.klass_set}')
    assert len(m) == 5
