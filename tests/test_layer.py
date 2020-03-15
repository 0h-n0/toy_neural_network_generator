import warnings
import unittest

import pytest
import networkx as nx
import numpy as np

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

def test_empty_concat_case():
    m1 = MultiHeadLinkedListLayer()
    m1.append([1, 2, 3, 4, 5])
    m1.append([6, 7, 8])
    m1.append([10])
    m2 = MultiHeadLinkedListLayer()
    m1.append([10, 20, 30, 40, 50])
    m1.append([60, 70, 80])
    m = m1 + m2
    m.append([100, 200])

def test_two_haed_linked_list_layer_depth():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([6, 7, 8])
    m.append([10])
    nodelist = list(range(len(m.graph.nodes)))
    adj = nx.to_numpy_matrix(m.graph, nodelist=nodelist)
    expected_adj = np.array([[0., 1., 0.],
                             [0., 0., 1.],
                             [0., 0., 0.]])
    np.testing.assert_array_equal(adj, expected_adj)
    print(adj)
    assert len(m) == 3

def test_two_modal_two_haed_linked_list_layer_depth():
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

def test_two_haed_linked_list_lazylayer_depth():
    m = MultiHeadLinkedListLayer()
    args = [{f'arg_{i}': i for i in range(10)},]
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    nodelist = list(range(len(m.graph.nodes)))
    adj = nx.to_numpy_matrix(m.graph, nodelist=nodelist)
    expected_adj = np.array([[0., 1., 0.],
                             [0., 0., 1.],
                             [0., 0., 0.]])
    np.testing.assert_array_equal(adj, expected_adj)
    print(adj)
    assert len(m) == 3

def test_two_modal_multi_haed_linked_list_lazylayer_depth():
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
    nodelist = list(range(len(m.graph.nodes)))
    adj = nx.to_numpy_matrix(m.graph, nodelist=nodelist)
    expected_adj = np.array([[0., 1., 0., 0., 0., 0., 0., 0.,],
                             [0., 0., 1., 0., 0., 0., 0., 0.,],
                             [0., 0., 0., 0., 0., 0., 1., 0.,],
                             [0., 0., 0., 0., 1., 0., 0., 0.,],
                             [0., 0., 0., 0., 0., 1., 0., 0.,],
                             [0., 0., 0., 0., 0., 0., 1., 0.,],
                             [0., 0., 0., 0., 0., 0., 0., 1.,],
                             [0., 0., 0., 0., 0., 0., 0., 0.,]])
    np.testing.assert_array_equal(adj, expected_adj)
    assert len(m) == 5


def test_three_modal_multi_haed_linked_list_lazylayer_depth_2():
    m = MultiHeadLinkedListLayer()
    m1 = MultiHeadLinkedListLayer()
    args = [{f'arg_{i}': i for i in range(10)},]
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m1.append_lazy(Hoge, args)
    m1.append_lazy(Hoge, args)
    m = m1 + m # add concat layer
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    warnings.warn(f'{m.klass_set}')
    nodelist = list(range(len(m.graph.nodes)))
    adj = nx.to_numpy_matrix(m.graph, nodelist=nodelist)
    expected_adj = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    np.testing.assert_array_equal(adj, expected_adj)


def test_three_modal_multi_haed_linked_list_lazylayer_depth_2():
    m = MultiHeadLinkedListLayer()
    m1 = MultiHeadLinkedListLayer()
    m2 = MultiHeadLinkedListLayer()
    args = [{f'arg_{i}': i for i in range(10)},]
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m1.append_lazy(Hoge, args)
    m1.append_lazy(Hoge, args)
    m2.append_lazy(Hoge, args)
    m = m1 + m # add concat layer
    m = m2 + m # add concat layer
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    m.append_lazy(Hoge, args)
    warnings.warn(f'{m.klass_set}')
    nodelist = list(range(len(m.graph.nodes)))
    adj = nx.to_numpy_matrix(m.graph, nodelist=nodelist)
    expected_adj = np.array([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                             [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    np.testing.assert_array_equal(adj, expected_adj)
