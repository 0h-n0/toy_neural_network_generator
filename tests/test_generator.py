import unittest

import pytest
import numpy as np

from tnng import MultiHeadLinkedListLayer, Generator
from tnng.layer import DummyConcat


try:
    import tensorflow
    import tensorflow.keras as keras
    exist_tensorflow = True
except:
    exist_tensorflow = False

try:
    import torch
    exist_torch = True
except:
    exist_torch = False




class Hoge1:
    def __init__(self, units):
        self.units = units
    def __call__(self, x):
        pass

class Hoge2:
    def __init__(self, units):
        self.units = units
    def __call__(self, x):
        pass

class Hoge3:
    def __init__(self, units):
        self.units = units
    def __call__(self, x):
        pass

class Hoge4:
    def __init__(self, units):
        self.units = units
    def __call__(self, x):
        pass


def test_generator_length():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([6, 7, 8])
    m.append([10])
    g = Generator(m)
    assert len(g) == 15

def test_multi_modal_generator_length():
    m1 = MultiHeadLinkedListLayer()
    m1.append([1, 2, 3, 4, 5])
    m1.append([6, 7, 8])
    m1.append([10])
    m2 = MultiHeadLinkedListLayer()
    m1.append([10, 20, 30, 40, 50])
    m1.append([60, 70, 80])
    m = m1 + m2
    m.append([100, 200])
    g = Generator(m)
    assert len(g) == 450

@pytest.fixture(
    params=[
        [0, [0, 0, 0]],
        [1, [1, 0, 0]],
        [2, [2, 0, 0]],
        [3, [3, 0, 0]],
        [4, [4, 0, 0]],
        [5, [0, 1, 0]],
        [14, [4, 2, 0]],
        [-1, [4, 2, 0]],
    ]
)
def index_get_layer_index_list_test_case(request):
    input = request.param[0]
    expected = request.param[1]
    return input, expected

def test_get_layer_index_list(index_get_layer_index_list_test_case):
    input = index_get_layer_index_list_test_case[0]
    expected = index_get_layer_index_list_test_case[1]
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([6, 7, 8])
    m.append([10])
    g = Generator(m)
    assert g._get_layer_index_list(input, g._node_type_layers) == expected


@pytest.fixture(
    params=[
        [0, [0, 0, 0, 0, 0, 0, 0]],
        [1, [1, 0, 0, 0, 0, 0, 0]],
        [2, [2, 0, 0, 0, 0, 0, 0]],
        [3, [3, 0, 0, 0, 0, 0, 0]],
        [4, [4, 0, 0, 0, 0, 0, 0]],
        [5, [0, 1, 0, 0, 0, 0, 0]],
        [449, [4, 2, 0, 4, 2, 0, 1]],
        [-1, [4, 2, 0, 4, 2, 0, 1]],
        [-2, [3, 2, 0, 4, 2, 0, 1]],
    ]
)
def index_multimodal_get_layer_index_list_test_case(request):
    input = request.param[0]
    expected = request.param[1]
    return input, expected

def test_layer_num_list_multimodal(index_multimodal_get_layer_index_list_test_case):
    input = index_multimodal_get_layer_index_list_test_case[0]
    expected = index_multimodal_get_layer_index_list_test_case[1]
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([6, 7, 8])
    m.append([10])
    m1 = MultiHeadLinkedListLayer()
    m1.append([10, 20, 30, 40, 50])
    m1.append([60, 70, 80])
    m = m + m1
    m.append([100, 200])
    g = Generator(m)
    assert len(g._node_type_layers) == 7
    print(g._get_layer_index_list(input, g._node_type_layers))
    assert g._get_layer_index_list(input, g._node_type_layers) == expected

@pytest.mark.xfail
def test_layer_index_access_error():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    g = Generator(m)
    g[10]


def test_index_access():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([10, 20, 30, 40, 50])
    m.append([100, 200])
    m.append([1000, 2000, 3000, 4000])
    m.append([10000, 20000, 30000, 40000])
    g = Generator(m)
    assert g[3] == [[4], [10], [100], [1000], [10000]]


def test_append_lazy_index_access():
    m = MultiHeadLinkedListLayer()
    m.append_lazy(Hoge1, [{'units': i} for i in range(1, 6)])
    m.append_lazy(Hoge1, [{'units': i} for i in range(10, 60, 10)])
    m.append_lazy(Hoge1, [{'units': i} for i in range(100, 300, 1000)])
    m.append_lazy(Hoge1, [{'units': i} for i in range(1000, 5000, 1000)])
    m.append_lazy(Hoge1, [{'units': i} for i in range(10000, 50000, 100000)])
    g = Generator(m)
    g[3]


def test_multi_modal_index_access():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([10, 20, 30, 40, 50])
    m1 = MultiHeadLinkedListLayer()
    m1.append([100, 200])
    m1.append([1000, 2000, 3000, 4000])
    m1.append([10000, 20000, 30000, 40000])
    m = m1 + m
    m.append([100])
    g = Generator(m)
    assert len(g[3]) == 5
    assert g[3] == [[200, None], [2000, 1], [10000, 10], ['concat'], [100]]


def test_multi_modal2_index_access():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([10, 20, 30, 40, 50])
    m1 = MultiHeadLinkedListLayer()
    m1.append([100, 200])
    m1.append([1000, 2000, 3000, 4000])
    m1.append([10000, 20000, 30000, 40000])
    m1.append([100000, 200000, 300000, 400000])
    m1.append([1000000, 2000000, 3000000, 4000000])
    m = m1 + m
    m.append([100])
    g = Generator(m)
    print(g[3])
    assert g[3] == [[200, None], [2000, None], [10000, None], [100000, 1], [1000000, 10], ['concat'], [100]]


def test_multi_modal2_index_access_2():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([10, 20, 30, 40, 50])
    m1 = MultiHeadLinkedListLayer()
    m1.append([100, 200])
    m1.append([1000, 2000, 3000, 4000])
    m1.append([10000, 20000, 30000, 40000])
    m1.append([100000, 200000, 300000, 400000])
    m1.append([1000000, 2000000, 3000000, 4000000])
    m = m + m1
    m.append([100])
    g = Generator(m)
    assert g[3] == [[None, 100], [None, 1000], [None, 10000], [4, 100000], [10, 1000000], ['concat'], [100]]

def test_multi_modal2_index_access_3():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([10, 20, 30, 40, 50])
    m1 = MultiHeadLinkedListLayer()
    m1.append([100, 200])
    m1.append([1000, 2000, 3000, 4000])
    m1.append([10000, 20000, 30000, 40000])
    m1.append([100000, 200000, 300000, 400000])
    m1.append([1000000, 2000000, 3000000, 4000000])
    m2 = MultiHeadLinkedListLayer()
    m2.append([0.1, 0.2])
    m2.append([0.01, 0.02, 0.03, 0.04])
    m2.append([0.001, 0.002, 0.003, 0.004, 0.005])
    m = m + m1
    m.append([100])
    m = m + m2
    m.append([11, 13, 15])
    g = Generator(m)
    print(g[3])
    assert g[3] == [[None, 100, None], [None, 1000, None], [None, 10000, None], [4, 100000, None], [10, 1000000, 0.1], ['concat', 0.01], [100, 0.001], ['concat'], [11]]


def test_multimodal_network_graph_dump_1():
    m1 = MultiHeadLinkedListLayer()
    num_nodes = 0
    num_features = 0 # coresponds to number of type of layers

    m1.append_lazy(Hoge1, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1; num_features += 1
    m1.append_lazy(Hoge2, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1; num_features += 1
    m1.append_lazy(Hoge3, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1; num_features += 1
    m1.append_lazy(Hoge4, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1; num_features += 1
    m2 = MultiHeadLinkedListLayer()
    m2.append_lazy(Hoge1, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1
    m = m1 + m2; num_nodes += 1; num_features += 1
    m.append_lazy(Hoge4, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1
    m.append_lazy(Hoge3, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1

    g = Generator(m, dump_nn_graph=True)
    graph, (adj, features) = g[0]
    assert adj.shape == (num_nodes, num_nodes)
    assert features.shape == (num_nodes, num_features)
    print(adj, features)
    expected_features = np.array([[0., 32., 0., 0., 0.],
                                  [0., 0., 32., 0., 0.],
                                  [0., 0., 0., 32., 0.],
                                  [0., 0., 0., 0., 32.],
                                  [0., 32., 0., 0., 0.],
                                  [1., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 32.],
                                  [0., 0., 0., 32., 0.]])

    expected_adj = np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 1., 0., 0., 0., 0., 0.],
                             [0., 1., 0., 1., 0., 0., 0., 0.],
                             [0., 0., 1., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 1., 1., 0., 1., 0.],
                             [0., 0., 0., 0., 0., 1., 0., 1.],
                             [0., 0., 0., 0., 0., 0., 1., 0.]])
    np.testing.assert_array_equal(features, expected_features)
    np.testing.assert_array_equal(adj, expected_adj)

def test_multimodal_network_graph_dump_2():
    m1 = MultiHeadLinkedListLayer()
    num_nodes = 0
    num_features = 0 # coresponds to number of type of layers

    m1.append_lazy(Hoge1, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1; num_features += 1
    m1.append_lazy(Hoge2, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1; num_features += 1
    m1.append_lazy(Hoge3, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1; num_features += 1
    m1.append_lazy(Hoge4, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1; num_features += 1
    m1.append_lazy(Hoge1, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1;
    m2 = MultiHeadLinkedListLayer()
    m2.append_lazy(Hoge1, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1
    m2.append_lazy(Hoge2, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1
    m2.append_lazy(Hoge3, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1
    m = m1 + m2; num_nodes += 1; num_features += 1
    m.append_lazy(Hoge4, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1
    m.append_lazy(Hoge3, [dict(units=i) for i in [32, 64, 128]]); num_nodes += 1

    g = Generator(m, dump_nn_graph=True)
    graph, (adj, features) = g[0]
    assert adj.shape == (num_nodes, num_nodes)
    assert features.shape == (num_nodes, num_features)
    print(adj)
    print(features)
    expected_features = np.array([[0., 32., 0., 0., 0.],
                                  [0., 0., 32., 0., 0.],
                                  [0., 0., 0., 32., 0.],
                                  [0., 0., 0., 0., 32.],
                                  [0., 32., 0., 0., 0.],
                                  [0., 32., 0., 0., 0.],
                                  [0., 0., 32., 0., 0.],
                                  [0., 0., 0., 32., 0.],
                                  [1., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 32.],
                                  [0., 0., 0., 32., 0.]])

    expected_adj = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])
    np.testing.assert_array_equal(features, expected_features)
    np.testing.assert_array_equal(adj, expected_adj)


@pytest.mark.skipif(not exist_tensorflow, reason="no tensorflow")
class TestTensorflow(unittest.TestCase):
    def create_model(self, graph):
        model = keras.Sequential()
        for layer in graph:
            model.add(layer[0])
        return model

    def test_tensorflow_layers_append(self):
        m = MultiHeadLinkedListLayer()
        # graph created
        m.append_lazy(keras.layers.Flatten, [dict(input_shape=(28, 28)),])
        m.append_lazy(keras.layers.Dense, [dict(units=32), dict(units=64), dict(units=128)])
        m.append_lazy(keras.layers.ReLU, [dict(),])
        m.append_lazy(keras.layers.Dense, [dict(units=16), dict(units=32), dict(units=64)])
        m.append_lazy(keras.layers.ReLU, [dict(),])
        m.append_lazy(keras.layers.Dense, [dict(units=10),])
        g = Generator(m)
        print(g[1])

    def test_tensorflow_layers_append_and_sequential_model(self):
        m = MultiHeadLinkedListLayer()
        # graph created
        # m.append_lazy(keras.layers.Flatten, [dict(input_shape=(28, 28)),])
        m.append_lazy(keras.layers.Dense, [dict(units=32), dict(units=64), dict(units=128)])
        m.append_lazy(keras.layers.ReLU, [dict(),])
        m.append_lazy(keras.layers.Dense, [dict(units=16), dict(units=32), dict(units=64)])
        m.append_lazy(keras.layers.ReLU, [dict(),])
        m.append_lazy(keras.layers.Dense, [dict(units=10),])
        g = Generator(m)
        self.create_model(g[1])

    def test_tensorflow_layers_append_and_dump_nn_graph(self):
        m = MultiHeadLinkedListLayer()
        m.append_lazy(keras.layers.Flatten, [dict(input_shape=(28, 28)),])
        m.append_lazy(keras.layers.Dense, [dict(units=32), dict(units=64), dict(units=128)])
        m.append_lazy(keras.layers.ReLU, [dict(),])
        m.append_lazy(keras.layers.Dense, [dict(units=16), dict(units=32), dict(units=64)])
        m.append_lazy(keras.layers.ReLU, [dict(),])
        m.append_lazy(keras.layers.Dense, [dict(units=10),])
        g = Generator(m, dump_nn_graph=True)
        graph, (adj, features) = g[0]
        print(graph)
        self.create_model(graph)
        assert (6, 3) == features.shape



@pytest.mark.skipif(not exist_torch, reason="no torch")
class TestTorch(unittest.TestCase):
    def test_torch_layers_append(self):
        pass
