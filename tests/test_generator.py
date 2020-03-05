import pytest

from tnng import MultiHeadLinkedListLayer, Generator

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
        [0, [[0], [0], [0]]],
        [1, [[0], [1], [0]]],
        [2, [[0], [2], [0]]],
        [3, [[0], [0], [1]]],
        [4, [[0], [1], [1]]],
        [5, [[0], [2], [1]]],
        [14, [[0], [2], [4]]],
        [-1, [[0], [2], [4]]],
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
    assert g._get_layer_index_list(input) == expected


@pytest.fixture(
    params=[
        [0, [[0], [0], [0, 0], [0, 0], [0, 0]]],
        [1, [[1], [0], [0, 0], [0, 0], [0, 0]]],
        [2, [[0], [0], [0, 1], [0, 0], [0, 0]]],
        [3, [[1], [0], [0, 1], [0, 0], [0, 0]]],
        [4, [[0], [0], [0, 2], [0, 0], [0, 0]]],
        [5, [[1], [0], [0, 2], [0, 0], [0, 0]]],
        [449, [[1], [0], [0, 2], [2, 4], [4, 0]]],
        [-1, [[1], [0], [0, 2], [2, 4], [4, 0]]],
        [-2, [[0], [0], [0, 2], [2, 4], [4, 0]]],
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
    assert g._get_layer_index_list(input) == expected

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
    assert g[3] == [[1], [10], [100], [1000], [40000]]


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
    assert g[3] == [[100, None], [1000, 1], [40000, 10], [None], [100]]


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
    assert g[3] == [[100, None], [1000, None], [10000, None], [100000, 1], [4000000, 10], [None], [100]]
