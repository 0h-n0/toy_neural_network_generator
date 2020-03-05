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
