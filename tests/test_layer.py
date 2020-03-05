from tnng import MultiHeadLinkedListLayer


def test_layer_depth():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([6, 7, 8])
    m.append([10])
    assert len(m) == 3

def test_multi_modal_layer_depth():
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
