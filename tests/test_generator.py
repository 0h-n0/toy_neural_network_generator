from tnng import MultiHeadLinkedListLayer, Generator


def test_generator_length():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([6, 7, 8])
    m.append([10])
    g = Generator(m)
    assert len(g) == 15


def test_multi_modal_generator_length_():
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
