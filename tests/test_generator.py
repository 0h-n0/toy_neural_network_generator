from tnng import MultiHeadLinkedListLayer, Generator

def test_generator_length():
    m = MultiHeadLinkedListLayer()
    m.append([1, 2, 3, 4, 5])
    m.append([6, 7, 8])
    m.append([10])
    g = Generator(m)
    assert len(g) == 15
