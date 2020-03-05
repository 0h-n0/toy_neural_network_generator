import typing

class Layer:
    def __init__(self,
                 layers: typing.List[typing.Callable] = None,
                 parent: typing.List['Layer'] = None,
                 child=None):
        self.layers = layers
        self._parent = parent
        self.child = child

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @parent.getter
    def parent(self):
        return self._parent

    def __str__(self):
        return f"Layer({self.layers}) parent:{self._parent}, child:{self.child}"

    def __repr__(self):
        return f"Layer({self.layers})"


class MultiHeadLinkedListLayer:
    def __init__(self, head=None, length: int = 1):
        if head is None:
            self.head = Layer()
        else:
            self.head = head
        self.tail = self.head
        self._immutable = False
        self.length = length

    def _set_immutable(self):
        self._immutable = True

    def append(self, layers: typing.List[typing.Callable]) -> 'MultiHeadLinkedListLayer':
        if self._immutable:
            print("can't append layer")
            return self
        self.length += 1
        new = Layer(layers)
        self.tail.child = new
        new.parent = [self.tail,]
        self.tail = new
        return self

    def __add__(self, other: 'MultiHeadLinkedListLayer') -> 'MultiHeadLinkedListLayer':
        concat_layer = Layer()
        self.tail.child = concat_layer
        other.tail.child = concat_layer
        concat_layer.parent = [self.tail, other.tail]
        self._set_immutable()
        other._set_immutable()
        if self.length > other.length:
            _length = self.length
        else:
            _length = other.length
        _length += 1 # for concat layer
        return MultiHeadLinkedListLayer(concat_layer, _length)

    def __str__(self):
        out = ""
        cur = [self.tail,]
        for _ in range(self.length):
            parents = []
            out += f"{cur}\n"
            for c in cur:
                if c is None or c.parent is None:
                    parents.append(None)
                    continue
                for p in c.parent:
                    parents.append(p)
            cur = parents
        return out

    def __rper__(self):
        return f"MultiHeadLinkedListLayer({self.length})"

    def __len__(self):
        return self.length

if __name__ == "__main__":
    def f1():
        pass
    def f2():
        pass
    def f3():
        pass
    def f4():
        pass
    m1 = MultiHeadLinkedListLayer()
    m2 = MultiHeadLinkedListLayer()
    m3 = MultiHeadLinkedListLayer()
    m3.append([f1])
    m1.append([f1])
    m1.append([f2])
    m1.append([f3])
    m1.append([f4])
    m = m1 + m3
    m = m + m
    print(m)
    # m.append([f])
