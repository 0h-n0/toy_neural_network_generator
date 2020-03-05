from .layer import MultiHeadLinkedListLayer


class Generator:
    def __init__(self, multi_head_linked_list_layer: MultiHeadLinkedListLayer):
        self.multi_head_linked_list_layer = multi_head_linked_list_layer
        self._len = self._preprocess(self.multi_head_linked_list_layer)

    def _preprocess(self, multi_head_linked_list_layer) -> int:
        cur = [multi_head_linked_list_layer.tail,]
        num = 1
        num *= len(multi_head_linked_list_layer.tail.layers)
        for _ in range(multi_head_linked_list_layer.depth):
            parents = []
            for c in cur:
                if c is None or c.parent is None:
                    parents.append(None)
                    continue
                for p in c.parent:
                    parents.append(p)
                    if p.layers is not None:
                        num *= len(p.layers)
                print(num)
            cur = parents
        return num

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return self._len
