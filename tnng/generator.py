from .layer import MultiHeadLinkedListLayer


class Generator:
    def __init__(self, multi_head_linked_list_layer: MultiHeadLinkedListLayer):
        self.multi_head_linked_list_layer = multi_head_linked_list_layer
        self._len = self._preprocess(self.multi_head_linked_list_layer)

    def _preprocess(self, multi_head_linked_list_layer) -> int:
        cur = [multi_head_linked_list_layer.tail,]
        num = 1
        for _ in range(multi_head_linked_list_layer.depth):
            parents = []
            num_parents = []
            for c in cur:
                if c is None or c.parent is None:
                    parents.append(None)
                    num_parents.append(1)
                    continue
                for p in c.parent:
                    parents.append(p)
                    if p.layers is not None:
                        num_parents.append(len(p.layers))
                        num *= len(p.layers)
                    else:
                        num_parents.append(1)
            cur = parents
        return num

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return self._len
