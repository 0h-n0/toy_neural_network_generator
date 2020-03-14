import typing
import warnings

## for graph dumpping
import networkx as nx
import numpy as np

from .layer import Layer, LazyLayer, MultiHeadLinkedListLayer, DummyConcat


class Generator:
    def __init__(self,
                 multi_head_linked_list_layer: MultiHeadLinkedListLayer,
                 dump_nn_graph: bool = False,
    ):
        self.multi_head_linked_list_layer = multi_head_linked_list_layer
        self.dump_nn_graph = dump_nn_graph
        self._len = self._calc_num_of_all_combination(multi_head_linked_list_layer)
        if self.dump_nn_graph:
            self._backbone_graph, self._node_features = \
                self._create_backbone_graph(multi_head_linked_list_layer)

    def _calc_num_of_all_combination(self, multi_head_linked_list_layer) -> int:
        ''' calculate each the number of layer's combination.
        '''
        num = 1
        for node in self.multi_head_linked_list_layer.graph.nodes(data=True):
            if not 'args' in node[1].keys():
                continue
            if not node[1]['args']:
                continue
            _n_type = len(node[1]['args'])
            num *= _n_type
        return num

    def __getitem__(self, idx):
        if idx > self._len:
            raise IndexError(f"access by {idx}, max length is {self._len}")
        layer_index_list = self._get_layer_index_list(idx)
        out = []
        cur = [self.multi_head_linked_list_layer.tail,]
        for layer_indcies in layer_index_list:
            # from tail to head
            layer = []
            parents = []
            for current_layer, l_idx in zip(cur, layer_indcies):
                if current_layer is None:
                    layer.append(None)
                elif current_layer.layers is not None:
                    if isinstance(current_layer, Layer):
                        layer.append(current_layer.layers[l_idx])
                    elif isinstance(current_layer, LazyLayer):
                        kwargs = current_layer.kwargs_list[l_idx]
                        klass = current_layer.klass
                        if klass == DummyConcat:
                            layer.append(current_layer.klass)
                        else:
                            layer.append(current_layer.klass(**kwargs))
                    else:
                        raise NotImplementedError
                else:
                    layer.append(None)
                if current_layer is None:
                    parents.append(None)
                    continue
                elif current_layer.parent is None:
                    parents.append(None)
                    continue
                for parent in current_layer.parent:
                    parents.append(parent)
            cur = parents
            out.append(layer)
        if self.dump_nn_graph:
            nodelist = range(len(self._backbone_graph.nodes()))
            adj = nx.to_numpy_matrix(self._backbone_graph, nodelist=nodelist)
            node_features = self._create_node_features(idx, layer_index_list)
            return out[::-1], (adj, node_features)
        else:
            return out[::-1]

    def _create_node_features(self, idx, layer_index_list):
        node_features = [[]]
        return np.vstack(node_features)

    def _get_layer_index_list(self, idx: int):
        _idx = idx
        layer_index_list = []
        graph = self.multi_head_linked_list_layer.graph
        for eles in self.layer_num_list:
            _index_list = []
            for _, ele in enumerate(eles):
                _index_list.append(_idx % ele)
                _idx //= ele
            layer_index_list.append(_index_list)
        return layer_index_list

    def __len__(self):
        return self._len

    def draw_graph(self, filename: str):
        import matplotlib.pyplot as plt
        G, _ = self._create_backbone_graph(self.multi_head_linked_list_layer)
        pos = nx.spring_layout(G, iterations=200)
        nx.draw(G, pos, with_labels=True)
        plt.savefig(filename)
        return G
