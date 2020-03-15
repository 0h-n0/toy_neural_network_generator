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
        self._len, self._node_type_layers = self._calc_num_of_all_combination(multi_head_linked_list_layer)

    def _calc_num_of_all_combination(self, multi_head_linked_list_layer) -> (int, typing.List[int]):
        ''' calculate each the number of layer's combination.
        '''
        num = 1
        node_type_layers = []
        for node in self.multi_head_linked_list_layer.graph.nodes(data=True):
            if not 'args' in node[1].keys():
                continue
            if not node[1]['args']:
                continue
            if 'layer' in node[1].keys():
                # append_lazy
                _n_type = len(node[1]['args'])
            else:
                _n_type = len(node[1]['args'][0])
            node_type_layers.append(_n_type)
            num *= _n_type
        return num, node_type_layers

    def __getitem__(self, idx):
        if idx > self._len:
            raise IndexError(f"access by {idx}, max length is {self._len}")
        graph = self.multi_head_linked_list_layer.graph
        layer_index_list = self._get_layer_index_list(idx, self._node_type_layers)
        last_node = len(graph.nodes()) - 1
        no_ordered_outputs = []
        outputs = []
        for layer_idx, node in zip(layer_index_list, graph.nodes(data=True)):
            if 'layer' in node[1].keys():
                # append_lazy
                layer = node[1]['layer']
                kwargs = node[1]['args'][layer_idx]
                no_ordered_outputs.append(layer(**kwargs))
            else:
                value = node[1]['args'][0][layer_idx]
                no_ordered_outputs.append(value)
        outputs.append([no_ordered_outputs[last_node],])

        parents = list(graph.predecessors(last_node))
        while True:
            if parents.count(None) == len(parents):
                break
            _output = []
            _parents = []
            for parent in parents:
                if parent is None:
                    _output.append(None)
                    __parents = [None,]
                else:
                    _output.append(no_ordered_outputs[parent])
                    __parents = list(graph.predecessors(parent))
                if not __parents:
                    _parents += [None,]
                else:
                    _parents += __parents
            outputs.append(_output)
            parents = _parents
        if self.dump_nn_graph:
            nodelist = range(len(self._backbone_graph.nodes()))
            adj = nx.to_numpy_matrix(self._backbone_graph, nodelist=nodelist)
            node_features = self._create_node_features(idx, layer_index_list)
            return outputs[::-1], (adj, node_features)
        else:
            return outputs[::-1]

    def _create_node_features(self, idx, layer_index_list):
        node_features = [[]]
        return np.vstack(node_features)

    def _get_layer_index_list(self, idx: int, node_type_layers):
        _idx = idx
        layer_index_list = []
        for n_args in node_type_layers:
            layer_index_list.append(_idx % n_args)
            _idx //= n_args
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
