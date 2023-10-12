from __future__ import annotations
from typing import Dict
from typing import NamedTuple

from drone_path_planning.graphs.edge_set import EdgeSet
from drone_path_planning.graphs.node_set import NodeSet


class Graph(NamedTuple):
    node_sets: Dict[str, NodeSet]
    edge_sets: Dict[str, EdgeSet]

    def __add__(self, other: Graph) -> Graph:
        if not isinstance(other, Graph):
            raise TypeError('unsupported operand type(s) for +: \'{self_type}\' and \'{other_type}\''.format(
                self_type=type(self).__name__,
                other_type=type(other).__name__,
            ))
        if self.node_sets.keys() != other.node_sets.keys() or self.edge_sets.keys() != other.edge_sets.keys():
            raise ValueError('Graph addition requires component set mappings with all the same keys')
        new_node_sets: Dict[str, NodeSet] = {}
        for key in self.node_sets:
            self_node_set = self.node_sets[key]
            other_node_set = other.node_sets[key]
            new_node_sets[key] = self_node_set + other_node_set
        new_edge_sets: Dict[str, EdgeSet] = {}
        for key in self.edge_sets:
            self_edge_set = self.edge_sets[key]
            other_edge_set = other.edge_sets[key]
            new_edge_sets[key] = self_edge_set + other_edge_set
        new_graph = self._replace(node_sets=new_node_sets, edge_sets=new_edge_sets)
        return new_graph
