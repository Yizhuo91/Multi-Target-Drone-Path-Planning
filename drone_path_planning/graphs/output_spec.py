from typing import Dict
from typing import List
from typing import NamedTuple

from drone_path_planning.graphs.edge_set import EdgeSet
from drone_path_planning.graphs.node_set import NodeSet


OutputHeadSpec = Dict[str, int]
OutputComponentSetSpec = List[OutputHeadSpec]


class OutputGraphSpec(NamedTuple):
    node_sets: Dict[str, OutputComponentSetSpec]
    edge_sets: Dict[str, OutputComponentSetSpec]

    def copy(self):
        return OutputGraphSpec(
            node_sets={key: [{**output_head_spec} for output_head_spec in output_head_specs] for key, output_head_specs in self.node_sets.items()},
            edge_sets={key: [{**output_head_spec} for output_head_spec in output_head_specs] for key, output_head_specs in self.edge_sets.items()},
        )
