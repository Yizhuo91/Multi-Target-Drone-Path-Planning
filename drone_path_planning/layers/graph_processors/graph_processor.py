import abc
from typing import Any
from typing import Dict
from typing import Sequence

import tensorflow as tf

from drone_path_planning.graphs import Graph
from drone_path_planning.layers.graph_network_blocks import GraphNetworkBlock


@tf.keras.utils.register_keras_serializable('dpp.layers.gps', 'gp')
class GraphProcessor(tf.keras.layers.Layer):
    def __init__(
        self,
        num_message_passing_steps: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._num_message_passing_steps = num_message_passing_steps

    def call(self, graph: Graph):
        for graph_network_block in self.graph_network_blocks:
            graph = graph_network_block(graph)
        return graph

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            num_message_passing_steps=self._num_message_passing_steps,
        )
        return config

    @property
    def graph_network_blocks(self) -> Sequence[GraphNetworkBlock]:
        if not hasattr(self, '_graph_network_blocks'):
            self._graph_network_blocks = [self._create_graph_network_block() for _ in range(self._num_message_passing_steps)]
        return self._graph_network_blocks

    @abc.abstractmethod
    def _create_graph_network_block(self) -> GraphNetworkBlock:
        raise NotImplementedError()
