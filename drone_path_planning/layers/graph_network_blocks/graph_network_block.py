import abc
from typing import Any
from typing import Dict

import tensorflow as tf

from drone_path_planning.graphs import Graph
from drone_path_planning.layers.edge_processors import EdgeProcessor
from drone_path_planning.layers.edge_to_node_aggregators import EdgeToNodeAggregator
from drone_path_planning.layers.node_processors import NodeProcessor


@tf.keras.utils.register_keras_serializable('dpp.layers.gnbs', 'gnbs')
class GraphNetworkBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        latent_size: int,
        num_hidden_layers: int,
        *args,
        activation=None,
        use_bias: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        should_layer_normalize: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._latent_size = latent_size
        self._num_hidden_layers = num_hidden_layers
        self._activation = tf.keras.activations.get(activation)
        self._use_bias = use_bias
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self._should_layer_normalize = should_layer_normalize

    def build(self, input_shape):
        # Indexing instead of calling attributes of NamedTuple objects for building when loading models.
        self._edge_processors: Dict[str, EdgeProcessor] = {set_name: self._create_edge_processor() for set_name, _ in input_shape[1].items()}
        self._edge_to_node_aggregators: Dict[str, EdgeToNodeAggregator] = {set_name: self._create_edge_to_node_aggregator() for set_name, _ in input_shape[1].items()}
        self._node_processors: Dict[str, NodeProcessor] = {set_name: self._create_node_processor() for set_name, _ in input_shape[0].items()}

    def call(self, graph: Graph):
        new_edge_graph = self._process_edge_sets(graph)
        new_edge_node_graph = self._process_node_sets(new_edge_graph)
        new_edge_node_residual_graph = new_edge_node_graph + graph
        return new_edge_node_residual_graph

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            latent_size=self._latent_size,
            num_hidden_layers=self._num_hidden_layers,
            activation=tf.keras.activations.serialize(self._activation),
            use_bias=self._use_bias,
            kernel_regularizer=tf.keras.regularizers.serialize(self._kernel_regularizer),
            bias_regularizer=tf.keras.regularizers.serialize(self._bias_regularizer),
            activity_regularizer=tf.keras.regularizers.serialize(self._activity_regularizer),
            should_layer_normalize=self._should_layer_normalize,
        )
        return config

    @abc.abstractmethod
    def _create_edge_processor(self) -> EdgeProcessor:
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_edge_to_node_aggregator(self) -> EdgeToNodeAggregator:
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_node_processor(self) -> NodeProcessor:
        raise NotImplementedError()

    @abc.abstractmethod
    def _process_edge_sets(self, graph: Graph):
        raise NotImplementedError()

    @abc.abstractmethod
    def _process_node_sets(self, graph: Graph):
        raise NotImplementedError()
