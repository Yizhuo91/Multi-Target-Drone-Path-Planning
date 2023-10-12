import tensorflow as tf

from drone_path_planning.graphs import Graph
from drone_path_planning.layers.edge_processors import EdgeProcessor
from drone_path_planning.layers.edge_processors import MultiLayerPerceptronEdgeProcessor
from drone_path_planning.layers.edge_to_node_aggregators import EdgeToNodeAggregator
from drone_path_planning.layers.edge_to_node_aggregators import SumEdgeToNodeAggregator
from drone_path_planning.layers.graph_network_blocks.graph_network_block import GraphNetworkBlock
from drone_path_planning.layers.node_processors import MultiLayerPerceptronNodeProcessor
from drone_path_planning.layers.node_processors import NodeProcessor


@tf.keras.utils.register_keras_serializable('dpp.layers.gnbs', 'mlp_gnb')
class MultiLayerPerceptronGraphNetworkBlock(GraphNetworkBlock):
    def _create_edge_processor(self) -> EdgeProcessor:
        return MultiLayerPerceptronEdgeProcessor(
            self._latent_size,
            self._num_hidden_layers,
            activation=self._activation,
            use_bias=self._use_bias,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            should_layer_normalize=self._should_layer_normalize,
        )

    def _create_edge_to_node_aggregator(self) -> EdgeToNodeAggregator:
        return SumEdgeToNodeAggregator()

    def _create_node_processor(self) -> NodeProcessor:
        return MultiLayerPerceptronNodeProcessor(
            self._latent_size,
            self._num_hidden_layers,
            activation=self._activation,
            use_bias=self._use_bias,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            should_layer_normalize=self._should_layer_normalize,
        )

    def _process_edge_sets(self, graph: Graph) -> Graph:
        new_edge_sets = dict()
        for edge_set_name in graph.edge_sets:
            edge_processor = self._edge_processors[edge_set_name]
            edge_set = graph.edge_sets[edge_set_name]
            sender_features = graph.node_sets[edge_set.sender_set].features
            receiver_features = graph.node_sets[edge_set.receiver_set].features
            inputs = sender_features, receiver_features, edge_set
            updated_edge_features = edge_processor(inputs)
            new_edge_set = edge_set._replace(features=updated_edge_features)
            new_edge_sets[edge_set_name] = new_edge_set
        new_edge_graph = graph._replace(edge_sets=new_edge_sets)
        return new_edge_graph

    def _process_node_sets(self, graph: Graph) -> Graph:
        new_node_sets = dict()
        for node_set_name in graph.node_sets:
            node_processor = self._node_processors[node_set_name]
            node_set = graph.node_sets[node_set_name]
            node_features = node_set.features
            num_nodes = tf.shape(node_features)[0]
            messages = []
            for edge_set_name, edge_set in graph.edge_sets.items():
                if edge_set.receiver_set != node_set_name:
                    continue
                edge_to_node_aggregator = self._edge_to_node_aggregators[edge_set_name]
                aggregator_inputs = edge_set, num_nodes
                message = edge_to_node_aggregator(aggregator_inputs)
                messages.append(message)
            inputs = node_features, messages
            updated_node_features = node_processor(inputs)
            new_node_set = node_set._replace(features=updated_node_features)
            new_node_sets[node_set_name] = new_node_set
        new_node_graph = graph._replace(node_sets=new_node_sets)
        return new_node_graph
