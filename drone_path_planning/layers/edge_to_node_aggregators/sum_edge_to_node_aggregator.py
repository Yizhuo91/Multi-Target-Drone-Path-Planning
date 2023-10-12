import tensorflow as tf

from drone_path_planning.layers.edge_to_node_aggregators.edge_to_node_aggregator import EdgeToNodeAggregator


@tf.keras.utils.register_keras_serializable('dpp.layers.enas', 'sum_ena')
class SumEdgeToNodeAggregator(EdgeToNodeAggregator):
    def call(self, inputs, *args, **kwargs):
        edge_set, node_count = inputs
        edge_features = edge_set.features
        receiver_nodes = edge_set.receivers
        aggregated_partial_messages = tf.math.unsorted_segment_sum(
            edge_features,
            receiver_nodes,
            node_count,
        )
        return aggregated_partial_messages
