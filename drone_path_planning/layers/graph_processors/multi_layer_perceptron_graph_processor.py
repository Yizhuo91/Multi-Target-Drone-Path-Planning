from typing import Any
from typing import Dict

import tensorflow as tf

from drone_path_planning.layers.graph_network_blocks import MultiLayerPerceptronGraphNetworkBlock
from drone_path_planning.layers.graph_processors.graph_processor import GraphProcessor


@tf.keras.utils.register_keras_serializable('dpp.layers.gps', 'mlp_gp')
class MultiLayerPerceptronGraphProcessor(GraphProcessor):
    def __init__(
        self,
        latent_size: int,
        num_hidden_layers: int,
        num_message_passing_steps: int,
        *args,
        activation=None,
        use_bias: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        should_layer_normalize: bool = False,
        **kwargs,
    ):
        super().__init__(num_message_passing_steps, *args, **kwargs)
        self._latent_size = latent_size
        self._num_hidden_layers = num_hidden_layers
        self._activation = tf.keras.activations.get(activation)
        self._use_bias = use_bias
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self._should_layer_normalize = should_layer_normalize

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

    def _create_graph_network_block(self):
        return MultiLayerPerceptronGraphNetworkBlock(
            self._latent_size,
            self._num_hidden_layers,
            activation=self._activation,
            use_bias=self._use_bias,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            should_layer_normalize=self._should_layer_normalize,
        )
