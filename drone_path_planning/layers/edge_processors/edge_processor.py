import abc
from typing import Any
from typing import Dict

import tensorflow as tf


@tf.keras.utils.register_keras_serializable('dpp.layers.eps', 'ep')
class EdgeProcessor(tf.keras.layers.Layer):
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
        self._edge_layer = self._create_edge_layer(
            latent_size,
            num_hidden_layers,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            should_layer_normalize=should_layer_normalize,
        )

    def call(self, inputs, *args, **kwargs):
        sender_features, receiver_features, edge_set = inputs
        edge_features = edge_set.features
        sender_nodes = edge_set.senders
        receiver_nodes = edge_set.receivers
        gathered_sender_features = tf.gather(sender_features, sender_nodes, axis=-2, batch_dims=-1)
        gathered_receiver_features = tf.gather(receiver_features, receiver_nodes, axis=-2, batch_dims=-1)
        features_list = [gathered_sender_features, gathered_receiver_features, edge_features]
        features = tf.concat(features_list, axis=-1)
        new_edge_features = self._edge_layer(features)
        return new_edge_features

    @abc.abstractmethod
    def _create_edge_layer(
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
    ) -> tf.keras.layers.Layer:
        raise NotImplementedError()

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
