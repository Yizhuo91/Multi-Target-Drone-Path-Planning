from typing import Any
from typing import Dict

import tensorflow as tf


@tf.keras.utils.register_keras_serializable('dpp.layers.bls', 'mlp')
class MultiLayerPerceptron(tf.keras.layers.Layer):
    def __init__(
        self,
        output_size: int,
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
        self._output_size = output_size
        self._latent_size = latent_size
        self._num_hidden_layers = num_hidden_layers
        self._activation = tf.keras.activations.get(activation)
        self._use_bias = use_bias
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self._should_layer_normalize = should_layer_normalize
        self._hidden_layers = [tf.keras.layers.Dense(
            latent_size,
            activation=tf.nn.relu,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        ) for _ in range(num_hidden_layers)]
        self._output_layer = tf.keras.layers.Dense(
            output_size,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
        self._should_layer_normalize = should_layer_normalize
        if should_layer_normalize:
            self._layer_normalization_layer = tf.keras.layers.LayerNormalization()

    def call(self, x, *args, **kwargs):
        for hidden_layer in self._hidden_layers:
            x = hidden_layer(x)
        x = self._output_layer(x)
        if self._should_layer_normalize:
            x = self._layer_normalization_layer(x)
           
        
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            output_size=self._output_size,
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
