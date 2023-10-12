import tensorflow as tf

from drone_path_planning.layers.graph_decoders.graph_decoder import GraphDecoder
from drone_path_planning.layers.basic_layers import MultiLayerPerceptron


@tf.keras.utils.register_keras_serializable('dpp.layers.gds', 'mlp_gd')
class MultiLayerPerceptronGraphDecoder(GraphDecoder):
    def _create_decoder(self, output_size: int) -> tf.keras.layers.Layer:
        decoder = MultiLayerPerceptron(
            output_size,
            self._latent_size,
            self._num_hidden_layers,
            activation=self._activation,
            use_bias=self._use_bias,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            should_layer_normalize=self._should_layer_normalize,
        )
        return decoder
