import tensorflow as tf

from drone_path_planning.layers.basic_layers import MultiLayerPerceptron
from drone_path_planning.layers.node_processors.node_processor import NodeProcessor


@tf.keras.utils.register_keras_serializable('dpp.layers.nps', 'mlp_np')
class MultiLayerPerceptronNodeProcessor(NodeProcessor):
    def _create_node_layer(
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
        return MultiLayerPerceptron(
            latent_size,
            latent_size,
            num_hidden_layers,
            *args,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            should_layer_normalize=should_layer_normalize,
            **kwargs,
        )
