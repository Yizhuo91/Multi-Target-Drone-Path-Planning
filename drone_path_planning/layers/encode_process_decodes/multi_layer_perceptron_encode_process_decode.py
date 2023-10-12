from typing import Any
from typing import Dict

import tensorflow as tf

from drone_path_planning.graphs import OutputGraphSpec
from drone_path_planning.layers.encode_process_decodes.encode_process_decode import EncodeProcessDecode
from drone_path_planning.layers.graph_decoders import MultiLayerPerceptronGraphDecoder
from drone_path_planning.layers.graph_encoders import GraphEncoder
from drone_path_planning.layers.graph_processors import MultiLayerPerceptronGraphProcessor


@tf.keras.utils.register_keras_serializable('dpp.layers.epds', 'mlp_epd')
class MultiLayerPerceptronEncodeProcessDecode(EncodeProcessDecode):
    def __init__(
        self,
        output_specs: OutputGraphSpec,
        latent_size: int,
        num_hidden_layers: int,
        num_message_passing_steps: int,
        *args,
        should_layer_normalize: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._output_specs = output_specs
        self._latent_size = latent_size
        self._num_hidden_layers = num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._should_layer_normalize = should_layer_normalize

    @property
    def encoder(self):
        if not hasattr(self, '_encoder'):
            self._encoder = GraphEncoder(
                self._latent_size,
                self._num_hidden_layers,
                should_layer_normalize=True,
            )
        return self._encoder

    @property
    def processor(self):
        if not hasattr(self, '_processor'):
            self._processor = MultiLayerPerceptronGraphProcessor(
                self._latent_size,
                self._num_hidden_layers,
                self._num_message_passing_steps,
                should_layer_normalize=True,
            )
        return self._processor

    @property
    def decoder(self):
        if not hasattr(self, '_decoder'):
            self._decoder = MultiLayerPerceptronGraphDecoder(
                self._output_specs.copy(),
                self._latent_size,
                self._num_hidden_layers,
                should_layer_normalize=self._should_layer_normalize,
            )
        return self._decoder

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            output_node_set_specs=self._output_specs.node_sets,
            output_edge_set_specs=self._output_specs.edge_sets,
            latent_size=self._latent_size,
            num_hidden_layers=self._num_hidden_layers,
            num_message_passing_steps=self._num_message_passing_steps,
            should_layer_normalize=self._should_layer_normalize,
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        output_node_set_specs = config.pop('output_node_set_specs')
        output_edge_set_specs = config.pop('output_edge_set_specs')
        output_specs = OutputGraphSpec(output_node_set_specs, output_edge_set_specs)
        config.update(
            output_specs=output_specs,
        )
        return super().from_config(config)
