import abc

import tensorflow as tf

from drone_path_planning.graphs import Graph
from drone_path_planning.layers.graph_decoders import GraphDecoder
from drone_path_planning.layers.graph_encoders import GraphEncoder
from drone_path_planning.layers.graph_processors import GraphProcessor


@tf.keras.utils.register_keras_serializable('dpp.layers.epds', 'epd')
class EncodeProcessDecode(tf.keras.layers.Layer):
    @property
    @abc.abstractmethod
    def encoder(self) -> GraphEncoder:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def processor(self) -> GraphProcessor:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def decoder(self) -> GraphDecoder:
        raise NotImplementedError()

    def call(self, graph: Graph):
        encoded_graph = self.encoder(graph)
        processed_graph = self.processor(encoded_graph)
        decoded_graph = self.decoder(processed_graph)
        return decoded_graph
