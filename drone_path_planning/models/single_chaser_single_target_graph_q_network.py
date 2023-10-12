from typing import Any
from typing import Dict

import tensorflow as tf

from drone_path_planning.graphs import EdgeSet
from drone_path_planning.graphs import Graph
from drone_path_planning.graphs import NodeSet
from drone_path_planning.graphs import OutputGraphSpec
from drone_path_planning.layers.encode_process_decodes import MultiLayerPerceptronEncodeProcessDecode
from drone_path_planning.utilities.constants import SELF
from drone_path_planning.utilities.constants import TARGET
from drone_path_planning.utilities.constants import SELF_ANGULAR_VELOCITY
from drone_path_planning.utilities.constants import SELF_TARGET
from drone_path_planning.utilities.constants import TARGET_ANGULAR_VELOCITY
from drone_path_planning.utilities.constants import TARGET_RELATIVE_DISPLACEMENT
from drone_path_planning.utilities.constants import TARGET_RELATIVE_VELOCITY
from drone_path_planning.utilities.constants import TARGET_SELF


@tf.keras.utils.register_keras_serializable('dpp.models', 'c1_t1_gqn')
class SingleChaserSingleTargetGraphQNetwork(tf.keras.Model):
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
        self._learned_model = MultiLayerPerceptronEncodeProcessDecode(
            output_specs.copy(),
            latent_size,
            num_hidden_layers,
            num_message_passing_steps,
            should_layer_normalize=should_layer_normalize,
        )

    def call(self, inputs: Dict[str, tf.Tensor]):
        graph = self._build_graph(inputs)
        raw_predictions: Graph = self._learned_model(graph)
        predictions = self._postprocess(raw_predictions)
        return predictions

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

    def _build_graph(self, inputs: Dict[str, tf.Tensor]) -> Graph:
        self_node_set: NodeSet = self._build_self_node_set(inputs[SELF_ANGULAR_VELOCITY])
        target_node_set: NodeSet = self._build_target_node_set(inputs[TARGET_RELATIVE_VELOCITY], inputs[TARGET_ANGULAR_VELOCITY])
        target_relative_displacement = inputs[TARGET_RELATIVE_DISPLACEMENT]
        num_targets = tf.shape(target_relative_displacement)[0]
        targets = tf.range(num_targets)
        selves = tf.zeros_like(targets)
        self_target_edge_set: EdgeSet = self._build_self_target_edge_set(target_relative_displacement, selves, targets)
        target_self_edge_set: EdgeSet = self._build_target_self_edge_set(target_relative_displacement, targets, selves)
        node_sets = {
            SELF: self_node_set,
            TARGET: target_node_set,
        }
        edge_sets = {
            SELF_TARGET: self_target_edge_set,
            TARGET_SELF: target_self_edge_set,
        }
        graph = Graph(node_sets, edge_sets)
        return graph

    def _postprocess(self, raw_predictions: Graph) -> Dict[str, tf.Tensor]:
        predictions = raw_predictions.node_sets[SELF].features
        return predictions

    def _build_self_node_set(self, self_angular_velocity: tf.Tensor) -> NodeSet:
        self_node_features = tf.concat([
            self_angular_velocity,
            tf.norm(self_angular_velocity, axis=-1, keepdims=True),
        ], axis=-1)
        self_node_set = NodeSet(self_node_features)
        return self_node_set

    def _build_target_node_set(self, target_relative_velocity: tf.Tensor, target_angular_velocity: tf.Tensor) -> NodeSet:
        target_node_features = tf.concat([
            target_relative_velocity,
            tf.norm(target_relative_velocity, axis=-1, keepdims=True),
            target_angular_velocity,
            tf.norm(target_angular_velocity, axis=-1, keepdims=True),
        ], axis=-1)
        target_node_set = NodeSet(target_node_features)
        return target_node_set

    def _build_self_target_edge_set(self, target_relative_displacement: tf.Tensor, selves: tf.Tensor, targets: tf.Tensor) -> EdgeSet:
        self_target_edge_features = tf.concat([
            target_relative_displacement,
            tf.norm(target_relative_displacement, axis=-1, keepdims=True),
        ], axis=-1)
        self_target_edge_set = EdgeSet(
            self_target_edge_features,
            selves,
            targets,
            SELF,
            TARGET,
        )
        return self_target_edge_set

    def _build_target_self_edge_set(self, target_relative_displacement: tf.Tensor, targets: tf.Tensor, selves: tf.Tensor) -> EdgeSet:
        target_self_edge_features = tf.concat([
            -target_relative_displacement,
            tf.norm(-target_relative_displacement, axis=-1, keepdims=True),
        ], axis=-1)
        target_self_edge_set = EdgeSet(
            target_self_edge_features,
            targets,
            selves,
            TARGET,
            SELF,
        )
        return target_self_edge_set
