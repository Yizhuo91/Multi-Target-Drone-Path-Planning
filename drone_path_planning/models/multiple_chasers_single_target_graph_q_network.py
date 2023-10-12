from typing import Any
from typing import Dict

import tensorflow as tf

from drone_path_planning.graphs import EdgeSet
from drone_path_planning.graphs import Graph
from drone_path_planning.graphs import NodeSet
from drone_path_planning.graphs import OutputGraphSpec
from drone_path_planning.layers.encode_process_decodes import MultiLayerPerceptronEncodeProcessDecode
from drone_path_planning.utilities.constants import AGENT
from drone_path_planning.utilities.constants import AGENT_ANGULAR_VELOCITIES
from drone_path_planning.utilities.constants import AGENT_CHASER
from drone_path_planning.utilities.constants import AGENT_TARGET
from drone_path_planning.utilities.constants import CHASER
from drone_path_planning.utilities.constants import CHASER_AGENT
from drone_path_planning.utilities.constants import CHASER_ANGULAR_VELOCITIES
from drone_path_planning.utilities.constants import CHASER_CHASER
from drone_path_planning.utilities.constants import CHASER_RELATIVE_DISPLACEMENTS
from drone_path_planning.utilities.constants import CHASER_RELATIVE_VELOCITIES
from drone_path_planning.utilities.constants import CHASER_TARGET
from drone_path_planning.utilities.constants import TARGET
from drone_path_planning.utilities.constants import TARGET_AGENT
from drone_path_planning.utilities.constants import TARGET_ANGULAR_VELOCITIES
from drone_path_planning.utilities.constants import TARGET_CHASER
from drone_path_planning.utilities.constants import TARGET_RELATIVE_DISPLACEMENTS
from drone_path_planning.utilities.constants import TARGET_RELATIVE_VELOCITIES
from drone_path_planning.utilities.functions import find_cartesian_product
from drone_path_planning.utilities.functions import find_cartesian_square_pairs_with_distinct_elements
from drone_path_planning.utilities.functions import find_pairs_from_cartesian_product
from drone_path_planning.utilities.functions import find_relative_quantities


@tf.keras.utils.register_keras_serializable('dpp.models', 'cn_t1_gqn')
class MultipleChasersSingleTargetGraphQNetwork(tf.keras.Model):
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
        agent_node_set = self._build_agent_node_set(inputs[AGENT_ANGULAR_VELOCITIES])
        chaser_node_set = self._build_node_set(inputs[CHASER_RELATIVE_VELOCITIES], inputs[CHASER_ANGULAR_VELOCITIES])
        target_node_set = self._build_node_set(inputs[TARGET_RELATIVE_VELOCITIES], inputs[TARGET_ANGULAR_VELOCITIES])
        chaser_relative_displacements = inputs[CHASER_RELATIVE_DISPLACEMENTS]
        target_relative_displacements = inputs[TARGET_RELATIVE_DISPLACEMENTS]
        num_chasers = tf.shape(chaser_relative_displacements)[0]
        num_targets = tf.shape(target_relative_displacements)[0]
        print("NUMTARGETS:",num_targets)
        chasers = tf.range(num_chasers)
        agents_for_chasers = tf.zeros_like(chasers)
        targets = tf.range(num_targets)
        agents_for_targets = tf.zeros_like(targets)
        agent_chaser_edge_set = self._build_edge_set(chaser_relative_displacements, agents_for_chasers, chasers, AGENT, CHASER)
        chaser_agent_edge_set = self._build_edge_set(-chaser_relative_displacements, chasers, agents_for_chasers, CHASER, AGENT)
        agent_target_edge_set = self._build_edge_set(target_relative_displacements, agents_for_targets, targets, AGENT, TARGET)
        target_agent_edge_set = self._build_edge_set(-target_relative_displacements, targets, agents_for_targets, TARGET, AGENT)
        chaser_chaser_edge_set = self._build_all_pairs_with_distinct_elements_relative_displacement_edge_set(chaser_relative_displacements, chasers, CHASER)
        chaser_target_edge_set = self._build_all_pairs_relative_displacement_edge_set(chaser_relative_displacements, target_relative_displacements, chasers, targets, CHASER, TARGET)
        target_chaser_edge_set = self._build_all_pairs_relative_displacement_edge_set(target_relative_displacements, chaser_relative_displacements, targets, chasers, TARGET, CHASER)
        node_sets = {
            AGENT: agent_node_set,
            CHASER: chaser_node_set,
            TARGET: target_node_set,
        }
        
        
        edge_sets = {
            AGENT_CHASER: agent_chaser_edge_set,
            CHASER_AGENT: chaser_agent_edge_set,
            AGENT_TARGET: agent_target_edge_set,
            TARGET_AGENT: target_agent_edge_set,
            CHASER_CHASER: chaser_chaser_edge_set,
            CHASER_TARGET: chaser_target_edge_set,
            TARGET_CHASER: target_chaser_edge_set,
        }
        graph = Graph(node_sets, edge_sets)
   
        return graph

    def _postprocess(self, raw_predictions: Graph) -> Dict[str, tf.Tensor]:
        predictions = raw_predictions.node_sets[AGENT].features
        return predictions

    def _build_agent_node_set(self, agent_angular_velocities: tf.Tensor) -> NodeSet:
        agent_node_features = tf.concat([
            agent_angular_velocities,
            tf.norm(agent_angular_velocities, axis=-1, keepdims=True),
        ], axis=-1)
        agent_node_set = NodeSet(agent_node_features)
        return agent_node_set

    def _build_node_set(self, relative_velocities: tf.Tensor, angular_velocities: tf.Tensor) -> NodeSet:
        node_features = tf.concat([
            relative_velocities,
            tf.norm(relative_velocities, axis=-1, keepdims=True),
            angular_velocities,
            tf.norm(angular_velocities, axis=-1, keepdims=True),
        ], axis=-1)
        node_set = NodeSet(node_features)
       
        
        return node_set

    def _build_edge_set(self, relative_displacements: tf.Tensor, senders: tf.Tensor, receivers: tf.Tensor, sender_set: str, receiver_set: str) -> EdgeSet:
        edge_features = tf.concat([
            relative_displacements,
            tf.norm(relative_displacements, axis=-1, keepdims=True),
        ], axis=-1)
        edge_set = EdgeSet(
            edge_features,
            senders,
            receivers,
            sender_set,
            receiver_set,
        )
        return edge_set

    def _build_all_pairs_relative_displacement_edge_set(
        self,
        senders_displacements: tf.Tensor,
        receivers_displacements: tf.Tensor,
        senders: tf.Tensor,
        receivers: tf.Tensor,
        sender_set: str,
        receiver_set: str,
    ) -> EdgeSet:
        relative_displacements = find_relative_quantities(senders_displacements, receivers_displacements)
        sender_receiver_cartesian_product = find_cartesian_product(senders, receivers)
        sender_receiver_pairs = find_pairs_from_cartesian_product(sender_receiver_cartesian_product)
        all_pairs_senders = sender_receiver_pairs[:, 0]
        all_pairs_receivers = sender_receiver_pairs[:, 1]
        edge_features = tf.concat([
            relative_displacements,
            tf.norm(relative_displacements, axis=-1, keepdims=True),
        ], axis=-1)
        edge_features = tf.ensure_shape(edge_features, [None, 4])
        edge_set = EdgeSet(
            edge_features,
            all_pairs_senders,
            all_pairs_receivers,
            sender_set,
            receiver_set,
        )
        return edge_set

    def _build_all_pairs_with_distinct_elements_relative_displacement_edge_set(
        self,
        displacements: tf.Tensor,
        entities: tf.Tensor,
        node_set_name: str,
    ):
        relative_displacements = find_relative_quantities(displacements)
        sender_receiver_cartesian_product = find_cartesian_product(entities, entities)
        sender_receiver_pairs = find_pairs_from_cartesian_product(sender_receiver_cartesian_product)
        sender_receiver_pairs_with_distinct_elements = find_cartesian_square_pairs_with_distinct_elements(sender_receiver_pairs)
        all_pairs_senders = sender_receiver_pairs_with_distinct_elements[:, 0]
        all_pairs_receivers = sender_receiver_pairs_with_distinct_elements[:, 1]
        edge_features = tf.concat([
            relative_displacements,
            tf.norm(relative_displacements, axis=-1, keepdims=True),
        ], axis=-1)
        edge_features = tf.ensure_shape(edge_features, [None, 4])
        edge_set = EdgeSet(
            edge_features,
            all_pairs_senders,
            all_pairs_receivers,
            node_set_name,
            node_set_name,
        )
        return edge_set
