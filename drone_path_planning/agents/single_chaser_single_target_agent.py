from typing import Any
from typing import Dict
from typing import List

import tensorflow as tf

from drone_path_planning.agents.deep_q_network_agent import DeepQNetworkAgent
from drone_path_planning.graphs import OutputGraphSpec
from drone_path_planning.models import SingleChaserSingleTargetGraphQNetwork


@tf.keras.utils.register_keras_serializable('dpp.agents', 'c1_t1_agent')
class SingleChaserSingleTargetAgent(DeepQNetworkAgent):
    def __init__(
        self,
        output_specs: OutputGraphSpec,
        latent_size: int,
        num_hidden_layers: int,
        num_message_passing_steps: int,
        *args,
        initial_epsilon: float = 1.0,
        epsilon_decay_rate: float = 0.9999912164,
        gamma: float = 0.999,
        tau: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._output_specs = output_specs
        self._latent_size = latent_size
        self._num_hidden_layers = num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._epsilon = tf.Variable(initial_epsilon, trainable=False)
        self._epsilon_decay_rate = epsilon_decay_rate
        self._gamma = gamma
        self._tau = tau
        self._model = self._create_model()
        self._target_model = self._create_model()
        self.update_target_model(tau=1.0)

    def update_target_model(self, tau: float = None):
        if tau is None:
            tau = self.tau
        if tau == 0.0:
            return
        for target_model_trainable_variable, model_trainable_variable in zip(self._target_model.trainable_variables, self._model.trainable_variables):
            target_model_trainable_variable.assign((1.0 - tau) * target_model_trainable_variable + tau * model_trainable_variable)
        for target_model_non_trainable_variable, model_non_trainable_variable in zip(self._target_model.non_trainable_variables, self._model.non_trainable_variables):
            target_model_non_trainable_variable.assign(model_non_trainable_variable)

    def update_epsilon(self):
        updated_epsilon = self._epsilon * self._epsilon_decay_rate
        self._epsilon.assign(updated_epsilon)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            output_node_set_specs=self._output_specs.node_sets,
            output_edge_set_specs=self._output_specs.edge_sets,
            latent_size=self._latent_size,
            num_hidden_layers=self._num_hidden_layers,
            num_message_passing_steps=self._num_message_passing_steps,
            epsilon_decay_rate=self._epsilon_decay_rate,
            gamma=self._gamma,
            tau=self._tau,
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

    @property
    def target_model(self) -> tf.keras.Model:
        return self._target_model

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @property
    def epsilon(self) -> tf.Tensor:
        return self._epsilon

    @property
    def gamma(self) -> tf.Tensor:
        return self._gamma

    @property
    def tau(self) -> tf.Tensor:
        return self._tau

    def _calculate_loss(self, target_q_value: tf.Tensor, q_value: tf.Tensor) -> tf.Tensor:
        error = (target_q_value - q_value) ** 2
        loss = tf.math.reduce_mean(error)
        return loss

    def _calculate_custom_metrics(self, rewards: List[tf.Tensor], episode_return: tf.Tensor, step_count: tf.Tensor) -> tf.Tensor:
        custom_metrics = dict()
        return_per_step = episode_return / step_count
        custom_metrics['return_per_step'] = return_per_step
        non_terminal_reward = tf.cond(step_count > 1, lambda: (episode_return - rewards[-1]) / (step_count - 1), lambda: 0.0)
        custom_metrics['non_terminal_reward'] = non_terminal_reward
        return custom_metrics

    def _create_model(self) -> tf.keras.Model:
        model = SingleChaserSingleTargetGraphQNetwork(
            self._output_specs.copy(),
            self._latent_size,
            self._num_hidden_layers,
            self._num_message_passing_steps,
        )
        return model
