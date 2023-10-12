import abc
from typing import Dict
from typing import List

import tensorflow as tf

from drone_path_planning.environments import Environment
from drone_path_planning.utilities.time_step import STEP_TYPE_LAST
from drone_path_planning.utilities.time_step import TimeStep
from drone_path_planning.utilities.training_helpers import Transition


@tf.keras.utils.register_keras_serializable('dpp.agents', 'dqn_agent')
class DeepQNetworkAgent(tf.keras.Model):
    def call(self, inputs: Dict[str, tf.Tensor], should_use_target_model: bool = False):
        predictions: Dict[str, tf.Tensor]
        if should_use_target_model:
            predictions = self.target_model(inputs)
        else:
            predictions = self.model(inputs)
        return predictions

    def collect_step(self, time_step: TimeStep) -> tf.Tensor:
        q_values = self(time_step.observation)
        concatenated_q_values = tf.concat(list(q_values.values()), -1)
        num_actions = tf.shape(concatenated_q_values)[-1]
        should_take_random_step = tf.random.uniform([]) < self.epsilon
        action = tf.cond(
            should_take_random_step,
            lambda: tf.random.uniform([], maxval=num_actions, dtype=tf.dtypes.int32),
            lambda: tf.squeeze(tf.math.argmax(concatenated_q_values, axis=-1, output_type=tf.dtypes.int32)),
        )
        return action

    def train_step(self, transition: Transition) -> Dict[str, tf.Tensor]:
        time_step, action, _ = transition
        target_q_value = tf.stop_gradient(self._determine_target_q_value(transition.next_time_step))
        with tf.GradientTape() as tape:
            q_values: Dict[str, tf.Tensor] = self(time_step.observation)
            concatenated_q_values = tf.concat(list(q_values.values()), -1)
            q_value = concatenated_q_values[:, action:action + 1]
            loss = self._calculate_loss(target_q_value, q_value)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics['loss'] = loss
        metrics['epsilon'] = self.epsilon
        return metrics

    def evaluate_step(self, environment: Environment, max_num_steps_per_episode: int) -> Dict[str, tf.Tensor]:
        time_step = environment.reset()
        rewards = []
        rewards.append(time_step.reward)
        episode_return = time_step.reward
        step_count = 0.0
        while step_count < max_num_steps_per_episode and time_step.step_type != STEP_TYPE_LAST:
            action = self._episode_step(time_step)
            time_step = environment.step(action)
            rewards.append(time_step.reward)
            episode_return += time_step.reward
            step_count += 1.0
        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics['return'] = episode_return
        metrics['step_count'] = step_count
        custom_metrics = self._calculate_custom_metrics(rewards, episode_return, step_count)
        metrics = {
            **metrics,
            **custom_metrics,
        }
        return metrics

    def predict_step(self, time_step: TimeStep) -> tf.Tensor:
        q_values: Dict[str, tf.Tensor] = self(time_step.observation)
        concatenated_q_values = tf.concat(list(q_values.values()), -1)
        action = tf.squeeze(tf.math.argmax(concatenated_q_values, axis=-1))
        return action

    @abc.abstractmethod
    def update_target_model(self, tau: float = None):
        raise NotImplementedError()

    @abc.abstractmethod
    def update_epsilon(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def target_model(self) -> tf.keras.Model:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def model(self) -> tf.keras.Model:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def epsilon(self) -> tf.Tensor:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def gamma(self) -> tf.Tensor:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def tau(self) -> tf.Tensor:
        raise NotImplementedError()

    def _determine_target_q_value(self, next_time_step: TimeStep) -> tf.Tensor:
        is_last = next_time_step.step_type == STEP_TYPE_LAST
        target_q_value = tf.cond(
            is_last,
            lambda: self._calculate_last_target_q_value(next_time_step),
            lambda: self._calculate_mid_target_q_value(next_time_step),
        )
        return target_q_value

    def _calculate_last_target_q_value(self, next_time_step: TimeStep) -> tf.Tensor:
        last_target_q_value = next_time_step.reward
        return last_target_q_value

    def _calculate_mid_target_q_value(self, next_time_step: TimeStep) -> tf.Tensor:
        next_next_q_values: Dict[str, tf.Tensor] = self(next_time_step.observation, should_use_target_model=True)
        concatenated_next_next_q_values = tf.concat(list(next_next_q_values.values()), -1)
        mid_target_q_value = next_time_step.reward + self.gamma * tf.reduce_max(concatenated_next_next_q_values, axis=-1, keepdims=True)
        return mid_target_q_value

    @tf.function
    def _episode_step(self, time_step: TimeStep) -> tf.Tensor:
        action = self.predict_step(time_step)
        return action

    @abc.abstractmethod
    def _calculate_loss(self, target_q_value: tf.Tensor, q_value: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def _calculate_custom_metrics(self, rewards: List[tf.Tensor], episode_return: tf.Tensor, step_count: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()
