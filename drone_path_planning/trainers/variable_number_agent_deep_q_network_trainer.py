import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import tensorflow as tf

from drone_path_planning.environments import VariableNumberAgentEnvironment
from drone_path_planning.trainers.trainer import Trainer
from drone_path_planning.utilities.agent_groups import VariableNumberAgentTrainingGroup
from drone_path_planning.utilities.time_step import STEP_TYPE_LAST
from drone_path_planning.utilities.time_step import TimeStep
from drone_path_planning.utilities.training_helpers import Transition


_NOT_NONE_VAL_PARAMETERS_ASSERTION_ERROR_MESSAGE = 'All validation parameters should be None if validation is not to be done (the validation Environment is None).'


class VariableNumberAgentDeepQNetworkTrainer(Trainer):
    def __init__(
        self,
        groups: Dict[str, VariableNumberAgentTrainingGroup],
        environment: VariableNumberAgentEnvironment,
        num_epochs: int,
        num_steps_per_epoch: int,
        max_num_steps_per_episode: int,
        save_dir: str,
        validation_environment: Optional[VariableNumberAgentEnvironment] = None,
        num_val_episodes: Optional[int] = None,
        max_num_steps_per_val_episode: Optional[int] = None,
        logs_dir: Optional[str] = None,
        tf_collect_step_input_signatures: Dict[str, Any] = dict(),
        tf_train_step_input_signatures: Dict[str, Any] = dict(),
        tf_predict_step_input_signatures: Dict[str, Any] = dict(),
    ):
        self._groups = groups
        self._environment = environment
        callbacks = []
        for group_id, group in self._groups.items():
            group_save_dir = os.path.join(save_dir, group_id)
            group_logs_dir = None if logs_dir is None else os.path.join(logs_dir, group_id)
            group_callbacks = self._create_training_callbacks(
                num_epochs,
                num_steps_per_epoch,
                group_save_dir,
                logs_dir=group_logs_dir,
            )
            for group_callback in group_callbacks:
                group_callback.set_model(group.agent)
            callbacks.extend(group_callbacks)
        self._callback_list_callback = tf.keras.callbacks.CallbackList(
            callbacks=callbacks,
        )
        self._num_epochs = num_epochs
        self._num_steps_per_epoch = num_steps_per_epoch
        self._max_num_steps_per_episode = max_num_steps_per_episode
        if validation_environment is None:
            has_none_num_val_episodes = num_val_episodes is None
            has_none_max_num_steps_per_val_episode = max_num_steps_per_val_episode is None
            has_none_val_parameters = has_none_num_val_episodes and has_none_max_num_steps_per_val_episode
            assert has_none_val_parameters, _NOT_NONE_VAL_PARAMETERS_ASSERTION_ERROR_MESSAGE
            return
        self._validation_environment = validation_environment
        self._num_val_episodes = num_val_episodes
        self._max_num_steps_per_val_episode = max_num_steps_per_val_episode
        self._replay_buffer_progbar_logger_callback = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps',
        )
        self._tf_collect_step_input_signatures = tf_collect_step_input_signatures
        self._tf_train_step_input_signatures = tf_train_step_input_signatures
        self._tf_predict_step_input_signatures = tf_predict_step_input_signatures

    def initialize(self):
        self._fill_replay_buffers()
        for group in self._groups.values():
            group.agent.compile(**group.agent_compile_kwargs)

    def train(self):
        tf_collect_steps = dict()
        tf_train_steps = dict()
        for group_id, group in self._groups.items():
            tf_collect_step_input_signature = self._tf_collect_step_input_signatures[group_id]
            tf_collect_steps[group_id] = tf.function(
                group.agent.collect_step,
                input_signature=tf_collect_step_input_signature,
            )
            tf_train_step_input_signature = self._tf_train_step_input_signatures[group_id]
            tf_train_steps[group_id] = tf.function(
                group.agent.train_step,
                input_signature=tf_train_step_input_signature,
            )
        self._environment.reset()
        time_steps = self._get_time_steps(self._environment)
        k = 0
        num_episodes = 0
        training_logs: Dict[str, tf.Tensor]
        self._callback_list_callback.on_train_begin()
        for i in range(self._num_epochs):
            epoch_logs: Dict[str, tf.Tensor] = dict()
            self._callback_list_callback.on_epoch_begin(i)
            for j in range(self._num_steps_per_epoch):
                epoch_logs['num_episodes'] = num_episodes
                for group_id, group in self._groups.items():
                    training_transition = group.replay_buffer.sample()
                    self._callback_list_callback.on_train_batch_begin(j)
                    batch_logs = tf_train_steps[group_id](training_transition)
                    epoch_logs.update({f'{group_id}_{key}': value for key, value in batch_logs.items()})
                    batch_logs['num_episodes'] = num_episodes
                    self._callback_list_callback.on_train_batch_end(float(j), logs=batch_logs)
                    group.agent.update_epsilon()
                    group.agent.update_target_model()
                actions = self._choose_actions(tf_collect_steps, time_steps)
                self._apply_actions(self._environment, actions)
                self._environment.update()
                next_time_steps = self._get_time_steps(self._environment)
                self._add_transitions_to_replay_buffers(time_steps, actions, next_time_steps)
                k += 1
                if k < self._max_num_steps_per_episode and not self._is_last_step(next_time_steps):
                    time_steps = next_time_steps
                    continue
                k = 0
                num_episodes += 1
                self._environment.reset()
                time_steps = self._get_time_steps(self._environment)
            if self._validation_environment is not None:
                val_logs = self._validate()
                epoch_logs.update({f'val_{key}': value for key, value in val_logs.items()})
            self._callback_list_callback.on_epoch_end(i, logs=epoch_logs)
            training_logs = epoch_logs
        self._callback_list_callback.on_train_end(logs=training_logs)

    def _validate(self) -> Dict[str, List[tf.Tensor]]:
        tf_predict_steps = dict()
        for group_id, group in self._groups.items():
            tf_predict_step_input_signature = self._tf_predict_step_input_signatures[group_id]
            tf_predict_steps[group_id] = tf.function(
                group.agent.predict_step,
                input_signature=tf_predict_step_input_signature,
            )
        episode_returns = {group_id: tf.constant(0.0) for group_id in self._groups}
        test_logs: Dict[str, List[tf.Tensor]] = dict()
        self._callback_list_callback.on_test_begin()
        for i in range(self._num_val_episodes):
            self._validation_environment.reset()
            time_steps = self._get_time_steps(self._validation_environment)
            self._update_episode_returns(time_steps, episode_returns)
            batch_logs = dict()
            step_count = 0
            self._callback_list_callback.on_test_batch_begin(i)
            for j in range(self._max_num_steps_per_episode):
                if self._is_last_step(time_steps):
                    break
                step_count = j + 1
                actions = self._choose_actions(tf_predict_steps, time_steps)
                self._apply_actions(self._validation_environment, actions)
                self._validation_environment.update()
                time_steps = self._get_time_steps(self._validation_environment)
                self._update_episode_returns(time_steps, episode_returns)
            batch_logs['step_count'] = float(step_count)
            for group_id in self._groups:
                batch_logs[f'{group_id}_return'] = episode_returns[group_id]
            self._callback_list_callback.on_test_batch_end(float(i), logs=batch_logs)
            for key, value in batch_logs.items():
                if key not in test_logs:
                    test_logs[key] = []
                test_logs[key].append(value)
        test_logs = {key: tf.reduce_mean(tf.stack(values)) for key, values in test_logs.items()}
        self._callback_list_callback.on_test_end(logs=test_logs)
        return test_logs

    def _fill_replay_buffers(self):
        policies = dict()
        for group_id, group in self._groups.items():
            tf_collect_step_input_signature = self._tf_collect_step_input_signatures[group_id]
            policies[group_id] = tf.function(
                group.agent.collect_step,
                input_signature=tf_collect_step_input_signature,
            )
        self._environment.reset()
        time_steps = self._get_time_steps(self._environment)
        i = 0
        j = 0
        self._replay_buffer_progbar_logger_callback.on_test_begin()
        while not self._has_filled_replay_buffers():
            self._replay_buffer_progbar_logger_callback.on_test_batch_begin(i)
            actions = self._choose_actions(policies, time_steps)
            self._apply_actions(self._environment, actions)
            self._environment.update()
            next_time_steps = self._get_time_steps(self._environment)
            self._replay_buffer_progbar_logger_callback.on_test_batch_end(float(i), logs=dict())
            num_added = self._add_transitions_to_replay_buffers(time_steps, actions, next_time_steps)
            i += num_added
            j += 1
            if j < self._max_num_steps_per_episode and not self._is_last_step(next_time_steps):
                time_steps = next_time_steps
                continue
            j = 0
            self._environment.reset()
            time_steps = self._get_time_steps(self._environment)
        self._replay_buffer_progbar_logger_callback.on_test_end()

    def _choose_actions(
        self,
        policies: Dict[str, Callable[[TimeStep], tf.Tensor]],
        time_steps: Dict[str, Dict[str, TimeStep]],
    ) -> Dict[str, Dict[str, tf.Tensor]]:
        actions: Dict[str, Dict[str, tf.Tensor]] = {group_id: dict() for group_id in self._groups}
        for group_id in self._groups:
            if group_id not in time_steps:
                continue
            policy = policies[group_id]
            agent_time_steps = time_steps[group_id]
            for agent_id, time_step in agent_time_steps.items():
                action = policy(time_step)
                actions[group_id][agent_id] = action
        return actions

    def _apply_actions(self, environment: VariableNumberAgentEnvironment, actions: Dict[str, Dict[str, tf.Tensor]]):
        for group_id in self._groups:
            if group_id not in actions:
                continue
            agent_actions = actions[group_id]
            for agent_id, action in agent_actions.items():
                environment.receive_action(agent_id, action)

    def _get_time_steps(self, environment: VariableNumberAgentEnvironment) -> Dict[str, Dict[str, TimeStep]]:
        time_steps: Dict[str, Dict[str, TimeStep]] = {group_id: dict() for group_id in self._groups}
        for group_id in self._groups:
            time_steps[group_id] = environment.get_steps(group_id)
        return time_steps

    def _add_transitions_to_replay_buffers(
        self,
        time_steps: Dict[str, Dict[str, TimeStep]],
        actions: Dict[str, Dict[str, tf.Tensor]],
        next_time_steps: Dict[str, Dict[str, TimeStep]],
    ) -> int:
        num_added = 0
        for group_id, group in self._groups.items():
            if group_id not in time_steps or group_id not in actions or group_id not in next_time_steps:
                continue
            agent_time_steps = time_steps[group_id]
            agent_actions = actions[group_id]
            agent_next_time_steps = next_time_steps[group_id]
            for agent_id, time_step in agent_time_steps.items():
                if agent_id not in agent_actions or agent_id not in agent_next_time_steps or group.replay_buffer_size < 1:
                    break
                time_step = agent_time_steps[agent_id]
                action = agent_actions[agent_id]
                next_time_step = agent_next_time_steps[agent_id]
                transition = Transition(time_step, action, next_time_step)
                while len(group.replay_buffer) >= group.replay_buffer_size:
                    group.replay_buffer.remove_oldest()
                group.replay_buffer.append(transition)
                num_added += 1
        return num_added

    def _has_filled_replay_buffers(self) -> bool:
        for group in self._groups.values():
            if len(group.replay_buffer) < group.replay_buffer_size:
                return False
        return True

    def _is_last_step(self, time_steps: Dict[str, Dict[str, TimeStep]]) -> bool:
        for group_id in self._groups:
            if group_id not in time_steps:
                continue
            agent_time_steps = time_steps[group_id]
            for time_step in agent_time_steps.values():
                if time_step.step_type == STEP_TYPE_LAST:
                    return True
        return False

    def _update_episode_returns(
        self,
        time_steps: Dict[str, Dict[str, TimeStep]],
        episode_returns: Dict[str, tf.Tensor],
    ):
        for group_id in self._groups:
            if group_id not in time_steps:
                continue
            agent_time_steps = time_steps[group_id]
            reward_sum = tf.constant(0.0)
            for time_step in agent_time_steps.values():
                reward_sum += time_step.reward
            episode_returns[group_id] += reward_sum / len(agent_time_steps)
