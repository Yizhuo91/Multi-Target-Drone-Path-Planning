import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import tensorflow as tf

from drone_path_planning.environments import VariableNumberAgentEnvironment
from drone_path_planning.evaluators.evaluator import Evaluator
from drone_path_planning.utilities.agent_groups import VariableNumberAgentGroup
from drone_path_planning.utilities.time_step import STEP_TYPE_LAST
from drone_path_planning.utilities.time_step import TimeStep


class VariableNumberAgentDeepQNetworkEvaluator(Evaluator):
    def __init__(
        self,
        groups: Dict[str, VariableNumberAgentGroup],
        environment: VariableNumberAgentEnvironment,
        plot_data_dir: str,
        num_episodes: int,
        max_num_steps_per_episode: int,
        logs_dir: Optional[str] = None,
        tf_predict_step_input_signatures: Dict[str, Any] = dict(),
    ):
        self._groups = groups
        self._environment = environment
        self._plot_data_dir = plot_data_dir
        callbacks = []
        for group_id, group in self._groups.items():
            group_logs_dir = None if logs_dir is None else os.path.join(logs_dir, group_id)
            group_callbacks = self._create_callbacks(
                logs_dir=group_logs_dir,
            )
            for group_callback in group_callbacks:
                group_callback.set_model(group.agent)
            callbacks.extend(group_callbacks)
        self._callback_list_callback = tf.keras.callbacks.CallbackList(
            callbacks=callbacks,
        )
        self._num_episodes = num_episodes
        self._max_num_steps_per_episode = max_num_steps_per_episode
        self._tf_predict_step_input_signatures = tf_predict_step_input_signatures

    def initialize(self):
        for group in self._groups.values():
            group.agent.compile(**group.agent_compile_kwargs)

    def evaluate(self):
        tf_predict_steps = dict()
        for group_id, group in self._groups.items():
            tf_predict_step_input_signature = self._tf_predict_step_input_signatures[group_id]
            tf_predict_steps[group_id] = tf.function(
                group.agent.predict_step,
                input_signature=tf_predict_step_input_signature,
            )
        trajectories = []
        episode_returns = {group_id: tf.constant(0.0) for group_id in self._groups}
        test_logs: Dict[str, List[tf.Tensor]] = dict()
        self._callback_list_callback.on_test_begin()
        for i in range(self._num_episodes):
            environment_states = []
            self._environment.reset()
            time_steps = self._get_time_steps(self._environment)
            environment_state = self._environment.generate_state_data_for_plotting()
            environment_states.append(environment_state)
            self._update_episode_returns(time_steps, episode_returns)
            batch_logs = dict()
            step_count = 0
            self._callback_list_callback.on_test_batch_begin(i)
            for j in range(self._max_num_steps_per_episode):
                if self._is_last_step(time_steps):
                    break
                step_count = j + 1
                actions = self._choose_actions(tf_predict_steps, time_steps)
                self._apply_actions(self._environment, actions)
                self._environment.update()
                time_steps = self._get_time_steps(self._environment)
                environment_state = self._environment.generate_state_data_for_plotting()
                environment_states.append(environment_state)
                self._update_episode_returns(time_steps, episode_returns)
            batch_logs['step_count'] = float(step_count)
            for group_id in self._groups:
                batch_logs[f'{group_id}_return'] = episode_returns[group_id]
            self._callback_list_callback.on_test_batch_end(float(i), logs=batch_logs)
            for key, value in batch_logs.items():
                if key not in test_logs:
                    test_logs[key] = []
                test_logs[key].append(value)
            trajectories.append(environment_states)
        test_logs = {key: tf.reduce_mean(tf.stack(values)) for key, values in test_logs.items()}
        self._callback_list_callback.on_test_end(logs=test_logs)
        self._save_plot_data(self._plot_data_dir, trajectories)
        self._show_model_summaries()

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

    def _show_model_summaries(self):
        for group_id, group in self._groups.items():
            print(group_id)
            group.agent.summary(
                expand_nested=True,
            )
