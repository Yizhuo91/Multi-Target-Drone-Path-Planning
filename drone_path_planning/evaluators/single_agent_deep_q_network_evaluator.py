from typing import Dict
from typing import List
from typing import Optional

import tensorflow as tf

from drone_path_planning.agents import DeepQNetworkAgent
from drone_path_planning.environments import Environment
from drone_path_planning.evaluators.evaluator import Evaluator
from drone_path_planning.utilities.time_step import STEP_TYPE_LAST


class SingleAgentDeepQNetworkEvaluator(Evaluator):
    def __init__(
        self,
        agent: DeepQNetworkAgent,
        environment: Environment,
        plot_data_dir: str,
        num_episodes: int,
        max_num_steps_per_episode: int,
        logs_dir: Optional[str] = None,
    ):
        self._agent = agent
        self._environment = environment
        self._plot_data_dir = plot_data_dir
        callbacks = self._create_callbacks(
            logs_dir=logs_dir,
        )
        self._callback_list_callback = tf.keras.callbacks.CallbackList(
            callbacks=callbacks,
            model=self._agent,
        )
        self._num_episodes = num_episodes
        self._max_num_steps_per_episode = max_num_steps_per_episode

    def initialize(self):
        self._agent.compile()

    def evaluate(self):
        tf_predict_step = tf.function(self._agent.predict_step)
        trajectories = []
        test_logs: Dict[str, List[tf.Tensor]] = dict()
        self._callback_list_callback.on_test_begin()
        for i in range(self._num_episodes):
            environment_states = []
            time_step = self._environment.reset()
            environment_state = self._environment.generate_state_data_for_plotting()
            environment_states.append(environment_state)
            rewards = []
            rewards.append(time_step.reward)
            episode_return = time_step.reward
            batch_logs = dict()
            step_count: int
            self._callback_list_callback.on_test_batch_begin(i)
            for j in range(self._max_num_steps_per_episode):
                step_count = j + 1
                if time_step.step_type == STEP_TYPE_LAST:
                    break
                action = tf_predict_step(time_step)
                time_step = self._environment.step(action)
                environment_state = self._environment.generate_state_data_for_plotting()
                environment_states.append(environment_state)
                rewards.append(time_step.reward)
                episode_return += time_step.reward
            batch_logs['step_count'] = float(step_count)
            batch_logs['return'] = episode_return
            self._callback_list_callback.on_test_batch_end(float(i), logs=batch_logs)
            for key, value in batch_logs.items():
                if key not in test_logs:
                    test_logs[key] = []
                test_logs[key].append(value)
            trajectories.append(environment_states)
        test_logs = {key: tf.reduce_mean(tf.stack(values)) for key, values in test_logs.items()}
        self._callback_list_callback.on_test_end(logs=test_logs)
        self._save_plot_data(self._plot_data_dir, trajectories)
