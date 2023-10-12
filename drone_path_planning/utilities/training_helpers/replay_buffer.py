from collections import deque
import random
from typing import Callable

import tensorflow as tf

from drone_path_planning.environments import Environment
from drone_path_planning.utilities.time_step import STEP_TYPE_LAST
from drone_path_planning.utilities.time_step import TimeStep
from drone_path_planning.utilities.training_helpers.transition import Transition


_NUM_EPISODES: str = 'num_episodes'


class ReplayBuffer:
    def __init__(self):
        self._transitions = deque()
        self._progbar_logger_callback = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps',
            stateful_metrics=[_NUM_EPISODES],
        )

    def fill(self, policy: Callable[[TimeStep], tf.Tensor], environment: Environment, num_transitions: int, max_num_steps_per_episode: int) -> TimeStep:
        self._update_progbar_logger_params(num_transitions)
        time_step = environment.reset()
        j = 0
        num_episodes = 0
        self._progbar_logger_callback.on_test_begin()
        for i in range(num_transitions):
            self._progbar_logger_callback.on_test_batch_begin(i)
            action = policy(time_step)
            next_time_step = environment.step(action)
            transition = Transition(time_step, action, next_time_step)
            self.append(transition)
            j += 1
            if j < max_num_steps_per_episode and time_step.step_type != STEP_TYPE_LAST:
                time_step = next_time_step
            else:
                time_step = environment.reset()
                j = 0
                num_episodes += 1
            logs = {_NUM_EPISODES: num_episodes}
            self._progbar_logger_callback.on_test_batch_end(float(i), logs=logs)
        self._progbar_logger_callback.on_test_end()
        return time_step

    def append(self, transition: Transition):
        self._transitions.append(transition)

    def sample(self) -> Transition:
        transition = random.choice(self._transitions)
        return transition

    def remove_oldest(self) -> Transition:
        oldest = self._transitions.popleft()
        return oldest

    def _update_progbar_logger_params(self, num_steps: int):
        progbar_logger_callback_params = {
            'verbose': 1,
            'epochs': 1,
            'steps': num_steps,
        }
        self._progbar_logger_callback.set_params(progbar_logger_callback_params)

    def __len__(self):
        return len(self._transitions)
