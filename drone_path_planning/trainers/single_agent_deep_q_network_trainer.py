from typing import Dict
from typing import List
from typing import Optional

import tensorflow as tf

from drone_path_planning.agents import DeepQNetworkAgent
from drone_path_planning.environments import Environment
from drone_path_planning.trainers.trainer import Trainer
from drone_path_planning.utilities.time_step import STEP_TYPE_LAST
from drone_path_planning.utilities.training_helpers import ReplayBuffer
from drone_path_planning.utilities.training_helpers import Transition


_NOT_NONE_VAL_PARAMETERS_ASSERTION_ERROR_MESSAGE = 'All validation parameters should be None if validation is not to be done (the validation Environment is None).'


class SingleAgentDeepQNetworkTrainer(Trainer):
    def __init__(
        self,
        agent: DeepQNetworkAgent,
        optimizer: tf.keras.optimizers.Optimizer,
        environment: Environment,
        replay_buffer: ReplayBuffer,
        replay_buffer_size: int,
        num_epochs: int,
        num_steps_per_epoch: int,
        max_num_steps_per_episode: int,
        save_dir: str,
        validation_environment: Optional[Environment],
        num_val_episodes: Optional[int] = None,
        max_num_steps_per_val_episode: Optional[int] = None,
        logs_dir: Optional[str] = None,
    ):
        self._agent = agent
        self._optimizer = optimizer
        self._environment = environment
        self._replay_buffer = replay_buffer
        self._replay_buffer_size = replay_buffer_size
        callbacks = self._create_training_callbacks(
            num_epochs,
            num_steps_per_epoch,
            save_dir,
            logs_dir=logs_dir,
        )
        self._callback_list_callback = tf.keras.callbacks.CallbackList(
            callbacks=callbacks,
            model=self._agent,
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

    def initialize(self):
        self._replay_buffer.fill(tf.function(self._agent.collect_step), self._environment, self._replay_buffer_size - 1, self._max_num_steps_per_episode)
        self._agent.compile(optimizer=self._optimizer)

    def train(self):
        tf_train_step = tf.function(self._agent.train_step)
        tf_collect_step = tf.function(self._agent.collect_step)
        time_step = self._environment.reset()
        k = 0
        num_episodes = 0
        training_logs: Dict[str, tf.Tensor]
        self._callback_list_callback.on_train_begin()
        for i in range(self._num_epochs):
            epoch_logs: Dict[str, tf.Tensor]
            self._callback_list_callback.on_epoch_begin(i)
            for j in range(self._num_steps_per_epoch):
                action = tf_collect_step(time_step)
                next_time_step = self._environment.step(action)
                transition = Transition(time_step, action, next_time_step)
                self._replay_buffer.append(transition)
                k += 1
                if k < self._max_num_steps_per_episode and time_step.step_type != STEP_TYPE_LAST:
                    time_step = next_time_step
                else:
                    time_step = self._environment.reset()
                    k = 0
                    num_episodes += 1
                training_transition = self._replay_buffer.sample()
                self._callback_list_callback.on_train_batch_begin(j)
                batch_logs = tf_train_step(training_transition)
                batch_logs['num_episodes'] = num_episodes
                self._callback_list_callback.on_train_batch_end(float(j), logs=batch_logs)
                self._agent.update_epsilon()
                self._agent.update_target_model()
                self._replay_buffer.remove_oldest()
                epoch_logs = batch_logs
            if self._validation_environment is not None:
                val_logs = self._validate()
                epoch_logs.update({f'val_{key}': value for key, value in val_logs.items()})
            self._callback_list_callback.on_epoch_end(i, logs=epoch_logs)
            training_logs = epoch_logs
        self._callback_list_callback.on_train_end(logs=training_logs)

    def _validate(self) -> Dict[str, List[tf.Tensor]]:
        test_logs: Dict[str, List[tf.Tensor]] = dict()
        self._callback_list_callback.on_test_begin()
        for i in range(self._num_val_episodes):
            self._callback_list_callback.on_test_batch_begin(i)
            batch_logs = self._agent.evaluate_step(self._validation_environment, self._max_num_steps_per_val_episode)
            self._callback_list_callback.on_test_batch_end(float(i), logs=batch_logs)
            for key, value in batch_logs.items():
                if key not in test_logs:
                    test_logs[key] = []
                test_logs[key].append(value)
        test_logs = {key: tf.reduce_mean(tf.stack(values)) for key, values in test_logs.items()}
        self._callback_list_callback.on_test_end(logs=test_logs)
        return test_logs
