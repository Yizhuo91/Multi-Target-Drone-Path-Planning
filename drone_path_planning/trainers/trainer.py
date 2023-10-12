import abc
import os
from typing import List
from typing import Optional

import tensorflow as tf


class Trainer:
    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError()

    def _create_training_callbacks(
        self,
        num_epochs: int,
        num_steps_per_epoch: int,
        save_dir: str,
        logs_dir: Optional[str] = None,
    ) -> List[tf.keras.callbacks.Callback]:
        callbacks = []
        progbar_logger_callback = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps',
            stateful_metrics=['epsilon', 'num_episodes'],
        )
        progbar_logger_callback_params = {
            'verbose': 1,
            'epochs': num_epochs,
            'steps': num_steps_per_epoch,
        }
        progbar_logger_callback.set_params(progbar_logger_callback_params)
        callbacks.append(progbar_logger_callback)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            save_dir,
        )
        callbacks.append(model_checkpoint_callback)
        if logs_dir is not None:
            os.makedirs(logs_dir, exist_ok=True)
            logs_filepath = os.path.join(logs_dir, 'training_logs.csv')
            csv_logger_callback = tf.keras.callbacks.CSVLogger(
                logs_filepath,
                append=True,
            )
            callbacks.append(csv_logger_callback)
            tensor_board_callback = tf.keras.callbacks.TensorBoard(
                logs_dir,
                histogram_freq=1,
                write_steps_per_second=True,
                profile_batch=(10, 20),
            )
            callbacks.append(tensor_board_callback)
        return callbacks
