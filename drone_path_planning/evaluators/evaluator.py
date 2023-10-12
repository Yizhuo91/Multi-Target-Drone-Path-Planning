import abc
import os
import pickle
from typing import List
from typing import Optional

import tensorflow as tf


class Evaluator:
    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(self):
        raise NotImplementedError()

    def _create_callbacks(
        self,
        logs_dir: Optional[str] = None,
    ) -> List[tf.keras.callbacks.Callback]:
        callbacks = []
        progbar_logger_callback = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps',
            stateful_metrics=['epsilon', 'num_episodes'],
        )
        progbar_logger_callback_params = {
            'verbose': 1,
            'epochs': 1,
        }
        progbar_logger_callback.set_params(progbar_logger_callback_params)
        callbacks.append(progbar_logger_callback)
        if logs_dir is not None:
            os.makedirs(logs_dir, exist_ok=True)
            logs_filepath = os.path.join(logs_dir, 'evaluation_logs.csv')
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

    def _save_plot_data(self, plot_data_dir: str, plot_data):
        os.makedirs(plot_data_dir, exist_ok=True)
        plot_data_filepath = os.path.join(plot_data_dir, 'plot_data.pkl')
        with open(plot_data_filepath, 'wb') as file:
            pickle.dump(plot_data, file)
