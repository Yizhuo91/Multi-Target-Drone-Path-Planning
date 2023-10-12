import abc
from typing import Dict

import tensorflow as tf

from drone_path_planning.utilities.time_step import TimeStep


class VariableNumberAgentEnvironment:
    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def receive_action(self, agent_id: str, action: tf.Tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_steps(self, group_id: str) -> Dict[str, TimeStep]:
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_state_data_for_plotting(self) -> Dict[str, tf.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_observation_spec(self) -> Dict[str, tf.TensorSpec]:
        raise NotImplementedError()
