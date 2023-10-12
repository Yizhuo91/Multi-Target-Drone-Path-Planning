from typing import NamedTuple

import tensorflow as tf

from drone_path_planning.utilities.time_step import TimeStep


class Transition(NamedTuple):
    time_step: TimeStep
    action: tf.Tensor
    next_time_step: TimeStep
