import enum
from typing import Dict
from typing import NamedTuple

import tensorflow as tf


STEP_TYPE_FIRST: tf.Tensor = tf.constant(0)
STEP_TYPE_MID: tf.Tensor = tf.constant(1)
STEP_TYPE_LAST: tf.Tensor = tf.constant(2)


class TimeStep(NamedTuple):
    step_type: tf.Tensor
    reward: tf.Tensor
    observation: Dict[str, tf.Tensor]
