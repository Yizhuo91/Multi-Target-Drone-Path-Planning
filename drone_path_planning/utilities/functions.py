from typing import Dict
from typing import Optional

import tensorflow as tf

from drone_path_planning.utilities.time_step import TimeStep
from drone_path_planning.utilities.training_helpers import Transition


def find_direction(angular_displacement: tf.Tensor) -> tf.Tensor:
    direction = tf.concat(
        [
            tf.math.cos(angular_displacement),
            tf.math.sin(angular_displacement),
            tf.zeros(tf.shape(angular_displacement)),
        ],
        axis=-1,
    )
    return direction


def find_cartesian_product(p: tf.Tensor, q: tf.Tensor) -> tf.Tensor:
    multiples = tf.ones_like(
        tf.concat([
            tf.shape(p),
            tf.constant([0]),
        ], axis=0),
    )
    horizontal_multiples = tf.tensor_scatter_nd_update(multiples, tf.constant([[1]]), tf.shape(q)[0:1])
    horizontally_tiled_p = tf.tile(tf.expand_dims(p, 1), horizontal_multiples)
    vertical_multiples = tf.tensor_scatter_nd_update(multiples, tf.constant([[0]]), tf.shape(p)[0:1])
    vertically_tiled_q = tf.tile(tf.expand_dims(q, 0), vertical_multiples)
    cartesian_product = tf.stack([horizontally_tiled_p, vertically_tiled_q], axis=2)
    return cartesian_product


def find_pairs_from_cartesian_product(cartesian_product: tf.Tensor) -> tf.Tensor:
    pairs_shape = tf.concat([
        tf.constant([-1]),
        tf.shape(cartesian_product)[2:],
    ], axis=0)
    pairs = tf.reshape(cartesian_product, pairs_shape)
    return pairs


def find_cartesian_square_pairs_with_distinct_elements(cartesian_square_pairs: tf.Tensor) -> tf.Tensor:
    size2 = tf.shape(cartesian_square_pairs)[0]
    size = tf.cast(tf.math.sqrt(tf.cast(size2, tf.dtypes.double)) + 0.5, tf.dtypes.int32)
    mask = tf.range(size * size) % (size + 1) != 0
    pairs_with_distinct_elements = tf.boolean_mask(cartesian_square_pairs, mask)
    return pairs_with_distinct_elements


def find_relative_quantities(p_quantities: tf.Tensor, q_quantities: Optional[tf.Tensor] = None) -> tf.Tensor:
    pairs_with_distinct_elements: tf.Tensor
    if q_quantities is None:
        cartesian_product = find_cartesian_product(p_quantities, -p_quantities)
        pairs = find_pairs_from_cartesian_product(cartesian_product)
        pairs_with_distinct_elements = find_cartesian_square_pairs_with_distinct_elements(pairs)
    else:
        cartesian_product = find_cartesian_product(p_quantities, -q_quantities)
        pairs_with_distinct_elements = find_pairs_from_cartesian_product(cartesian_product)
    relative_quantities = tf.math.reduce_sum(pairs_with_distinct_elements, axis=1)
    return relative_quantities


def create_time_step_spec(observation_spec: Dict[str, tf.TensorSpec]) -> TimeStep:
    time_step_spec = TimeStep(
        tf.TensorSpec([], dtype=tf.dtypes.int32),
        tf.TensorSpec([]),
        observation_spec,
    )
    return time_step_spec


def create_transition_spec(observation_spec: Dict[str, tf.TensorSpec]) -> Transition:
    time_step_spec = create_time_step_spec(observation_spec)
    transition_spec = Transition(
        time_step_spec,
        tf.TensorSpec([], dtype=tf.dtypes.int32),
        time_step_spec,
    )
    return transition_spec
