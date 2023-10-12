import math
from typing import Dict

import tensorflow as tf

from drone_path_planning.environments.environment import Environment
from drone_path_planning.utilities.constants import ANTI_CLOCKWISE
from drone_path_planning.utilities.constants import BACKWARD
from drone_path_planning.utilities.constants import CLOCKWISE
from drone_path_planning.utilities.constants import FORWARD
from drone_path_planning.utilities.constants import REST
from drone_path_planning.utilities.constants import SELF_ANGULAR_VELOCITY
from drone_path_planning.utilities.constants import CHASER_DIRECTIONS
from drone_path_planning.utilities.constants import CHASER_DISPLACEMENTS
from drone_path_planning.utilities.constants import TARGET_ANGULAR_VELOCITY
from drone_path_planning.utilities.constants import TARGET_DIRECTIONS
from drone_path_planning.utilities.constants import TARGET_DISPLACEMENTS
from drone_path_planning.utilities.constants import TARGET_RELATIVE_DISPLACEMENT
from drone_path_planning.utilities.constants import TARGET_RELATIVE_VELOCITY
from drone_path_planning.utilities.functions import find_direction
from drone_path_planning.utilities.time_step import STEP_TYPE_FIRST
from drone_path_planning.utilities.time_step import STEP_TYPE_LAST
from drone_path_planning.utilities.time_step import STEP_TYPE_MID
from drone_path_planning.utilities.time_step import TimeStep


HEIGHT = 4.0
WIDTH = 4.0
POSITIVE_Z = tf.constant([[0.0, 0.0, 1.0]])
ACCELERATION_NORM = 1.0
ANGULAR_ACCELERATION_NORM = 1.0
TARGET_MIN_DISTANCE = 0.25
MASS = 1.0
TERMINAL_SPEED = 1.0
DRAG_COEFFICIENT = MASS * ACCELERATION_NORM / TERMINAL_SPEED ** 2
MOMENT_OF_INERTIA = 1.0
TERMINAL_ANGULAR_SPEED = math.pi / 4.0
ANGULAR_DRAG_COEFFICIENT = MOMENT_OF_INERTIA * ANGULAR_ACCELERATION_NORM / TERMINAL_ANGULAR_SPEED ** 2
DELTA_T = 0.1
TARGET_REWARD = 4.0


TARGET_MASS = 0.25
TARGET_MOMENT_OF_INERTIA = 0.5
TARGET_TERMINAL_SPEED = 0.25
TARGET_ACCELERATION_NORM = DRAG_COEFFICIENT * TARGET_TERMINAL_SPEED ** 2 / TARGET_MASS
TARGET_TERMINAL_ANGULAR_SPEED = math.pi / 4.0
TARGET_ANGULAR_ACCELERATION_NORM = ANGULAR_DRAG_COEFFICIENT * TARGET_TERMINAL_ANGULAR_SPEED ** 2 / TARGET_MOMENT_OF_INERTIA


ACTION_BRANCHES = {
    REST: 0,
    FORWARD: 1,
    BACKWARD: 2,
    ANTI_CLOCKWISE: 3,
    CLOCKWISE: 4,
}


class SingleChaserSingleMovingTargetEnvironment(Environment):
    def __init__(self):
        self._POSSIBLE_ACTIONS = {
            ACTION_BRANCHES[REST]: self._rest,
            ACTION_BRANCHES[FORWARD]: self._move_forward,
            ACTION_BRANCHES[BACKWARD]: self._move_backward,
            ACTION_BRANCHES[ANTI_CLOCKWISE]: self._turn_anti_clockwise,
            ACTION_BRANCHES[CLOCKWISE]: self._turn_clockwise,
        }
        self._displacement: tf.Tensor
        self._velocity: tf.Tensor
        self._acceleration: tf.Tensor
        self._angular_displacement: tf.Tensor
        self._angular_velocity: tf.Tensor
        self._angular_acceleration: tf.Tensor
        self._target_displacement: tf.Tensor
        self._target_velocity: tf.Tensor
        self._target_angular_displacement: tf.Tensor
        self._target_angular_velocity: tf.Tensor
        self._old_observation: tf.Tensor

    def generate_state_data_for_plotting(self) -> Dict[str, tf.Tensor]:
        state_data = dict()
        state_data[CHASER_DIRECTIONS] = find_direction(self._angular_displacement)
        state_data[CHASER_DISPLACEMENTS] = self._displacement
        state_data[TARGET_DIRECTIONS] = find_direction(self._target_angular_displacement)
        state_data[TARGET_DISPLACEMENTS] = self._target_displacement
        return state_data

    def _reset(self) -> TimeStep:
        self._initialise_state()
        observation = self._get_observation()
        reward = self._get_first_reward()
        first_time_step = TimeStep(STEP_TYPE_FIRST, reward, observation)
        self._old_observation = observation
        return first_time_step

    def _step(self, action: tf.Tensor) -> TimeStep:
        tf.switch_case(action, self._POSSIBLE_ACTIONS)
        self._update()
        observation: Dict = self._get_observation()
        if self._has_reached_target():
            time_step = TimeStep(STEP_TYPE_LAST, tf.constant(TARGET_REWARD), observation)
            return time_step
        reward = self._get_reward(action, observation)
        time_step = TimeStep(STEP_TYPE_MID, reward, observation)
        self._old_observation = observation
        return time_step

    def _initialise_state(self):
        self._displacement = tf.concat([
            tf.random.uniform([1, 1], maxval=WIDTH),
            tf.random.uniform([1, 1], maxval=HEIGHT),
            tf.zeros([1, 1]),
        ], axis=-1)
        self._velocity = tf.zeros([1, 3])
        self._acceleration = tf.zeros([1, 3])
        self._angular_displacement = tf.random.uniform(
            [1, 1],
            minval=-math.pi,
            maxval=math.pi,
        )
        self._angular_velocity = tf.zeros([1, 1])
        self._angular_acceleration = tf.zeros([1, 1])
        self._target_displacement = tf.concat([
            tf.random.uniform([1, 1], maxval=WIDTH),
            tf.random.uniform([1, 1], maxval=HEIGHT),
            tf.zeros([1, 1]),
        ], axis=-1)
        while self._has_reached_target():
            self._target_displacement = tf.concat([
                tf.random.uniform([1, 1], maxval=WIDTH),
                tf.random.uniform([1, 1], maxval=HEIGHT),
                tf.zeros([1, 1]),
            ], axis=-1)
        self._target_velocity = tf.zeros([1, 3])
        self._target_acceleration = tf.zeros([1, 3])
        self._target_angular_displacement = tf.random.uniform(
            [1, 1],
            minval=-math.pi,
            maxval=math.pi,
        )
        self._target_angular_velocity = tf.zeros([1, 1])
        self._target_angular_acceleration = tf.zeros([1, 1])

    def _has_reached_target(self) -> tf.Tensor:
        target_relative_displacement = self._target_displacement - self._displacement
        target_relative_distance = tf.reduce_min(tf.norm(target_relative_displacement, axis=-1))
        return target_relative_distance < TARGET_MIN_DISTANCE

    def _get_observation(self) -> Dict[str, tf.Tensor]:
        direction = find_direction(self._angular_displacement)
        change_of_basis_matrix = self._find_change_of_basis_matrix(direction)
        target_relative_velocity = self._target_velocity - self._velocity
        target_relative_displacement = self._target_displacement - self._displacement
        transformed_target_relative_velocity = self._change_basis(change_of_basis_matrix, target_relative_velocity)
        transformed_target_relative_displacement = self._change_basis(change_of_basis_matrix, target_relative_displacement)
        observation: Dict[str, tf.Tensor] = {
            SELF_ANGULAR_VELOCITY: self._angular_velocity,
            TARGET_RELATIVE_VELOCITY: transformed_target_relative_velocity,
            TARGET_ANGULAR_VELOCITY: self._target_angular_velocity,
            TARGET_RELATIVE_DISPLACEMENT: transformed_target_relative_displacement,
        }
        return observation

    def _get_first_reward(self) -> tf.Tensor:
        reward = 0.0
        return reward

    def _get_reward(self, action: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        reward = 0.0
        time_reward = self._get_time_reward()
        reward += time_reward
        distance_reward = self._get_distance_reward(observation)
        reward += distance_reward
        return reward

    def _get_time_reward(self) -> tf.Tensor:
        reward = -DELTA_T
        return reward

    def _get_distance_reward(self, observation: tf.Tensor) -> tf.Tensor:
        old_target_relative_displacement = self._old_observation[TARGET_RELATIVE_DISPLACEMENT]
        old_target_relative_distance = tf.reduce_min(tf.norm(old_target_relative_displacement, axis=-1))
        target_relative_displacement = observation[TARGET_RELATIVE_DISPLACEMENT]
        target_relative_distance = tf.reduce_min(tf.norm(target_relative_displacement, axis=-1))
        score = old_target_relative_distance - target_relative_distance
        reward = score
        return reward

    def _rest(self):
        direction = find_direction(self._angular_displacement)
        self._acceleration = tf.zeros_like(direction)
        self._angular_acceleration = tf.zeros_like(self._angular_displacement)

    def _move_forward(self):
        direction = find_direction(self._angular_displacement)
        self._acceleration = ACCELERATION_NORM * direction
        self._angular_acceleration = tf.zeros_like(self._angular_displacement)

    def _move_backward(self):
        direction = find_direction(self._angular_displacement)
        self._acceleration = -ACCELERATION_NORM * direction
        self._angular_acceleration = tf.zeros_like(self._angular_displacement)

    def _turn_anti_clockwise(self):
        direction = find_direction(self._angular_displacement)
        self._acceleration = tf.zeros_like(direction)
        self._angular_acceleration = ANGULAR_ACCELERATION_NORM * tf.ones_like(self._angular_displacement)

    def _turn_clockwise(self):
        direction = find_direction(self._angular_displacement)
        self._acceleration = tf.zeros_like(direction)
        self._angular_acceleration = -ANGULAR_ACCELERATION_NORM * tf.ones_like(self._angular_displacement)

    def _update(self):
        self._update_self()
        self._update_target()

    def _update_self(self):
        drag_acceleration = self._find_drag_acceleration(self._velocity, self._acceleration, MASS, DRAG_COEFFICIENT)
        net_acceleration = self._acceleration + drag_acceleration
        self._velocity = self._velocity + net_acceleration * DELTA_T
        self._displacement = self._displacement + self._velocity * DELTA_T
        angular_drag_acceleration = self._find_drag_acceleration(self._angular_velocity, self._angular_acceleration, MOMENT_OF_INERTIA, ANGULAR_DRAG_COEFFICIENT)
        net_angular_acceleration = self._angular_acceleration + angular_drag_acceleration
        self._angular_velocity = self._angular_velocity + net_angular_acceleration * DELTA_T
        self._angular_displacement = tf.math.floormod(self._angular_displacement + self._angular_velocity * DELTA_T, 2 * math.pi)

    def _update_target(self):
        target_relative_displacement = self._target_displacement - self._displacement
        direction = find_direction(self._target_angular_displacement)
        dot_product = tf.reduce_sum(target_relative_displacement * direction, axis=-1, keepdims=True)
        self._target_acceleration = tf.sign(dot_product) * direction * TARGET_ACCELERATION_NORM
        sin_relative_angular_displacement = tf.clip_by_value(tf.linalg.cross(direction, target_relative_displacement)[:, -1:] / tf.norm(target_relative_displacement, axis=-1), -1.0, 1.0)
        relative_angular_displacement = tf.math.asin(sin_relative_angular_displacement)
        self._target_angular_acceleration = relative_angular_displacement / math.pi * TARGET_ANGULAR_ACCELERATION_NORM
        drag_acceleration = self._find_drag_acceleration(self._target_velocity, self._target_acceleration, TARGET_MASS, DRAG_COEFFICIENT)
        net_acceleration = self._target_acceleration + drag_acceleration
        self._target_velocity = self._target_velocity + net_acceleration * DELTA_T
        self._target_displacement = self._target_displacement + self._target_velocity * DELTA_T
        angular_drag_acceleration = self._find_drag_acceleration(self._target_angular_velocity, self._target_angular_acceleration, TARGET_MOMENT_OF_INERTIA, ANGULAR_DRAG_COEFFICIENT)
        net_angular_acceleration = self._target_angular_acceleration + angular_drag_acceleration
        self._target_angular_velocity = self._target_angular_velocity + net_angular_acceleration * DELTA_T
        self._target_angular_displacement = tf.math.floormod(self._target_angular_displacement + self._target_angular_velocity * DELTA_T, 2 * math.pi)

    def _find_drag_acceleration(self, velocity: tf.Tensor, acceleration: tf.Tensor, mass: tf.Tensor, drag_coefficient: tf.Tensor) -> tf.Tensor:
        velocity_norm = tf.norm(velocity, axis=-1)
        if velocity_norm == 0.0:
            drag_acceleration = tf.zeros_like(acceleration)
            return drag_acceleration
        direction = velocity / velocity_norm
        drag_magnitude = tf.reduce_sum(velocity ** 2, axis=-1, keepdims=True) * drag_coefficient
        drag = -direction * drag_magnitude
        drag_acceleration = drag / mass
        return drag_acceleration

    def _find_change_of_basis_matrix(self, direction: tf.Tensor) -> tf.Tensor:
        perpendicular_basis = tf.linalg.cross(POSITIVE_Z, direction)
        normalised_perpendicular_basis = perpendicular_basis / tf.norm(perpendicular_basis, axis=-1)
        change_of_basis_matrix = tf.stack([
            direction,
            normalised_perpendicular_basis,
            POSITIVE_Z,
        ], axis=-2)
        return change_of_basis_matrix

    def _change_basis(self, change_of_basis_matrix: tf.Tensor, vector: tf.Tensor):
        expanded_transformed_vector = tf.tensordot(vector, change_of_basis_matrix, [[1], [2]])
        transformed_vector = tf.squeeze(expanded_transformed_vector, axis=1)
        return transformed_vector
