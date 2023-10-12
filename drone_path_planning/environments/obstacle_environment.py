import bisect
import math
import random
from typing import Dict
from typing import Set
from typing import Optional
from typing import Tuple

import tensorflow as tf

from drone_path_planning.environments.variable_number_agent_environment import VariableNumberAgentEnvironment
from drone_path_planning.utilities.constants import AGENT_ANGULAR_VELOCITIES
from drone_path_planning.utilities.constants import ANTI_CLOCKWISE
from drone_path_planning.utilities.constants import BACKWARD
from drone_path_planning.utilities.constants import CHASER_ANGULAR_VELOCITIES
from drone_path_planning.utilities.constants import CHASER_DIRECTIONS
from drone_path_planning.utilities.constants import CHASER_DISPLACEMENTS
from drone_path_planning.utilities.constants import CHASER_RELATIVE_DISPLACEMENTS
from drone_path_planning.utilities.constants import CHASER_RELATIVE_VELOCITIES
from drone_path_planning.utilities.constants import CLOCKWISE
from drone_path_planning.utilities.constants import DOWN
from drone_path_planning.utilities.constants import FORWARD
from drone_path_planning.utilities.constants import HAS_COLLIDED
from drone_path_planning.utilities.constants import HAS_CRASHED
from drone_path_planning.utilities.constants import HAS_INTERCEPTED
from drone_path_planning.utilities.constants import OBSTACLE_DISPLACEMENTS
from drone_path_planning.utilities.constants import OBSTACLE_RELATIVE_DISPLACEMENTS
from drone_path_planning.utilities.constants import REST
from drone_path_planning.utilities.constants import TARGET_ANGULAR_VELOCITIES
from drone_path_planning.utilities.constants import TARGET_DIRECTIONS
from drone_path_planning.utilities.constants import TARGET_DISPLACEMENTS
from drone_path_planning.utilities.constants import TARGET_RELATIVE_DISPLACEMENTS
from drone_path_planning.utilities.constants import TARGET_RELATIVE_VELOCITIES
from drone_path_planning.utilities.constants import TIME
from drone_path_planning.utilities.constants import UP
from drone_path_planning.utilities.functions import find_cartesian_product
from drone_path_planning.utilities.functions import find_direction
from drone_path_planning.utilities.functions import find_relative_quantities
from drone_path_planning.utilities.time_step import STEP_TYPE_FIRST
from drone_path_planning.utilities.time_step import STEP_TYPE_LAST
from drone_path_planning.utilities.time_step import STEP_TYPE_MID
from drone_path_planning.utilities.time_step import TimeStep


_CHASER_ANGULAR_DISPLACEMENT_KEY: str = 'chaser_angular_displacement'
_CHASER_ACCELERATION_NORM_KEY: str = 'chaser_acceleration_norm'
_CHASER_ANGULAR_ACCELERATION_NORM_KEY: str = 'chaser_angular_acceleration_norm'
_MASSES: str = 'masses'
_MOMENTS_OF_INERTIA: str = 'moments_of_inertia'
_DISPLACEMENTS: str = 'displacements'
_VELOCITIES: str = 'd/dt_{quantity}'.format(quantity=_DISPLACEMENTS)
_APPLIED_ACCELERATIONS: str = 'applied_accelerations'
_ANGULAR_DISPLACEMENTS: str = 'angular_displacements'
_ANGULAR_VELOCITIES: str = 'd/dt_{quantity}'.format(quantity=_ANGULAR_DISPLACEMENTS)
_APPLIED_ANGULAR_ACCELERATIONS: str = 'applied_angular_accelerations'


_CHASER_ID_TEMPLATE: str = '{group:s}_{index:d}'


_CENTER_BOUNDS_WIDTH: float = 128.0
_CENTER_BOUNDS_HEIGHT: float = 128.0
_CENTER_BOUNDS_DEPTH: float = 128.0
_OBSTACLE_BOUNDS_WIDTH: float = 64.0
_OBSTACLE_BOUNDS_HEIGHT: float = 64.0
_OBSTACLE_BOUNDS_DEPTH: float = 64.0
_INITIAL_BOUNDS_WIDTH: float = 4.0
_INITIAL_BOUNDS_HEIGHT: float = 4.0
_INITIAL_BOUNDS_DEPTH: float = 4.0
_DELTA_T: float = 0.1
_CHASER_SELF_COLLISION_DISTANCE: float = 0.5
_CRASHING_DISTANCE: float = 0.5
_CATCHING_DISTANCE: float = 0.25
_DRAG_COEFFICIENT: float = 1.0
_ANGULAR_DRAG_COEFFICIENT: float = 16.0 / math.pi ** 2
_ACTION_BRANCHES = {
    REST: 0,
    FORWARD: 1,
    BACKWARD: 2,
    ANTI_CLOCKWISE: 3,
    CLOCKWISE: 4,
    UP: 5,
    DOWN: 6,
}
_CRASHING_REWARD = -32.0
_COLLISION_REWARD = -32.0
_CATCHING_REWARD = 4.0


_CHASER_MASS: float = 1.0
_CHASER_TERMINAL_SPEED: float = 1.0
_CHASER_ACCELERATION_NORM: float = _DRAG_COEFFICIENT * _CHASER_TERMINAL_SPEED ** 2 / _CHASER_MASS
_CHASER_MOMENT_OF_INERTIA: float = 1.0
_CHASER_TERMINAL_ANGULAR_SPEED: float = math.pi / 4.0
_CHASER_ANGULAR_ACCELERATION_NORM: float = _ANGULAR_DRAG_COEFFICIENT * _CHASER_TERMINAL_ANGULAR_SPEED ** 2 / _CHASER_MOMENT_OF_INERTIA


_TARGET_MASS: float = 0.25
_TARGET_TERMINAL_SPEED: float = 0.25
_TARGET_ACCELERATION_NORM: float = _DRAG_COEFFICIENT * _TARGET_TERMINAL_SPEED ** 2 / _TARGET_MASS
_TARGET_MOMENT_OF_INERTIA: float = 0.5
_TARGET_TERMINAL_ANGULAR_SPEED: float = math.pi / 4.0
_TARGET_ANGULAR_ACCELERATION_NORM: float = _ANGULAR_DRAG_COEFFICIENT * _TARGET_TERMINAL_ANGULAR_SPEED ** 2 / _TARGET_MOMENT_OF_INERTIA


class ObstacleEnvironment(VariableNumberAgentEnvironment):
    def __init__(
        self,
        group_min_max_nums: Dict[str, Tuple[int, int]],
        min_num_obstacles: int = 0,
        max_num_obstacles: int = 0,
    ):
        self._min_num_obstacles = min_num_obstacles
        self._max_num_obstacles = max_num_obstacles
        self._group_min_max_nums = group_min_max_nums
        num_targets = 1
        self._target_masses = tf.fill([num_targets, 1], _TARGET_MASS)
        self._target_acceleration_norms = tf.fill([num_targets, 1], _TARGET_ACCELERATION_NORM)
        self._target_moments_of_inertia = tf.fill([num_targets, 1], _TARGET_MOMENT_OF_INERTIA)
        self._target_angular_acceleration_norms = tf.fill([num_targets, 1], _TARGET_ANGULAR_ACCELERATION_NORM)
        self._time = tf.constant(0.)
        self._has_updated = False
        self._has_crashed = tf.constant(False)
        self._has_collided = tf.constant(False)
        self._has_intercepted = tf.constant(False)

    def reset(self):
        center_x = (random.random() - 0.5) * _CENTER_BOUNDS_WIDTH
        center_y = (random.random() - 0.5) * _CENTER_BOUNDS_HEIGHT
        center_z = (random.random() - 0.5) * _CENTER_BOUNDS_DEPTH
        num_obstacles = random.randint(self._min_num_obstacles, self._max_num_obstacles)
        self._obstacle_displacements = self._create_initial_obstacle_displacements(num_obstacles, center_x=center_x, center_y=center_y, center_z=center_z)
        num_chasers = 0
        self._chaser_ids: Set[str] = set()
        self._group_chaser_ids: Dict[str, Set[str]] = dict()
        for group, (min_num, max_num) in self._group_min_max_nums.items():
            group_chaser_count = random.randrange(min_num, max_num + 1)
            num_chasers += group_chaser_count
            self._group_chaser_ids[group] = set()
            for i in range(group_chaser_count):
                chaser_id = _CHASER_ID_TEMPLATE.format(group=group, index=i)
                self._chaser_ids.add(chaser_id)
                self._group_chaser_ids[group].add(chaser_id)
        self._chaser_masses = tf.fill([num_chasers, 1], _CHASER_MASS)
        self._chaser_acceleration_norms = tf.fill([num_chasers, 1], _CHASER_ACCELERATION_NORM)
        self._chaser_moments_of_inertia = tf.fill([num_chasers, 1], _CHASER_MOMENT_OF_INERTIA)
        self._chaser_angular_acceleration_norms = tf.fill([num_chasers, 1], _CHASER_ANGULAR_ACCELERATION_NORM)
        self._chaser_displacements = self._create_initial_chaser_displacements(1, center_x=center_x, center_y=center_y, center_z=center_z)
        while self._has_any_within_threshold_distance(_CRASHING_DISTANCE, self._obstacle_displacements, self._chaser_displacements):
            self._chaser_displacements = self._create_initial_chaser_displacements(1, center_x=center_x, center_y=center_y, center_z=center_z)
        for _ in range(1, num_chasers):
            chaser_displacement = self._create_initial_chaser_displacements(1, center_x=center_x, center_y=center_y, center_z=center_z)
            while (self._has_any_within_threshold_distance(_CHASER_SELF_COLLISION_DISTANCE, self._chaser_displacements, chaser_displacement)
                    or self._has_any_within_threshold_distance(_CRASHING_DISTANCE, self._obstacle_displacements, self._chaser_displacements)):
                chaser_displacement = self._create_initial_chaser_displacements(1, center_x=center_x, center_y=center_y, center_z=center_z)
            self._chaser_displacements = tf.concat([self._chaser_displacements, chaser_displacement], axis=0)
        self._chaser_velocities = tf.zeros([num_chasers, 3])
        self._chaser_applied_accelerations = tf.zeros([num_chasers, 3])
        self._chaser_angular_displacements = self._create_initial_chaser_angular_displacements(num_chasers)
        self._chaser_angular_velocities = tf.zeros([num_chasers, 1])
        self._chaser_applied_angular_accelerations = tf.zeros([num_chasers, 1])
        num_targets = 1
        self._target_displacements = self._create_initial_target_displacements(1, center_x=center_x, center_y=center_y, center_z=center_z)
        while self._has_any_within_threshold_distance(_CATCHING_DISTANCE, self._chaser_displacements, self._target_displacements):
            self._target_displacements = self._create_initial_target_displacements(1, center_x=center_x, center_y=center_y, center_z=center_z)
        self._target_velocities = tf.zeros([num_targets, 3])
        self._target_applied_accelerations = tf.zeros([num_targets, 3])
        self._target_angular_displacements = self._create_initial_target_angular_displacements(num_targets)
        self._target_angular_velocities = tf.zeros([num_targets, 1])
        self._target_applied_angular_accelerations = tf.zeros([num_targets, 1])
        self._prev_chaser_displacements = self._chaser_displacements
        self._prev_target_displacements = self._target_displacements
        self._time = tf.constant(0.)
        self._has_updated = False
        self._has_collided = tf.constant(False)
        self._has_crashed = tf.constant(False)
        self._has_intercepted = tf.constant(False)

    def receive_action(self, agent_id: str, action: tf.Tensor):
        agent_index = bisect.bisect_left(sorted(self._chaser_ids), agent_id)
        agent_data = self._get_agent_data(agent_index)
        self._apply_action(agent_index, agent_data, action)

    def update(self):
        self._prev_chaser_displacements = self._chaser_displacements
        self._prev_target_displacements = self._target_displacements
        self._apply_target_actions()
        self._update_chasers()
        self._update_targets()
        self._time += _DELTA_T
        self._has_updated = True
        self._has_collided = self._has_any_within_threshold_distance(_CHASER_SELF_COLLISION_DISTANCE, self._chaser_displacements)
        self._has_crashed = self._has_any_within_threshold_distance(_CRASHING_DISTANCE, self._chaser_displacements, self._obstacle_displacements)
        self._has_intercepted = self._has_any_within_threshold_distance(_CATCHING_DISTANCE, self._chaser_displacements, self._target_displacements)

    def get_steps(self, group_id: str) -> Dict[str, TimeStep]:
        group_steps: Dict[str, TimeStep] = dict()
        for agent_id in self._group_chaser_ids[group_id]:
            group_steps[agent_id] = self._get_step(agent_id)
        return group_steps

    def _get_step(self, agent_id: str) -> TimeStep:
        agent_index = bisect.bisect_left(sorted(self._chaser_ids), agent_id)
        observation = self._get_observation(agent_index)
        step_type: tf.Tensor
        reward: tf.Tensor
        if self._has_collided:
            step_type = STEP_TYPE_LAST
            reward = tf.constant(_COLLISION_REWARD)
        elif self._has_crashed:
            step_type = STEP_TYPE_LAST
            reward = tf.constant(_CRASHING_REWARD)
        elif self._has_intercepted:
            step_type = STEP_TYPE_LAST
            reward = tf.constant(_CATCHING_REWARD)
        else:
            if self._has_updated:
                step_type = STEP_TYPE_MID
            else:
                step_type = STEP_TYPE_FIRST
            reward = self._calculate_non_terminal_reward()
        time_step = TimeStep(step_type, reward, observation)
        return time_step

    def generate_state_data_for_plotting(self) -> Dict[str, tf.Tensor]:
        state_data = dict()
        state_data[CHASER_DIRECTIONS] = find_direction(self._chaser_angular_displacements)
        state_data[CHASER_DISPLACEMENTS] = self._chaser_displacements
        state_data[HAS_COLLIDED] = self._has_collided
        state_data[HAS_CRASHED] = self._has_crashed
        state_data[HAS_INTERCEPTED] = self._has_intercepted
        state_data[OBSTACLE_DISPLACEMENTS] = self._obstacle_displacements
        state_data[TARGET_DIRECTIONS] = find_direction(self._target_angular_displacements)
        state_data[TARGET_DISPLACEMENTS] = self._target_displacements
        state_data[TIME] = self._time
        return state_data

    def get_observation_spec(self) -> Dict[str, tf.TensorSpec]:
        observation_spec: Dict[str, tf.TensorSpec] = {
            AGENT_ANGULAR_VELOCITIES: tf.TensorSpec([None, 1]),
            CHASER_RELATIVE_DISPLACEMENTS: tf.TensorSpec([None, 3]),
            CHASER_RELATIVE_VELOCITIES: tf.TensorSpec([None, 3]),
            CHASER_ANGULAR_VELOCITIES: tf.TensorSpec([None, 1]),
            OBSTACLE_RELATIVE_DISPLACEMENTS: tf.TensorSpec([None, 3]),
            TARGET_RELATIVE_DISPLACEMENTS: tf.TensorSpec([None, 3]),
            TARGET_RELATIVE_VELOCITIES: tf.TensorSpec([None, 3]),
            TARGET_ANGULAR_VELOCITIES: tf.TensorSpec([None, 1]),
        }
        return observation_spec

    def _create_initial_obstacle_displacements(
        self,
        num_obstacles: int,
        center_x: float = 0.0,
        center_y: float = 0.0,
        center_z: float = 0.0,
        bounds_width: float = _OBSTACLE_BOUNDS_WIDTH,
        bounds_height: float = _OBSTACLE_BOUNDS_HEIGHT,
        bounds_depth: float = _OBSTACLE_BOUNDS_DEPTH,
    ) -> tf.Tensor:
        return self._create_initial_displacements(
            num_obstacles,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            bounds_width=bounds_width,
            bounds_height=bounds_height,
            bounds_depth=bounds_depth,
        )

    def _create_initial_chaser_displacements(
        self,
        num_chasers: int,
        center_x: float = 0.0,
        center_y: float = 0.0,
        center_z: float = 0.0,
        bounds_width: float = _INITIAL_BOUNDS_WIDTH,
        bounds_height: float = _INITIAL_BOUNDS_HEIGHT,
        bounds_depth: float = _INITIAL_BOUNDS_DEPTH,
    ) -> tf.Tensor:
        return self._create_initial_displacements(
            num_chasers,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            bounds_width=bounds_width,
            bounds_height=bounds_height,
            bounds_depth=bounds_depth,
        )

    def _create_initial_target_displacements(
        self,
        num_targets: int,
        center_x: float = 0.0,
        center_y: float = 0.0,
        center_z: float = 0.0,
        bounds_width: float = _INITIAL_BOUNDS_WIDTH,
        bounds_height: float = _INITIAL_BOUNDS_HEIGHT,
        bounds_depth: float = _INITIAL_BOUNDS_DEPTH,
    ) -> tf.Tensor:
        return self._create_initial_displacements(
            num_targets,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            bounds_width=bounds_width,
            bounds_height=bounds_height,
            bounds_depth=bounds_depth,
        )

    def _create_initial_displacements(
        self,
        num_entities: int,
        center_x: float = 0.0,
        center_y: float = 0.0,
        center_z: float = 0.0,
        bounds_width: float = _INITIAL_BOUNDS_WIDTH,
        bounds_height: float = _INITIAL_BOUNDS_HEIGHT,
        bounds_depth: float = _INITIAL_BOUNDS_DEPTH,
    ) -> tf.Tensor:
        half_initial_bounds_width = bounds_width / 2.
        half_initial_bounds_height = bounds_height / 2.
        half_initial_bounds_depth = bounds_depth / 2.
        min_x = center_x - half_initial_bounds_width
        min_y = center_y - half_initial_bounds_height
        min_z = center_z - half_initial_bounds_depth
        max_x = center_x + half_initial_bounds_width
        max_y = center_y + half_initial_bounds_height
        max_z = center_z + half_initial_bounds_depth
        displacements = tf.random.uniform(
            [num_entities, 3],
            minval=tf.constant([[min_x, min_y, min_z]]),
            maxval=tf.constant([[max_x, max_y, max_z]]),
        )
        return displacements

    def _create_initial_chaser_angular_displacements(self, num_chasers: int) -> tf.Tensor:
        return self._create_initial_angular_displacements(num_chasers)

    def _create_initial_target_angular_displacements(self, num_targets: int) -> tf.Tensor:
        return self._create_initial_angular_displacements(num_targets)

    def _create_initial_angular_displacements(self, num_entities: int) -> tf.Tensor:
        angular_displacements = tf.random.uniform(
            [num_entities, 1],
            minval=-math.pi,
            maxval=math.pi,
        )
        return angular_displacements

    def _has_any_within_threshold_distance(self, threshold_distance: float, p_displacements: tf.Tensor, q_displacements: Optional[tf.Tensor] = None) -> tf.Tensor:
        relative_displacements = find_relative_quantities(p_displacements, q_displacements)
        relative_distances = tf.norm(relative_displacements, axis=-1)
        has_self_collisions = relative_distances <= threshold_distance
        has_existing_self_collision = tf.math.reduce_any(has_self_collisions)
        return has_existing_self_collision

    def _get_agent_data(self, agent_index: int) -> Dict[str, tf.Tensor]:
        agent_data = dict()
        agent_data[_CHASER_ANGULAR_DISPLACEMENT_KEY] = self._chaser_angular_displacements[agent_index:agent_index + 1]
        agent_data[_CHASER_ACCELERATION_NORM_KEY] = tf.gather(self._chaser_acceleration_norms, agent_index)
        agent_data[_CHASER_ANGULAR_ACCELERATION_NORM_KEY] = tf.gather(self._chaser_angular_acceleration_norms, agent_index)
        return agent_data

    def _apply_action(self, agent_index: int, agent_data: Dict[str, tf.Tensor], action: tf.Tensor):
        action_branches = {
            _ACTION_BRANCHES[REST]: lambda: self._rest(agent_index, agent_data),
            _ACTION_BRANCHES[FORWARD]: lambda: self._accelerate_forward(agent_index, agent_data),
            _ACTION_BRANCHES[BACKWARD]: lambda: self._accelerate_backward(agent_index, agent_data),
            _ACTION_BRANCHES[ANTI_CLOCKWISE]: lambda: self._accelerate_anti_clockwise(agent_index, agent_data),
            _ACTION_BRANCHES[CLOCKWISE]: lambda: self._accelerate_clockwise(agent_index, agent_data),
            _ACTION_BRANCHES[UP]: lambda: self._accelerate_up(agent_index, agent_data),
            _ACTION_BRANCHES[DOWN]: lambda: self._accelerate_down(agent_index, agent_data),
        }
        tf.switch_case(action, action_branches)

    def _rest(self, agent_index: int, agent_data: Dict[str, tf.Tensor]):
        chaser_angular_displacement = agent_data[_CHASER_ANGULAR_DISPLACEMENT_KEY]
        direction = find_direction(chaser_angular_displacement)
        indices = tf.constant([[agent_index]])
        self._chaser_applied_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_accelerations, indices, tf.zeros_like(direction))
        self._chaser_applied_angular_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_angular_accelerations, indices, tf.zeros_like(chaser_angular_displacement))

    def _accelerate_forward(self, agent_index: int, agent_data: Dict[str, tf.Tensor]):
        chaser_angular_displacement = agent_data[_CHASER_ANGULAR_DISPLACEMENT_KEY]
        chaser_acceleration_norm = agent_data[_CHASER_ACCELERATION_NORM_KEY]
        direction = find_direction(chaser_angular_displacement)
        indices = tf.constant([[agent_index]])
        self._chaser_applied_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_accelerations, indices, chaser_acceleration_norm * direction)
        self._chaser_applied_angular_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_angular_accelerations, indices, tf.zeros_like(chaser_angular_displacement))

    def _accelerate_backward(self, agent_index: int, agent_data: Dict[str, tf.Tensor]):
        chaser_angular_displacement = agent_data[_CHASER_ANGULAR_DISPLACEMENT_KEY]
        chaser_acceleration_norm = agent_data[_CHASER_ACCELERATION_NORM_KEY]
        direction = find_direction(chaser_angular_displacement)
        indices = tf.constant([[agent_index]])
        self._chaser_applied_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_accelerations, indices, -chaser_acceleration_norm * direction)
        self._chaser_applied_angular_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_angular_accelerations, indices, tf.zeros_like(chaser_angular_displacement))

    def _accelerate_anti_clockwise(self, agent_index: int, agent_data: Dict[str, tf.Tensor]):
        chaser_angular_displacement = agent_data[_CHASER_ANGULAR_DISPLACEMENT_KEY]
        chaser_angular_acceleration_norm = agent_data[_CHASER_ANGULAR_ACCELERATION_NORM_KEY]
        direction = find_direction(chaser_angular_displacement)
        indices = tf.constant([[agent_index]])
        self._chaser_applied_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_accelerations, indices, tf.zeros_like(direction))
        self._chaser_applied_angular_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_angular_accelerations, indices, chaser_angular_acceleration_norm * tf.ones_like(chaser_angular_displacement))

    def _accelerate_clockwise(self, agent_index: int, agent_data: Dict[str, tf.Tensor]):
        chaser_angular_displacement = agent_data[_CHASER_ANGULAR_DISPLACEMENT_KEY]
        chaser_angular_acceleration_norm = agent_data[_CHASER_ANGULAR_ACCELERATION_NORM_KEY]
        direction = find_direction(chaser_angular_displacement)
        indices = tf.constant([[agent_index]])
        self._chaser_applied_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_accelerations, indices, tf.zeros_like(direction))
        self._chaser_applied_angular_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_angular_accelerations, indices, -chaser_angular_acceleration_norm * tf.ones_like(chaser_angular_displacement))

    def _accelerate_up(self, agent_index: int, agent_data: Dict[str, tf.Tensor]):
        chaser_angular_displacement = agent_data[_CHASER_ANGULAR_DISPLACEMENT_KEY]
        chaser_acceleration_norm = agent_data[_CHASER_ACCELERATION_NORM_KEY]
        chaser_acceleration = tf.constant([[0., 0., 1.]]) * chaser_acceleration_norm
        indices = tf.constant([[agent_index]])
        self._chaser_applied_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_accelerations, indices, chaser_acceleration)
        self._chaser_applied_angular_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_angular_accelerations, indices, tf.zeros_like(chaser_angular_displacement))

    def _accelerate_down(self, agent_index: int, agent_data: Dict[str, tf.Tensor]):
        chaser_angular_displacement = agent_data[_CHASER_ANGULAR_DISPLACEMENT_KEY]
        chaser_acceleration_norm = agent_data[_CHASER_ACCELERATION_NORM_KEY]
        chaser_acceleration = - tf.constant([[0., 0., 1.]]) * chaser_acceleration_norm
        indices = tf.constant([[agent_index]])
        self._chaser_applied_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_accelerations, indices, chaser_acceleration)
        self._chaser_applied_angular_accelerations = tf.tensor_scatter_nd_update(self._chaser_applied_angular_accelerations, indices, tf.zeros_like(chaser_angular_displacement))

    def _apply_target_actions(self):
        target_chaser_displacement_cartesian_product = find_cartesian_product(self._target_displacements, -self._chaser_displacements)
        target_relative_displacement_cartesian_product = tf.reduce_sum(target_chaser_displacement_cartesian_product, axis=2)
        target_relative_displacement_norm_cartesian_product = tf.norm(target_relative_displacement_cartesian_product, axis=-1)
        target_min_relative_displacement_norm_indices = tf.cast(tf.math.argmin(target_relative_displacement_norm_cartesian_product, axis=-1), tf.dtypes.int32)
        target_indices = tf.range(tf.shape(target_min_relative_displacement_norm_indices)[0])
        indices = tf.stack([target_indices, target_min_relative_displacement_norm_indices], axis=-1)
        target_min_relative_displacements = tf.gather_nd(target_relative_displacement_cartesian_product, indices)
        target_directions = find_direction(self._target_angular_displacements)
        dot_products = tf.reduce_sum(target_min_relative_displacements * target_directions, axis=-1, keepdims=True)
        num_targets = tf.shape(self._target_displacements)[0]
        target_applied_z_acceleration_mask = tf.concat(
            [
                tf.zeros([num_targets, 2]),
                tf.ones([num_targets, 1])
            ],
            axis=-1,
        )
        target_applied_z_accelerations = target_applied_z_acceleration_mask * tf.sign(target_min_relative_displacements)
        self._target_applied_accelerations = (tf.sign(dot_products) * target_directions + target_applied_z_accelerations) * self._target_acceleration_norms
        sin_relative_angular_displacements = tf.clip_by_value(tf.linalg.cross(target_directions, target_min_relative_displacements)[:, -1:] / tf.norm(target_min_relative_displacements, axis=-1), -1.0, 1.0)
        relative_angular_displacements = tf.math.asin(sin_relative_angular_displacements)
        self._target_applied_angular_accelerations = relative_angular_displacements / math.pi * self._target_angular_acceleration_norms

    def _update_chasers(self):
        quantities = {
            _MASSES: self._chaser_masses,
            _DISPLACEMENTS: self._chaser_displacements,
            _VELOCITIES: self._chaser_velocities,
            _APPLIED_ACCELERATIONS: self._chaser_applied_accelerations,
            _MOMENTS_OF_INERTIA: self._chaser_moments_of_inertia,
            _ANGULAR_DISPLACEMENTS: self._chaser_angular_displacements,
            _ANGULAR_VELOCITIES: self._chaser_angular_velocities,
            _APPLIED_ANGULAR_ACCELERATIONS: self._chaser_applied_angular_accelerations,
        }
        updated_quantities = self._integrate(quantities)
        self._chaser_displacements = updated_quantities[_DISPLACEMENTS]
        self._chaser_velocities = updated_quantities[_VELOCITIES]
        self._chaser_angular_displacements = updated_quantities[_ANGULAR_DISPLACEMENTS]
        self._chaser_angular_velocities = updated_quantities[_ANGULAR_VELOCITIES]

    def _update_targets(self):
        quantities = {
            _MASSES: self._target_masses,
            _DISPLACEMENTS: self._target_displacements,
            _VELOCITIES: self._target_velocities,
            _APPLIED_ACCELERATIONS: self._target_applied_accelerations,
            _MOMENTS_OF_INERTIA: self._target_moments_of_inertia,
            _ANGULAR_DISPLACEMENTS: self._target_angular_displacements,
            _ANGULAR_VELOCITIES: self._target_angular_velocities,
            _APPLIED_ANGULAR_ACCELERATIONS: self._target_applied_angular_accelerations,
        }
        updated_quantities = self._integrate(quantities)
        self._target_displacements = updated_quantities[_DISPLACEMENTS]
        self._target_velocities = updated_quantities[_VELOCITIES]
        self._target_angular_displacements = updated_quantities[_ANGULAR_DISPLACEMENTS]
        self._target_angular_velocities = updated_quantities[_ANGULAR_VELOCITIES]

    @tf.function
    def _integrate(self, quantities: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        non_integrated_quantities = {
            _MASSES,
            _APPLIED_ACCELERATIONS,
            _MOMENTS_OF_INERTIA,
            _APPLIED_ANGULAR_ACCELERATIONS,
        }
        k1 = self._differentiate(quantities)
        k2dt = dict()
        for key, value in quantities.items():
            if key in non_integrated_quantities:
                k2dt[key] = value
                continue
            k2dt[key] = quantities[key] + 0.5 * _DELTA_T * k1['d/dt_{key}'.format(key=key)]
        k2 = self._differentiate(k2dt)
        k3dt = dict()
        for key, value in quantities.items():
            if key in non_integrated_quantities:
                k3dt[key] = value
                continue
            k3dt[key] = quantities[key] + 0.5 * _DELTA_T * k2['d/dt_{key}'.format(key=key)]
        k3 = self._differentiate(k3dt)
        k4dt = dict()
        for key, value in quantities.items():
            if key in non_integrated_quantities:
                k4dt[key] = value
                continue
            k4dt[key] = quantities[key] + _DELTA_T * k3['d/dt_{key}'.format(key=key)]
        k4 = self._differentiate(k4dt)
        updated_quantities = dict()
        for key, value in quantities.items():
            if key in non_integrated_quantities:
                updated_quantities[key] = value
                continue
            derivative_key = 'd/dt_{key}'.format(key=key)
            updated_quantities[key] = 0.0
            updated_quantities[key] += k1[derivative_key]
            updated_quantities[key] += 2.0 * k2[derivative_key]
            updated_quantities[key] += 2.0 * k3[derivative_key]
            updated_quantities[key] += k4[derivative_key]
            updated_quantities[key] *= _DELTA_T / 6.0
            updated_quantities[key] += quantities[key]
        return updated_quantities

    def _differentiate(self, quantities: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        accelerations = -_DRAG_COEFFICIENT / quantities[_MASSES] * tf.norm(quantities[_VELOCITIES], axis=-1, keepdims=True) * quantities[_VELOCITIES] + quantities[_APPLIED_ACCELERATIONS]
        angular_accelerations = -_ANGULAR_DRAG_COEFFICIENT / quantities[_MOMENTS_OF_INERTIA] * tf.norm(quantities[_ANGULAR_VELOCITIES], axis=-1, keepdims=True) * quantities[_ANGULAR_VELOCITIES] + quantities[_APPLIED_ANGULAR_ACCELERATIONS]
        differentiated_quantities = {
            'd/dt_{key}'.format(key=_DISPLACEMENTS): quantities[_VELOCITIES],
            'd/dt_{key}'.format(key=_VELOCITIES): accelerations,
            'd/dt_{key}'.format(key=_ANGULAR_DISPLACEMENTS): quantities[_ANGULAR_VELOCITIES],
            'd/dt_{key}'.format(key=_ANGULAR_VELOCITIES): angular_accelerations,
        }
        return differentiated_quantities

    def _get_observation(self, agent_index: int) -> Dict[str, tf.Tensor]:
        agent_displacements = self._chaser_displacements[agent_index:agent_index + 1]
        agent_velocities = self._chaser_velocities[agent_index:agent_index + 1]
        agent_angular_displacement = self._chaser_angular_displacements[agent_index]
        agent_angular_velocities = self._chaser_angular_velocities[agent_index:agent_index + 1]
        agent_direction = find_direction(agent_angular_displacement)
        change_of_basis_matrix = self._find_change_of_basis_matrix(agent_direction)
        chaser_displacements = tf.concat(
            [
                self._chaser_displacements[:agent_index],
                self._chaser_displacements[agent_index + 1:],
            ],
            axis=0,
        )
        chaser_velocities = tf.concat(
            [
                self._chaser_velocities[:agent_index],
                self._chaser_velocities[agent_index + 1:],
            ],
            axis=0,
        )
        chaser_angular_velocities = tf.concat(
            [
                self._chaser_angular_velocities[:agent_index],
                self._chaser_angular_velocities[agent_index + 1:],
            ],
            axis=0,
        )
        chaser_relative_displacements = chaser_displacements - agent_displacements
        chaser_relative_velocities = chaser_velocities - agent_velocities
        obstacle_relative_displacements = self._obstacle_displacements - agent_displacements
        target_relative_displacements = self._target_displacements - agent_displacements
        target_relative_velocities = self._target_velocities - agent_velocities
        transformed_chaser_relative_displacements = self._change_basis(change_of_basis_matrix, chaser_relative_displacements)
        transformed_chaser_relative_velocities = self._change_basis(change_of_basis_matrix, chaser_relative_velocities)
        transformed_obstacle_relative_displacements = self._change_basis(change_of_basis_matrix, obstacle_relative_displacements)
        transformed_target_relative_displacements = self._change_basis(change_of_basis_matrix, target_relative_displacements)
        transformed_target_relative_velocities = self._change_basis(change_of_basis_matrix, target_relative_velocities)
        observation: Dict[str, tf.Tensor] = {
            AGENT_ANGULAR_VELOCITIES: agent_angular_velocities,
            CHASER_RELATIVE_DISPLACEMENTS: transformed_chaser_relative_displacements,
            CHASER_RELATIVE_VELOCITIES: transformed_chaser_relative_velocities,
            CHASER_ANGULAR_VELOCITIES: chaser_angular_velocities,
            OBSTACLE_RELATIVE_DISPLACEMENTS: transformed_obstacle_relative_displacements,
            TARGET_RELATIVE_DISPLACEMENTS: transformed_target_relative_displacements,
            TARGET_RELATIVE_VELOCITIES: transformed_target_relative_velocities,
            TARGET_ANGULAR_VELOCITIES: self._target_angular_velocities,
        }
        return observation

    def _find_change_of_basis_matrix(self, direction: tf.Tensor) -> tf.Tensor:
        positive_z = tf.constant([0.0, 0.0, 1.0])
        perpendicular_basis = tf.linalg.cross(positive_z, direction)
        normalised_perpendicular_basis = perpendicular_basis / tf.norm(perpendicular_basis, axis=-1)
        change_of_basis_matrix = tf.stack([
            direction,
            normalised_perpendicular_basis,
            positive_z,
        ], axis=-2)
        return change_of_basis_matrix

    def _change_basis(self, change_of_basis_matrix: tf.Tensor, vectors: tf.Tensor):
        transformed_vectors = tf.tensordot(vectors, change_of_basis_matrix, [[1], [1]])
        return transformed_vectors

    def _calculate_non_terminal_reward(self):
        reward = 0.0
        time_reward = self._calculate_time_reward()
        reward += time_reward
        distance_reward = self._calculate_distance_reward()
        reward += distance_reward
        return reward

    def _calculate_time_reward(self) -> tf.Tensor:
        reward = -_DELTA_T
        return reward

    def _calculate_distance_reward(self) -> tf.Tensor:
        prev_relative_displacements = find_relative_quantities(self._prev_chaser_displacements, self._prev_target_displacements)
        prev_relative_distances = tf.norm(prev_relative_displacements, axis=-1)
        prev_relative_distance_sum = tf.math.reduce_sum(prev_relative_distances)
        curr_relative_displacements = find_relative_quantities(self._chaser_displacements, self._target_displacements)
        curr_relative_distances = tf.norm(curr_relative_displacements, axis=-1)
        curr_relative_distance_sum = tf.math.reduce_sum(curr_relative_distances)
        reward = prev_relative_distance_sum - curr_relative_distance_sum
        return reward
