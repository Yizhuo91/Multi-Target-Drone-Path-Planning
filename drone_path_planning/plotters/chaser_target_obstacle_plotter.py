import bisect
import csv
import logging
import os
import pickle
import random
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple

from matplotlib import animation
from matplotlib import artist
from matplotlib import axes
from matplotlib import figure
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from drone_path_planning.plotters.plotter import Plotter
from drone_path_planning.utilities.constants import CHASER_DIRECTIONS
from drone_path_planning.utilities.constants import CHASER_DISPLACEMENTS
from drone_path_planning.utilities.constants import HAS_COLLIDED
from drone_path_planning.utilities.constants import HAS_CRASHED
from drone_path_planning.utilities.constants import HAS_INTERCEPTED
from drone_path_planning.utilities.constants import OBSTACLE_DISPLACEMENTS
from drone_path_planning.utilities.constants import TARGET_DIRECTIONS
from drone_path_planning.utilities.constants import TARGET_DISPLACEMENTS
from drone_path_planning.utilities.constants import TIME
from drone_path_planning.utilities.functions import find_relative_quantities


_EnvironmentState = Dict[str, tf.Tensor]
_Trajectory = List[_EnvironmentState]
_TrajectoryList = List[_Trajectory]


_PLOT_DATA_DIRS: str = 'plot_data_dirs'
_ANIMATION: str = 'animation'
_MIN_WIDTH: str = 'min_width'
_SAVE_FILENAME: str = 'save_filename'
_FIGSIZE: str = 'figsize'
_MAX_NUM_TRAJECTORIES: str = 'max_num_trajectories'
_SHOULD_RANDOMIZE: str = 'should_randomize'
_ARROW_LENGTH: str = 'arrow_length'
_MS_PER_FRAME: str = 'ms_per_frame'
_REQUIRED_TIME: str = 'required_time'
_DATASETS: str = 'datasets'
_MARKER_SIZE: str = 'marker_size'
_SCATTER_ALPHA: str = 'scatter_alpha'
_LINE_ALPHA: str = 'line_alpha'
_CHASER_TERMINAL_SPEED: str = 'chaser_terminal_speed'
_TARGET_TERMINAL_SPEED: str = 'target_terminal_speed'
_CATCHING_DISTANCE: str = 'catching_distance'
_LABEL: str = 'label'
_BASELINE_LABEL: str = 'baseline_label'
_FIT_LABEL_TEMPLATE: str = 'fit_label_template'
_X_LABEL: str = 'x_label'
_Y_LABEL: str = 'y_label'
_TITLE: str = 'title'
_STATISTICS: str = 'statistics'


_TRAJECTORIES: str = 'trajectories'
_CENTER: str = 'center'
_HALF_WIDTH: str = 'half_width'
_MIN_Z: str = 'min_z'
_ANIMATION_TITLE_TEMPLATE: str = 'Trajectory {run:d} Step {step:d} Time {time:.1f} ({outcome:s})'
_COLLISION_OUTCOME: str = 'collided'
_CRASH_OUTCOME: str = 'crashed'
_INTERCEPTION_OUTCOME: str = 'intercepted'
_TIME_OUT_OUTCOME: str = 'timed out'
_MIN_CHASER_TARGET_DISTANCES: str = 'min_chaser_target_distances'
_REQUIRED_TIMES: str = 'required_times'
_FIT_REQUIRED_TIME_PARAMS: str = 'fit_require_time_params'
_FIT_REQUIRED_TIMES: str = 'fit_required_times'
_FIT_REQUIRED_TIME_R2: str = 'fit_required_time_r2'
_NUM_TRAJECTORIES: str = 'num_trajectories'
_NUM_COLLIDED: str = 'num_collided'
_NUM_CRASHED: str = 'num_crashed'
_NUM_INTERCEPTED: str = 'num_intercepted'
_NUM_TIMED_OUT: str = 'num_timed_out'
_DATASET: str = 'dataset'
_W_1: str = 'w_1'
_W_0: str = 'w_0'
_R2: str = 'r2'
_BASELINE: str = 'baseline'
_HAS_INTERCEPTS: str = 'has_intercepts'


class ChaserTargetObstaclePlotter(Plotter):
    def __init__(
        self,
        plot_data_config: Any,
    ):
        self._logger = logging.getLogger(__name__)
        self._logger.info('Initializing plotter.')
        self._animation_plot_configs = plot_data_config[_ANIMATION]
        self._required_time_plot_configs = plot_data_config[_REQUIRED_TIME]
        self._statistics_configs = plot_data_config[_STATISTICS]
        self._plot_data_dirs = plot_data_config[_PLOT_DATA_DIRS]
        self._scenario_runs: Dict[str, Set[str]] = dict()
        self._raw_plot_datasets: Dict[str, Dict[str, _TrajectoryList]] = dict()
        self._plot_datasets: Dict[str, Dict[str, Dict[str, Any]]] = dict()
        self._logger.info('Plotter initialized.')

    def load_data(self):
        self._logger.info('Loading data.')
        for scenario, runs in self._plot_data_dirs.items():
            self._scenario_runs[scenario] = set(runs)
            self._raw_plot_datasets[scenario] = dict()
            for run, plot_data_dir in runs.items():
                plot_data_filepath = os.path.join(plot_data_dir, 'plot_data.pkl')
                with open(plot_data_filepath, 'rb') as fp:
                    self._raw_plot_datasets[scenario][run] = pickle.load(fp)
        self._logger.info('Data loaded.')

    def process_data(self):
        self._logger.info('Processing data.')
        for scenario, runs in self._scenario_runs.items():
            self._plot_datasets[scenario] = dict()
            for run in runs:
                min_width = self._animation_plot_configs[scenario][run][_MIN_WIDTH]
                process_trajectory = lambda trajectory: self._process_trajectory(trajectory, min_width)
                raw_plot_dataset = self._raw_plot_datasets[scenario][run]
                trajectories = list(map(process_trajectory, raw_plot_dataset))
                self._plot_datasets[scenario][run] = dict()
                self._plot_datasets[scenario][run][_TRAJECTORIES] = trajectories
                num_trajectories = len(trajectories)
                num_collided = 0
                num_crashed = 0
                num_intercepted = 0
                num_timed_out = 0
                for environment_states in trajectories:
                    last_environment_state = environment_states[-1]
                    if last_environment_state[HAS_COLLIDED]:
                        num_collided += 1
                    elif last_environment_state[HAS_CRASHED]:
                        num_crashed += 1
                    elif last_environment_state[HAS_INTERCEPTED]:
                        num_intercepted += 1
                    else:
                        num_timed_out += 1
                self._plot_datasets[scenario][run][_NUM_TRAJECTORIES] = num_trajectories
                self._plot_datasets[scenario][run][_NUM_COLLIDED] = num_collided
                self._plot_datasets[scenario][run][_NUM_CRASHED] = num_crashed
                self._plot_datasets[scenario][run][_NUM_INTERCEPTED] = num_intercepted
                self._plot_datasets[scenario][run][_NUM_TIMED_OUT] = num_timed_out
                if num_intercepted < 1:
                    self._plot_datasets[scenario][run][_HAS_INTERCEPTS] = False
                    continue
                self._plot_datasets[scenario][run][_HAS_INTERCEPTS] = True
                min_chaser_target_distance_arr = self._find_min_chaser_target_distances(trajectories)
                required_time_arr = self._find_required_times(trajectories)
                sorted_min_chaser_target_distance_arr = np.sort(min_chaser_target_distance_arr)
                sorted_required_time_arr = required_time_arr[np.argsort(min_chaser_target_distance_arr)]
                fit_required_time_param_arr = self._find_linear_least_squares_parameters(sorted_min_chaser_target_distance_arr, sorted_required_time_arr)
                fit_required_time_arr = self._find_linear_least_squares_fit(sorted_min_chaser_target_distance_arr, fit_required_time_param_arr)
                fit_required_time_r2 = self._find_coefficient_of_determination(sorted_required_time_arr, fit_required_time_arr)
                self._plot_datasets[scenario][run][_MIN_CHASER_TARGET_DISTANCES] = sorted_min_chaser_target_distance_arr
                self._plot_datasets[scenario][run][_REQUIRED_TIMES] = sorted_required_time_arr
                self._plot_datasets[scenario][run][_FIT_REQUIRED_TIME_PARAMS] = fit_required_time_param_arr
                self._plot_datasets[scenario][run][_FIT_REQUIRED_TIMES] = fit_required_time_arr
                self._plot_datasets[scenario][run][_FIT_REQUIRED_TIME_R2] = fit_required_time_r2
        self._logger.info('Data processed.')

    def _process_trajectory(self, trajectory: _Trajectory, min_width: float) -> _Trajectory:
        process_environment_state = lambda environment_state: self._process_environment_state(environment_state, min_width)
        processed_trajectory = list(map(process_environment_state, trajectory))
        return processed_trajectory

    def _process_environment_state(self, environment_state: _EnvironmentState, min_width: float) -> _EnvironmentState:
        processed_environment_state = {**environment_state}
        chaser_displacements = environment_state[CHASER_DISPLACEMENTS]
        target_displacements = environment_state[TARGET_DISPLACEMENTS]
        all_displacements = tf.concat([chaser_displacements, target_displacements], axis=0)
        center = tf.math.reduce_mean(all_displacements, axis=0)
        maximum_distance_from_center = tf.math.reduce_max(tf.linalg.norm(all_displacements - center, axis=-1))
        half_width = tf.math.maximum(maximum_distance_from_center, min_width / 2)
        min_z = tf.math.reduce_min(all_displacements[:, -1])
        processed_environment_state[_CENTER] = center
        processed_environment_state[_HALF_WIDTH] = half_width
        processed_environment_state[_MIN_Z] = min_z
        return processed_environment_state

    def _find_min_chaser_target_distances(self, trajectories: _TrajectoryList) -> np.ndarray:
        min_chaser_target_distances = []
        for trajectory in trajectories:
            last_environment_state = trajectory[-1]
            if (last_environment_state[HAS_COLLIDED]
                or last_environment_state[HAS_CRASHED]
                or not last_environment_state[HAS_INTERCEPTED]):
                continue
            min_chaser_target_distances.extend(map(self._find_min_chaser_target_distance, trajectory))
        min_chaser_target_distance_arr = np.array(min_chaser_target_distances)
        return min_chaser_target_distance_arr

    def _find_min_chaser_target_distance(self, environment_state: _EnvironmentState) -> tf.Tensor:
        chaser_displacements = environment_state[CHASER_DISPLACEMENTS]
        target_displacements = environment_state[TARGET_DISPLACEMENTS]
        relative_displacements = find_relative_quantities(chaser_displacements, target_displacements)
        relative_distances = tf.norm(relative_displacements, axis=-1)
        min_chaser_target_distance = tf.math.reduce_min(relative_distances)
        return min_chaser_target_distance

    def _find_required_times(self, trajectories: _TrajectoryList) -> np.ndarray:
        required_times = []
        for trajectory in trajectories:
            last_environment_state = trajectory[-1]
            if (last_environment_state[HAS_COLLIDED]
                or last_environment_state[HAS_CRASHED]
                or not last_environment_state[HAS_INTERCEPTED]):
                continue
            last_time = last_environment_state[TIME]
            find_required_time = lambda environment_state: self._find_required_time(environment_state, last_time)
            required_times.extend(map(find_required_time, trajectory))
        required_time_arr = np.array(required_times)
        return required_time_arr

    def _find_required_time(self, environment_state: _EnvironmentState, last_time: tf.Tensor) -> tf.Tensor:
        required_time = last_time - environment_state[TIME]
        return required_time

    def _find_linear_least_squares_parameters(self, x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        x_mat = np.stack([np.ones(len(x_arr)), x_arr], axis=-1)
        xT_x_mat = x_mat.T @ x_mat
        xT_y_arr = x_mat.T @ y_arr
        w_arr = np.linalg.pinv(xT_x_mat) @ xT_y_arr
        return w_arr

    def _find_linear_least_squares_fit(self, x_arr: np.ndarray, w_arr: np.ndarray) -> np.ndarray:
        x_mat = np.stack([np.ones(len(x_arr)), x_arr], axis=-1)
        hat_y_arr = x_mat @ w_arr
        return hat_y_arr

    def _find_coefficient_of_determination(self, y_arr: np.ndarray, hat_y_arr: np.ndarray) -> float:
        squares_residual_sum = np.sum((y_arr - hat_y_arr) ** 2.)
        mean_y = np.mean(y_arr)
        squares_total_sum = np.sum((y_arr - mean_y) ** 2.)
        r2 = 1. - squares_residual_sum / squares_total_sum
        return r2

    def plot(self, plots_dir: str):
        self._plot_animations(plots_dir)
        self._plot_required_times(plots_dir)
        self._save_statistics_files(plots_dir)

    def _plot_animations(self, plots_dir: str):
        for scenario, runs in self._animation_plot_configs.items():
            for run, plot_config in runs.items():
                save_filename = plot_config[_SAVE_FILENAME]
                save_filepath = os.path.join(plots_dir, save_filename)
                trajectories = self._plot_datasets[scenario][run][_TRAJECTORIES]
                max_num_trajectories = min(len(trajectories), plot_config[_MAX_NUM_TRAJECTORIES])
                animated_trajectories: _TrajectoryList
                if plot_config[_SHOULD_RANDOMIZE]:
                    animated_trajectories = random.sample(trajectories, max_num_trajectories)
                else:
                    animated_trajectories = trajectories[:max_num_trajectories]
                figsize = plot_config[_FIGSIZE]
                arrow_length = plot_config[_ARROW_LENGTH]
                ms_per_frame = plot_config[_MS_PER_FRAME]
                self._logger.info('Plotting animation for scenario %s run %s.', scenario, run)
                self._plot_animation(
                    animated_trajectories,
                    save_filepath,
                    figsize,
                    arrow_length,
                    ms_per_frame,
                )
                self._logger.info('Animation plotted for scenario %s run %s.', scenario, run)

    def _plot_animation(
        self,
        trajectories: _TrajectoryList,
        save_filepath: str,
        figsize: Tuple[float, float],
        arrow_length: float,
        ms_per_frame: int,
    ):
        num_trajectories = len(trajectories)
        if num_trajectories < 1:
            return
        trajectory_lengths = [len(trajectory) for trajectory in trajectories]
        trajectory_length_prefix_sums = [trajectory_lengths[0]]
        for i in range(1, num_trajectories):
            trajectory_length_prefix_sums.append(trajectory_length_prefix_sums[i - 1] + trajectory_lengths[i])
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        trajectory_outcomes = self._determine_trajectory_outcomes(trajectories)
        update = self._create_animation_frame_updater(
            fig,
            ax,
            trajectory_length_prefix_sums,
            trajectories,
            trajectory_outcomes,
            arrow_length,
        )
        num_frames = trajectory_length_prefix_sums[-1]
        func_animation = animation.FuncAnimation(fig, update, frames=num_frames, interval=ms_per_frame)
        func_animation.save(save_filepath)

    def _determine_trajectory_outcomes(self, trajectories: _TrajectoryList):
        outcomes = []
        for trajectory in trajectories:
            last_environment_state = trajectory[-1]
            outcome: str
            if last_environment_state[HAS_COLLIDED]:
                outcome = _COLLISION_OUTCOME
            elif last_environment_state[HAS_CRASHED]:
                outcome = _CRASH_OUTCOME
            elif last_environment_state[HAS_INTERCEPTED]:
                outcome = _INTERCEPTION_OUTCOME
            else:
                outcome = _TIME_OUT_OUTCOME
            outcomes.append(outcome)
        return outcomes

    def _create_animation_frame_updater(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        trajectory_length_prefix_sums: List[int],
        trajectories: List[List[Dict[str, tf.Tensor]]],
        trajectory_outcomes: List[str],
        arrow_length: float,
    ) -> Callable[[int], Iterable[artist.Artist]]:
        def update(frame_index: int) -> Iterable[artist.Artist]:
            trajectory_index = bisect.bisect_right(trajectory_length_prefix_sums, frame_index)
            trajectory_length_prefix_sum = 0 if trajectory_index < 1 else trajectory_length_prefix_sums[trajectory_index - 1]
            environment_states = trajectories[trajectory_index]
            environment_state_index = frame_index - trajectory_length_prefix_sum
            environment_state = environment_states[environment_state_index]
            ax.clear()
            chaser_displacements = environment_state[CHASER_DISPLACEMENTS]
            chaser_directions = environment_state[CHASER_DIRECTIONS]
            obstacle_displacements = environment_state[OBSTACLE_DISPLACEMENTS]
            target_displacements = environment_state[TARGET_DISPLACEMENTS]
            target_directions = environment_state[TARGET_DIRECTIONS]
            center = environment_state[_CENTER]
            half_width = environment_state[_HALF_WIDTH]
            min_z = environment_state[_MIN_Z]
            ax.set_xlim3d(left=(center[0] - half_width), right=(center[0] + half_width))
            ax.set_ylim3d(bottom=(center[1] - half_width), top=(center[1] + half_width))
            ax.set_zlim3d(bottom=(min_z), top=(min_z + 2 * half_width))
            title_format_kwargs = dict(
                run=trajectory_index,
                step=environment_state_index,
                time=environment_state[TIME],
                outcome=trajectory_outcomes[trajectory_index],
            )
            title = _ANIMATION_TITLE_TEMPLATE.format(**title_format_kwargs)
            ax.set_title(title)
            chaser_quiver = ax.quiver(
                chaser_displacements[:, 0],
                chaser_displacements[:, 1],
                chaser_displacements[:, 2],
                chaser_directions[:, 0],
                chaser_directions[:, 1],
                chaser_directions[:, 2],
                length=arrow_length,
                normalize=True,
                colors=[(0.0, 0.0, 1.0, 0.9)],
            )
            target_quiver = ax.quiver(
                target_displacements[:, 0],
                target_displacements[:, 1],
                target_displacements[:, 2],
                target_directions[:, 0],
                target_directions[:, 1],
                target_directions[:, 2],
                length=arrow_length,
                normalize=True,
                colors=[(1.0, 0.0, 0.0, 0.9)],
            )
            ax.scatter(
                obstacle_displacements[:, 0],
                obstacle_displacements[:, 1],
                obstacle_displacements[:, 2],
            )
            return fig,
        return update

    def _plot_required_times(self, plots_dir: str):
        for config in self._required_time_plot_configs:
            save_filename = config[_SAVE_FILENAME]
            save_filepath = os.path.join(plots_dir, save_filename)
            self._logger.info('Plotting required time for plot %s.', save_filename)
            self._plot_required_time(config, save_filepath)
            self._logger.info('Required time plotted for plot %s.', save_filename)

    def _plot_required_time(self, config: Any, save_filepath: str):
        chaser_terminal_speed = config[_CHASER_TERMINAL_SPEED]
        target_terminal_speed = config[_TARGET_TERMINAL_SPEED]
        catching_distance = config[_CATCHING_DISTANCE]
        max_distance = catching_distance
        for scenario, runs in config[_DATASETS].items():
            for run in runs:
                plot_dataset = self._plot_datasets[scenario][run]
                if not plot_dataset[_HAS_INTERCEPTS]:
                    continue
                distances = plot_dataset[_MIN_CHASER_TARGET_DISTANCES]
                scenario_run_max_distance = distances[-1]
                max_distance = max(max_distance, scenario_run_max_distance)
        max_baseline_required_time = (max_distance - catching_distance) / (chaser_terminal_speed - target_terminal_speed)
        baseline_distances = [catching_distance, max_distance]
        baseline_required_times = [0., max_baseline_required_time]
        figsize = config[_FIGSIZE]
        baseline_label = config[_BASELINE_LABEL]
        fit_label_template = config[_FIT_LABEL_TEMPLATE]
        marker_size = config[_MARKER_SIZE]
        scatter_alpha = config[_SCATTER_ALPHA]
        line_alpha = config[_LINE_ALPHA]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        for scenario, runs in config[_DATASETS].items():
            for run, run_config in runs.items():
                plot_dataset = self._plot_datasets[scenario][run]
                if not plot_dataset[_HAS_INTERCEPTS]:
                    continue
                label = run_config[_LABEL]
                fit_label = fit_label_template.format(label=label, r2=plot_dataset[_FIT_REQUIRED_TIME_R2])
                min_chaser_target_distances = plot_dataset[_MIN_CHASER_TARGET_DISTANCES]
                required_times = plot_dataset[_REQUIRED_TIMES]
                fit_required_times = plot_dataset[_FIT_REQUIRED_TIMES]
                ax.scatter(min_chaser_target_distances, required_times, label=label, alpha=scatter_alpha, s=marker_size, marker='2')
                ax.plot(min_chaser_target_distances, fit_required_times, label=fit_label, alpha=line_alpha)
        ax.plot(baseline_distances, baseline_required_times, label=baseline_label, alpha=line_alpha)
        ax.set_xlabel(config[_X_LABEL])
        ax.set_ylabel(config[_Y_LABEL])
        ax.set_title(config[_TITLE])
        ax.legend()
        fig.savefig(save_filepath)

    def _save_statistics_files(self, plots_dir: str):
        for config in self._statistics_configs:
            save_filename = config[_SAVE_FILENAME]
            save_filepath = os.path.join(plots_dir, save_filename)
            self._logger.info('Saving statistics for %s.', save_filename)
            self._save_statistics_file(config, save_filepath)
            self._logger.info('Statistics saved for %s.', save_filename)

    def _save_statistics_file(self, config: Any, save_filepath: str):
        chaser_terminal_speed = config[_CHASER_TERMINAL_SPEED]
        target_terminal_speed = config[_TARGET_TERMINAL_SPEED]
        catching_distance = config[_CATCHING_DISTANCE]
        baseline_w1 = 1. / (chaser_terminal_speed - target_terminal_speed)
        baseline_w0 = - catching_distance * baseline_w1
        with open(save_filepath, 'w', newline='') as csv_file:
            field_names = [
                _DATASET,
                _NUM_TRAJECTORIES,
                _NUM_COLLIDED,
                _NUM_CRASHED,
                _NUM_INTERCEPTED,
                _NUM_TIMED_OUT,
                _HAS_INTERCEPTS,
                _R2,
                _W_0,
                _W_1,
            ]
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            for scenario, runs in config[_DATASETS].items():
                for run, run_config in runs.items():
                    plot_dataset = self._plot_datasets[scenario][run]
                    label = run_config[_LABEL]
                    params: np.ndarray
                    r2: float
                    if plot_dataset[_HAS_INTERCEPTS]:
                        params = plot_dataset[_FIT_REQUIRED_TIME_PARAMS]
                        r2 = plot_dataset[_FIT_REQUIRED_TIME_R2]
                    else:
                        params = np.array([0., 0.])
                        r2 = 0.
                    num_trajectories = plot_dataset[_NUM_TRAJECTORIES]
                    num_collided = plot_dataset[_NUM_COLLIDED]
                    num_crashed = plot_dataset[_NUM_CRASHED]
                    num_intercepted = plot_dataset[_NUM_INTERCEPTED]
                    num_timed_out = plot_dataset[_NUM_TIMED_OUT]
                    row_dict = {
                        _DATASET: label,
                        _NUM_TRAJECTORIES: num_trajectories,
                        _NUM_COLLIDED: num_collided,
                        _NUM_CRASHED: num_crashed,
                        _NUM_INTERCEPTED: num_intercepted,
                        _NUM_TIMED_OUT: num_timed_out,
                        _HAS_INTERCEPTS: plot_dataset[_HAS_INTERCEPTS],
                        _R2: r2,
                        _W_0: params[0],
                        _W_1: params[1],
                    }
                    writer.writerow(row_dict)
            baseline_row_dict = {
                _DATASET: _BASELINE,
                _NUM_TRAJECTORIES: 0,
                _NUM_COLLIDED: 0,
                _NUM_CRASHED: 0,
                _NUM_INTERCEPTED: 0,
                _NUM_TIMED_OUT: 0,
                _HAS_INTERCEPTS: False,
                _R2: 1,
                _W_0: baseline_w0,
                _W_1: baseline_w1,
            }
            writer.writerow(baseline_row_dict)
