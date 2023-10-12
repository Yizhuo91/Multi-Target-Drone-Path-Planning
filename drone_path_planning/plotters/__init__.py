from typing import Dict
from typing import Type

from drone_path_planning.plotters.chaser_target_obstacle_plotter import ChaserTargetObstaclePlotter
from drone_path_planning.plotters.chaser_target_plotter import ChaserTargetPlotter
from drone_path_planning.plotters.plotter import Plotter


PLOTTERS: Dict[str, Type[Plotter]] = {
    'chaser_target_obstacle_plotter': ChaserTargetObstaclePlotter,
    'chaser_target_plotter': ChaserTargetPlotter,
}
