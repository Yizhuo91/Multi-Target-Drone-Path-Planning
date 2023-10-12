from typing import Dict
from typing import Type

from drone_path_planning.scenarios.four_chasers_one_target_2d_scenario import FourChasersOneTarget2DScenario
from drone_path_planning.scenarios.four_chasers_single_moving_target_scenario import FourChasersSingleMovingTargetScenario
from drone_path_planning.scenarios.scenario import Scenario
from drone_path_planning.scenarios.one_chaser_one_target_2d_scenario import OneChaserOneTarget2DScenario
from drone_path_planning.scenarios.one_chaser_single_moving_target_scenario import OneChaserSingleMovingTargetScenario
from drone_path_planning.scenarios.one_three_chasers_one_target_obstacle_scenario import OneThreeChasersOneTargetObstacleScenario
from drone_path_planning.scenarios.one_three_chasers_one_target_obstacle_2d_scenario import OneThreeChasersOneTargetObstacle2DScenario
from drone_path_planning.scenarios.one_three_chasers_single_moving_target_scenario import OneThreeChasersSingleMovingTargetScenario
from drone_path_planning.scenarios.single_chaser_single_moving_target_scenario import SingleChaserSingleMovingTargetScenario
from drone_path_planning.scenarios.three_chasers_one_target_2d_scenario import ThreeChasersOneTarget2DScenario
from drone_path_planning.scenarios.three_chasers_single_moving_target_scenario import ThreeChasersSingleMovingTargetScenario
from drone_path_planning.scenarios.two_chasers_one_target_2d_scenario import TwoChasersOneTarget2DScenario
from drone_path_planning.scenarios.two_chasers_single_moving_target_scenario import TwoChasersSingleMovingTargetScenario


SCENARIOS: Dict[str, Type[Scenario]] = {
    'four-chasers_one-target_2d': FourChasersOneTarget2DScenario,
    'four-chasers_single-moving-target': FourChasersSingleMovingTargetScenario,
    'one-chaser_one-target_2d': OneChaserOneTarget2DScenario,
    'one-chaser_single-moving-target': OneChaserSingleMovingTargetScenario,
    'one-three-chasers_one-target_obstacle': OneThreeChasersOneTargetObstacleScenario,
    'one-three-chasers_one-target_obstacle_2d': OneThreeChasersOneTargetObstacle2DScenario,
    'one-three-chasers_single-moving-target': OneThreeChasersSingleMovingTargetScenario,
    'single-chaser_single-moving-target': SingleChaserSingleMovingTargetScenario,
    'three-chasers_one-target_2d': ThreeChasersOneTarget2DScenario,
    'three-chasers_single-moving-target': ThreeChasersSingleMovingTargetScenario,
    'two-chasers_one-target_2d': TwoChasersOneTarget2DScenario,
    'two-chasers_single-moving-target': TwoChasersSingleMovingTargetScenario,
}
