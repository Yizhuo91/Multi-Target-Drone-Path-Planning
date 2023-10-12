import abc
from typing import Optional

from drone_path_planning.evaluators import Evaluator
from drone_path_planning.trainers import Trainer


class Scenario:
    @abc.abstractmethod
    def create_trainer(self, save_dir: str, logs_dir: Optional[str] = None) -> Trainer:
        raise NotImplementedError()

    @abc.abstractmethod
    def create_evaluator(self, save_dir: str, plot_data_dir: str, logs_dir: Optional[str] = None) -> Evaluator:
        raise NotImplementedError()
