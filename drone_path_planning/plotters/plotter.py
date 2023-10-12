import abc
from typing import Any


class Plotter:
    def __init__(self, plot_data_config: Any):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def process_data(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def plot(self, plots_dir: str):
        raise NotImplementedError()
