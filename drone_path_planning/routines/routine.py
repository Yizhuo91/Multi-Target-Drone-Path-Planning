import abc
import argparse


class Routine:
    @abc.abstractmethod
    def setup_parser(self, parser: argparse.ArgumentParser):
        raise NotImplementedError()

    @abc.abstractmethod
    def run(self, args: argparse.Namespace):
        raise NotImplementedError()
