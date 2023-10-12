import argparse

from drone_path_planning.routines.routine import Routine
from drone_path_planning.scenarios import SCENARIOS


class TrainRoutine(Routine):
    def setup_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('scenario', choices=SCENARIOS)
        parser.add_argument('--save_dir', required=True)
        parser.add_argument('--logs_dir')

    def run(self, args: argparse.Namespace):
        scenario = SCENARIOS[args.scenario]()
        trainer = scenario.create_trainer(args.save_dir, args.logs_dir)
        trainer.initialize()
        trainer.train()
