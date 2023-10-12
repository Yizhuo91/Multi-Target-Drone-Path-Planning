import argparse
import json
import os

from drone_path_planning.plotters import PLOTTERS
from drone_path_planning.routines.routine import Routine


class PlotRoutine(Routine):
    def setup_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('plotter', choices=PLOTTERS)
        parser.add_argument('--plot_data_config', required=True)
        parser.add_argument('--plots_dir', required=True)

    def run(self, args: argparse.Namespace):
        with open(args.plot_data_config) as f:
            plot_data_config = json.load(f)
        plotter = PLOTTERS[args.plotter](plot_data_config)
        plotter.load_data()
        plotter.process_data()
        os.makedirs(args.plots_dir, exist_ok=True)
        plotter.plot(args.plots_dir)
