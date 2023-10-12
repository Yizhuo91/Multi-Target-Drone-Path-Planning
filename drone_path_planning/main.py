import argparse
import logging

from drone_path_planning.routines import ROUTINES


_PROGRAM_DESCRIPTION = 'Use reinforcement learning to train a drone to plan its path.'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=_PROGRAM_DESCRIPTION,
    )
    subparsers = parser.add_subparsers(
        title='routines',
        description='valid routines',
        dest='routine',
    )
    for routine_name, routine in ROUTINES.items():
        routine_parser = subparsers.add_parser(routine_name)
        routine.setup_parser(routine_parser)
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    routine = ROUTINES[args.routine]
    routine.run(args)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    args = _parse_args()
    main(args)
