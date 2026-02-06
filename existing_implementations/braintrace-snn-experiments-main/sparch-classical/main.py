"""
This is the script used to run experiments.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from args import add_training_options, add_model_options
from general_utils import MyArgumentParser


def parse_args():
    parser = MyArgumentParser(description="Model training on spiking speech commands datasets.")
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args()
    return args


def main():
    """
    Runs model training/testing using the configuration specified
    by the parser arguments. Run `python main.py -h` for details.
    """

    # Get experiment configuration from parser
    args = parse_args()

    # Instantiate class for the desired experiment
    from exp import Experiment
    experiment = Experiment(args)

    # Run experiment
    if args.mode == 'train':
        experiment.f_train()
    elif args.mode == 'test':
        experiment.f_test(8)
    else:
        raise ValueError("Mode must be either 'train' or 'test'")


if __name__ == "__main__":
    main()
