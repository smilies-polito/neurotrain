"""
This is the script used to run experiments.
"""

import os
import platform
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from general_utils import MyArgumentParser


def strtobool(val):
    """
    Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")


num_worker = 0 if platform.system() == 'Windows' else 5
shd_path = '../data/SHD/'


def add_training_options(parser_):
    parser_.add_argument("--load_exp_folder", type=str, default=None,
                         help="Path to experiment folder with a pretrained model to load. Note "
                              "that the same path will be used to store the current experiment.")
    parser_.add_argument("--mode", type=str, default='train', )
    parser_.add_argument("--new_exp_folder", type=str, default=None, help="Path to output folder to store experiment.")
    parser_.add_argument("--dataset_name", choices=["shd", "ssc", "gesture", "gesturev2", "nmnist", "nmnistv2"],
                         type=str, default="shd", help="Dataset name (shd, ssc, hd or sc).")
    args, _ = parser_.parse_known_args()

    parser_.add_argument('--data_length', type=int, default=100)
    parser_.add_argument("--data_folder", type=str, default=shd_path, help="Path to dataset folder.", )
    parser_.add_argument("--save_best", type=lambda x: bool(strtobool(str(x))), default=True,
                         help="If True, the model from the epoch with the highest validation "
                              "accuracy is saved, if False, no model is saved.")
    parser_.add_argument("--batch_size", type=int, default=128, help="Number of input examples inside a single batch.")
    parser_.add_argument("--nb_epochs", type=int, default=5, help="Number of training epochs.")
    parser_.add_argument("--num_workers", type=int, default=num_worker)
    parser_.add_argument("--start_epoch", type=int, default=0,
                         help="Epoch number to start training at. Will be 0 if no pretrained "
                              "model is given. First epoch will be start_epoch+1.")
    parser_.add_argument("--lr", type=float, default=1e-2,
                         help="Initial learning rate for training. The default value of 0.01 "
                              "is good for SHD and SC, but 0.001 seemed to work better for HD and SC.")
    parser_.add_argument("--lr_step_size", type=int, default=10,
                         help="Number of epochs without progress before the learning rate gets decreased.")
    parser_.add_argument("--lr_step_gamma", type=float, default=0.9,
                         help="Factor between 0 and 1 by which the learning rate gets "
                              "decreased when the scheduler patience is reached.")
    parser_.add_argument("--use_augm", type=lambda x: bool(strtobool(str(x))), default=False,
                         help="Whether to use data augmentation or not. Only implemented for "
                              "non-spiking HD and SC datasets.")
    return parser_


def add_model_options(parser_):
    parser_.add_argument(
        "--model_type", type=str, default="LIF", help="Type of ANN or SNN model.",
        choices=["LIF", "adLIF", "RLIF", "RadLIF", "MLP", "RNN", "LiGRU", "GRU"],
    )
    parser_.add_argument("--nb_layers", type=int, default=3, help="Number of layers (including readout layer).")
    parser_.add_argument("--nb_hiddens", type=int, default=128, help="Number of neurons in all hidden layers.")
    parser_.add_argument("--pdrop", type=float, default=0.1, help="Dropout rate, must be between 0 and 1.")
    parser_.add_argument("--inp_scale", type=float, default=5 ** 0.5)
    parser_.add_argument("--rec_scale", type=float, default=1.0)
    parser_.add_argument("--momentum", type=float, default=0.99)
    parser_.add_argument("--relu_width", type=float, default=1.0)
    parser_.add_argument(
        "--normalization", type=str, default="none", choices=["none", "batchnorm", "layernorm"],
        help="Type of normalization, every string different from batchnorm "
    )
    parser_.add_argument(
        "--use_bias", type=lambda x: bool(strtobool(str(x))), default=False,
        help="Whether to include trainable bias with feedforward weights."
    )
    return parser_


parser = MyArgumentParser(description="Model training on spiking speech commands datasets.", method='esd-rtrl')
parser = add_model_options(parser)
parser = add_training_options(parser)
args = parser.parse_args()


def main():
    from online_model import Experiment

    # Instantiate class for the desired experiment
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
