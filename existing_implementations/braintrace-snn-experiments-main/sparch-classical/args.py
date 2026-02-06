#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the parser for the training configuration is defined.
"""
import platform


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


if platform.platform() == 'Windows-10-10.0.26100-SP0':
    shd_path = 'D:/data/shd/'
    ssc_path = 'D:/data/ssc/'
    num_worker = 0
# 雷神WSL
elif platform.platform() == 'Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.35':
    shd_path = '/mnt/d/data/shd/'
    ssc_path = '/mnt/d/data/ssc/'
    num_worker = 10

# 吴思Lab A100
elif platform.platform() == 'Linux-6.8.0-52-generic-x86_64-with-glibc2.35':
    shd_path = '/home/chaomingwang/data/shd/'
    ssc_path = '/home/chaomingwang/data/ssc/'
    num_worker = 10

# 横琴A100
elif platform.platform() == 'Linux-5.15.0-84-generic-x86_64-with-glibc2.31':
    shd_path = '/home/chaoming/data/shd/'
    ssc_path = '/home/chaoming/data/ssc/'
    num_worker = 10

# 吴思Lab brainpy-tower1-brainpy
elif platform.platform() == 'Linux-6.8.0-48-generic-x86_64-with-glibc2.35':
    shd_path = '/home/brainpy/data/shd/'
    ssc_path = '/home/brainpy/data/ssc/'
    num_worker = 10

else:
    num_worker = 8
    shd_path = '../data/SHD/'
    ssc_path = None


def add_training_options(parser):
    parser.add_argument(
        "--load_exp_folder",
        type=str,
        default=None,
        help="Path to experiment folder with a pretrained model to load. Note "
             "that the same path will be used to store the current experiment."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='train',
    )
    parser.add_argument(
        "--new_exp_folder",
        type=str,
        default=None,
        help="Path to output folder to store experiment."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["shd", "ssc", "gesture", "gesturev2", "nmnist", "nmnistv2"],
        default="shd", help="Dataset name (shd, ssc, hd or sc)."
    )
    args, _ = parser.parse_known_args()

    if args.dataset_name.startswith("shd"):
        path = shd_path
        data_length = 100
    elif args.dataset_name.startswith("ssc"):
        path = ssc_path
        data_length = 100
    elif args.dataset_name.startswith('gesture'):
        path = '../data'
        data_length = 200
    elif args.dataset_name.startswith('nmnist'):
        path = '../data'
        data_length = 200
    else:
        path = '../data'
        data_length = 200

    parser.add_argument(
        '--data_length',
        type=int,
        default=data_length
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=path,
        help="Path to dataset folder.",
    )
    parser.add_argument(
        "--save_best",
        type=lambda x: bool(strtobool(str(x))), default=True,
        help="If True, the model from the epoch with the highest validation "
             "accuracy is saved, if False, no model is saved."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of input examples inside a single batch.",
    )
    parser.add_argument(
        "--nb_epochs",
        type=int,
        default=5,
        help="Number of training epochs (i.e. passes through the dataset)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=num_worker,
        help="Number of training epochs (i.e. passes through the dataset)."
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Epoch number to start training at. Will be 0 if no pretrained "
             "model is given. First epoch will be start_epoch+1."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Initial learning rate for training. The default value of 0.01 "
             "is good for SHD and SC, but 0.001 seemed to work better for HD and SC."
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=10,
        help="Number of epochs without progress before the learning rate gets decreased."
    )
    parser.add_argument(
        "--lr_step_gamma",
        type=float,
        default=0.9,
        help="Factor between 0 and 1 by which the learning rate gets "
             "decreased when the scheduler patience is reached."
    )
    parser.add_argument(
        "--use_augm",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to use data augmentation or not. Only implemented for "
             "non-spiking HD and SC datasets."
    )
    return parser


def add_model_options(parser):
    parser.add_argument(
        "--model_type",
        type=str,
        default="LIF",
        help="Type of ANN or SNN model.",
        choices=["LIF", "adLIF", "RLIF", "RadLIF", "MLP", "RNN", "LiGRU", "GRU"],
    )
    parser.add_argument(
        "--nb_layers",
        type=int,
        default=3,
        help="Number of layers (including readout layer)."
    )
    parser.add_argument(
        "--nb_hiddens",
        type=int,
        default=128,
        help="Number of neurons in all hidden layers."
    )
    parser.add_argument(
        "--pdrop",
        type=float,
        default=0.1,
        help="Dropout rate, must be between 0 and 1."
    )
    parser.add_argument(
        "--inp_scale",
        type=float,
        default=5 ** 0.5
    )
    parser.add_argument(
        "--rec_scale",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="none",
        choices=["none", "batchnorm", "layernorm"],
        help="Type of normalization, every string different from batchnorm "
    )
    parser.add_argument(
        "--use_bias",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to include trainable bias with feedforward weights."
    )
    return parser


def print_model_options(logger, args):
    logger.warning(
        """
        Model Config
        ------------
        Model Type: {model_type}
        Number of layers: {nb_layers}
        Number of hidden neurons: {nb_hiddens}
        Dropout rate: {pdrop}
        Normalization: {normalization}
        Use bias: {use_bias}
    """.format(**vars(args))
    )


def print_training_options(logger, args):
    logger.warning(
        """
        Training Config
        ---------------
        Load experiment folder: {load_exp_folder}
        New experiment folder: {new_exp_folder}
        Dataset name: {dataset_name}
        Data folder: {data_folder}
        Save best model: {save_best}
        Batch size: {batch_size}
        Number of epochs: {nb_epochs}
        Start epoch: {start_epoch}
        Initial learning rate: {lr}
        Use data augmentation: {use_augm}
    """.format(**vars(args))
    )
