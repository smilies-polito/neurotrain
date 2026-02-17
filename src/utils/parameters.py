import argparse
try:
    from ray import tune  # type: ignore
except ModuleNotFoundError:
    # Keep argument parsing usable when Ray Tune is not installed.
    class _TuneStub:
        @staticmethod
        def choice(values):
            return values[0]

        @staticmethod
        def loguniform(low, _high):
            return low

        @staticmethod
        def randint(low, _high):
            return low

    tune = _TuneStub()

##########################
# Exploration Parameters #
##########################
# In ray tune you can set the values for exploration like this:
#   "parameter_name": tune.choice([list, of, possible, values])
#   "parameter_name": tune.loguniform(min_value, max_value)
#   "parameter_name": tune.randint(min_value, max_value)

search_spaces = {
    "MNIST": {
        "default": {
            "epochs":       100,
            "batch_size":   256,
            "lr":           tune.loguniform(1e-3, 1e-1),
            "T":            10,
            "threshold":    1.0,
            "beta":         0.9375,
            "n_layers":     1,
            "layer_size":   tune.choice([100, 125, 150, 175, 200, 225, 250, 275, 300]),
            "seed":         tune.randint(1, 2**31-1),
            "update_last":  False,
            "update_every": 1,
            "seq_batch":    1
        },
        "quantized": {
            "epochs":       1,
            "batch_size":   1,
            "lr":           tune.loguniform(0.08, 0.15),
            "T":            10,
            "threshold":    tune.choice([0.75, 0.875, 1.0, 1.125, 1.25]),
            "beta":         tune.choice([0.5, 0.75,0.875, 0.9375, 0.96875, 0.984375]),
            "n_layers":     1,
            "layer_size":   200,
            "seed":         tune.randint(1, 2**31-1),
            "update_last":  False,
            "update_every": 5,
            "seq_batch":    1
        }
    },
    "FashionMNIST": {
        "default": {
            "epochs":       100,
            "batch_size":   256,
            "lr":           tune.loguniform(0.08, 0.15),
            "T":            10,
            "threshold":    1.0,
            "beta":         0.9375,
            "n_layers":     1,
            "layer_size":   tune.choice([400, 500, 600, 700, 800, 900, 1000, 1100]),
            "seed":         tune.randint(1, 2**31-1),
            "update_last":  False,
            "update_every": 1,
            "seq_batch":    1
        },
        "quantized": {
            "epochs":       10,
            "batch_size":   1,
            "lr":           tune.loguniform(0.08, 0.15),
            "T":            10,
            "threshold":    1.0,
            "beta":         0.9375,
            "n_layers":     1,
            "layer_size":   tune.choice([400, 500, 600, 700, 800, 900, 1000, 1100]),
            "seed":         tune.randint(1, 2**31-1),
            "update_last":  False,
            "update_every": 5,
            "seq_batch":    1
        }
    }
}

MACHINE_CONFIGS = {
    "hpc": {
        "TOTAL_CPUS": 48,
        "TOTAL_GPUS": 4,
        "CPU_PER_TRIAL": 4,
        "GPU_PER_TRIAL": 0.5,
        "CPU_PER_TRIAL_QUANT": 2,
        "GPU_PER_TRIAL_QUANT": 0
    },
    "server": {
        "TOTAL_CPUS": 64,
        "TOTAL_GPUS": 1,
        "CPU_PER_TRIAL": 16,
        "GPU_PER_TRIAL": 0.5,
        "CPU_PER_TRIAL_QUANT": 4,
        "GPU_PER_TRIAL_QUANT": 0
    },
    "generic": {
        "TOTAL_CPUS": 1,
        "TOTAL_GPUS": 1,
        "CPU_PER_TRIAL": 1,
        "GPU_PER_TRIAL": 1,
        "CPU_PER_TRIAL_QUANT": 1,
        "GPU_PER_TRIAL_QUANT": 0
    }
}



####################
# Hardcoded values #
####################

# Results log interval
LOG_N = 1
# Quantization parameters
FP_DEC = 8      # Number of fractional bits in fixed-point representation
BW = 16         # Bit-width of the quantized values
# Default values
DEFAULT_DATASET         = "MNIST"
DEFAULT_LAYER_SIZE      = 200
DEFAULT_N_LAYERS        = 1
DEFAULT_EPOCHS          = 100
DEFAULT_BATCH_SIZE      = 256
DEFAULT_LR              = 1e-2
DEFAULT_T_STEPS         = 10
DEFAULT_THRESHOLD       = 1.0
DEFAULT_BETA            = 0.9375
DEFAULT_VMAX            = 1.0       # TODO: remove this since it's not used
DEFAULT_SEED            = 42
DEFAULT_QUANTIZATION    = False
DEFAULT_UPDATE_EVERY    = 1  


###########
# PARSING #
###########

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train spiking network with different datasets and hyperparameters"
    )

    # Configuration file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML/JSON config file. CLI args override config file values."
    )

    # Resume from checkpoint
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint in experiment directory"
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from specific checkpoint file path"
    )
    
    # Dataset / architecture
    parser.add_argument(
        "--dataset", type=str,
        choices=["MNIST", "CIFAR10", "FashionMNIST", "SVHN", "NMNIST", "DVSGesture"],
        default=DEFAULT_DATASET,
        help="Which dataset to use."
    )
    parser.add_argument(
        "--layer-size", type=int,
        default=DEFAULT_LAYER_SIZE,
        help="Number of neurons per hidden layer"
    )
    parser.add_argument(
        "--n-layers", type=int,
        default=DEFAULT_N_LAYERS,
        help="Number of hidden layers"
    )

    # Hyperparameters from search_space
    parser.add_argument(
        "--epochs", type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float,
        default=DEFAULT_LR,
        help="Learning rate"
    )
    parser.add_argument(
        "--T", type=int,
        default=DEFAULT_T_STEPS,
        help="Number of time steps for each sample"
    )
    parser.add_argument(
        "--threshold", type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold for LIF neuron"
    )
    parser.add_argument(
        "--beta", type=float,
        default=DEFAULT_BETA,
        help="beta for LIF neuron"
    )
    parser.add_argument(
        "--seed", type=int,
        default=42,
        help="Random seed"
    )

    # Hyperparam exploration
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable Ray Tune hyperparameter search"
    )
    parser.add_argument(
        "--exploration-samples", type=int,
        default=50,
        help="Number of samples for hyperparameter exploration (default: 50)"
    )

    # Run variations
    parser.add_argument(
        "--quantization", action="store_true",
        help="Enable quantization of the network"
    )
    parser.add_argument(
        "--optimizer", action="store_true",
        help="Use an optimizer for the training"
    )
    parser.add_argument(
        "--bptt", action="store_true",
        help="Use Backpropagation through time (BPTT) instead of STSF"
    )
    parser.add_argument(
        "--update-last", action="store_true",
        help="Update the last layer only during the last time step"
    )
    parser.add_argument(
        "--update-every", type=int,
        default=DEFAULT_UPDATE_EVERY,
        help="Update weights every N time steps (default: 1)"
    )
    parser.add_argument(
        "--seq-batch", type=int,
        default=1,
        help="Number of samples to process before updating (default: 1)"
    )
    parser.add_argument(
        "--machine", type=str,
        choices=["hpc", "server", "generic"],
        default="server",
        help="Machine configuration to use for the experiment"
    )
    parser.add_argument(
        "--debug-mode", action="store_true",
        help="Enable debug mode to save runtime information files"
    )
    parser.add_argument(
        "--debug-max-samples", type=int, default=1,
        help="Maximum number of samples to debug (default: 1)"
    )
    # Parse arguments
    args = parser.parse_args()

    # Conditions on arguments (TODO: add more maybe)
    if args.seq_batch > 1: args.batch_size = 1  # If seq-batch > 1, batch size must be 1

    # Set default values based on dataset
    if args.dataset == "MNIST":
        args.in_size, args.n_class, args.exp_name = 28*28, 10, "STSF_MNIST"
    elif args.dataset == "CIFAR10":
        args.in_size, args.n_class, args.exp_name = 32*32*3, 10, "STSF_CIFAR10"
    elif args.dataset == "FashionMNIST":
        args.in_size, args.n_class, args.exp_name = 28*28, 10, "STSF_FASHIONMNIST"
    elif args.dataset == "SVHN":
        args.in_size, args.n_class, args.exp_name = 32*32*3, 10, "STSF_SVHN"
    elif args.dataset == "DVSGesture":
        args.in_size, args.n_class, args.exp_name = 34*34, 11, "STSF_DVSGesture"
    # elif DATASET == "NMNIST":
    #     args.in_size, args.nclass, args.expname = 34*34*T_STEPS, 10, "STSF_NMNIST"
    else:
        raise ValueError("Unsupported dataset")

    return args
