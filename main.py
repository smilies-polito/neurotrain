import sys
import os
import torch
from pathlib import Path
from torch.optim import Adam

from collections import deque
import statistics

# Custom includes
from utils.helpers import get_device, setup_storage_path, set_random_seed
from utils.parameters import parse_args  # adjust if constants live elsewhere
from datasets.get_loader import get_loader
from networks.fc_network import FCNetwork
from trainers.stsf_trainer import STSFTrainer
from LearningAlgorithms import LearningAlgorithms

# Add the src folder to the Python module search path - this might become a pip package later
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# TEMP, just for printing dataset
DUMP_DATASET = False

############
# TRAINING #
############

def trainable(config, params, trainer_class):
    # Seed and device
    set_random_seed(config["seed"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Get the data loaders for training and testing
    trainloader, testloader = get_loader(params["dataset"], config["batch_size"], config["T"])

    # Create the network
    network = FCNetwork(
        layer_sizes = [params["in_size"]] + [config["layer_size"]] * config["n_layers"] + [params["n_class"]],
        beta        = config["beta"]
    )

    # Optimizer
    if params["optimizer"]:
        optimizer = Adam(network.parameters(), lr=config["lr"])
    else:
        optimizer = None
        
    # Create the trainer
    torch.set_grad_enabled(False)
    trainer = trainer_class(
        network=network,
        lr=config["lr"],
        batch_size=config["batch_size"],
        quant=params["quantization"],
        use_optimizer=params["optimizer"],
        optimizer=optimizer,
        update_last=config["update_last"],
        update_every=config["update_every"],
        seq_batch_size=config["seq_batch"]
    ).to(device)

    trainer.network.train()

    rolling_acc = deque(maxlen=5)
    prev_test = None

    # Training loop on epochs
    for _epoch in range(config["epochs"]):
        # TRAINING STEP
        training_metrics = LearningAlgorithms.train_epoch(trainer, trainloader, device=device, print_every=1000)
        training_loss = training_metrics["loss"]
        training_accuracy = training_metrics["accuracy"]

        # TESTING STEP
        testing_metrics = LearningAlgorithms.evaluate(network, testloader, device=device, print_every=1000)
        testing_accuracy = testing_metrics["accuracy"]

        # REPORT (with epoch and stability helpers)
        rolling_acc.append(testing_accuracy)
        std_last5 = statistics.pstdev(rolling_acc) if len(rolling_acc) > 1 else 0.0
        delta = (testing_accuracy - prev_test) if prev_test is not None else 0.0
        prev_test = testing_accuracy

        print({
            "testing_accuracy": testing_accuracy,
            "training_accuracy": training_accuracy,
            "training_loss": training_loss,
            "epoch": _epoch + 1,
            "test_acc_std_last5": std_last5,
            "test_acc_delta": delta
        })

# Trainer factory
def get_trainer(trainer_name):
    trainers = {
        "stsf": STSFTrainer,
    }
    return trainers[trainer_name]

########
# MAIN #
########

def main(args):
    # ----------- Storage -----------------
    path = setup_storage_path(args.exp_name)
    print(f"Experiment path: {path}")

    # -------- Set Parameters -------------

    trainable_parameters = {
        "exp_name":             args.exp_name,
        "n_class":              args.n_class,
        "in_size":              args.in_size,
        "quantization":         args.quantization,
        "optimizer":            args.optimizer,
        "bptt":                 args.bptt,
        "dataset":              args.dataset,
        "print_intermediate":   args.quantization and not args.tune
    }

    # ------------ Training ------------------

    search_space = {
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "T":            args.T,
        "threshold":    args.threshold,
        "beta":         args.beta,
        "n_layers":     args.n_layers,
        "layer_size":   args.layer_size,
        "seed":         args.seed,
        "update_last":  args.update_last,
        "update_every": args.update_every,
        "seq_batch":    args.seq_batch
    }

    trainer_class = get_trainer("stsf")
    trainable(config=search_space, params=trainable_parameters, trainer_class=trainer_class)

###############
# ENTRY POINT # 
###############

if __name__ == "__main__":
    # Parse command-line arguments from parameters.py
    args = parse_args()
    # Call the main function with parsed arguments
    main(args)
