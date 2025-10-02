import sys
import os
import ray                                      # type: ignore
import torch
from pathlib import Path  

from torch.optim import Adam                    # type: ignore

from ray import tune                            # type: ignore 
from ray.air import session                     # type: ignore                         # type: ignore 
# from ray.tune.schedulers import FIFOScheduler   # type: ignore
from ray.tune.schedulers import ASHAScheduler   # type: ignore
from collections import deque
import statistics

# Custom includes
from ..src.parameters import *
from ..src.Networks import *
from ..src.LearningAlgorithms import train_network, test_network, set_random_seed
from ..src.dataset import get_loader
from ..src.helpers import get_device, setup_storage_path

# TEMP, just for printing dataset
DUMP_DATASET = False

############
# TRAINING # --------------------------------------------------------------------------
############

def trainable(config, params):

    # Seed and device
    set_random_seed(config["seed"])
    assigned = ray.get_gpu_ids()
    device = get_device(params["quantization"], assigned)
    trial_id = session.get_trial_id()
    print(f"[trial {trial_id}] Using device: {device}")

    # Get the data loaders for training and testing
    trainloader, testloader = get_loader(params["dataset"], config["batch_size"], config["T"])

    # Create the network
    network = FCNetwork(
        layer_sizes = [params["in_size"]] + [config["layer_size"]] * config["n_layers"] + [params["n_class"]],
        beta        = config["beta"],
        quant       = params["quantization"]
    )
    
    # no debug set default dump dir to None
    run_dir = None

    # Optimizer
    if params["optimizer"]:
        optimizer = Adam(network.parameters(), lr=config["lr"])
    else:
        optimizer = None
        
    # Create the trainer
    torch.set_grad_enabled(False)
    trainer = STSFTrainer(network,
                            lr=config["lr"],
                            batch_size=config["batch_size"],
                            quant=params["quantization"],
                            use_optimizer=params["optimizer"],
                            optimizer=optimizer,
                            update_last=config["update_last"],
                            update_every=config["update_every"],
                            seq_batch_size=config["seq_batch"],
                            run_dir=run_dir,
                            debug_mode=params["debug_mode"],
                            debug_max_samples=params["debug_max_samples"]).to(device)

    trainer.network.train()  # set FCNetwork into train mode (no dropout/BN here)
    
    rolling_acc = deque(maxlen=5)
    prev_test = None
    # Training loop on epochs
    for _epoch in range(config["epochs"]):
        # TRAINING STEP
        training_loss, training_accuracy = train_network(trainer, trainloader, print_intermediate=params["print_intermediate"], device=device)
        # EARLY EXIT IF DEBUG ONLY
        if params["debug_mode"]:
            session.report({
                "testing_accuracy": -1,
                "training_accuracy": -1,
                "training_loss": -1
            })
            if hasattr(trainer, "close_debug_files"):
                trainer.close_debug_files()
            return
        # TESTING STEP
        testing_accuracy = test_network(network, testloader, print_intermediate=params["print_intermediate"], device=device)
        # REPORT (with epoch and stability helpers)
        rolling_acc.append(testing_accuracy)
        std_last5 = statistics.pstdev(rolling_acc) if len(rolling_acc) > 1 else 0.0
        delta = (testing_accuracy - prev_test) if prev_test is not None else 0.0
        prev_test = testing_accuracy

        session.report(
            testing_accuracy=testing_accuracy,
            training_accuracy=training_accuracy,
            training_loss=training_loss,
            epoch=_epoch + 1,
            test_acc_std_last5=std_last5,
            test_acc_delta=delta
        )



########
# MAIN # ------------------------------------------------------------------------------
########

def main(args):

    # ----------- Storage -----------------
    path = setup_storage_path(args.exp_name)
    print(f"Experiment path: {path}")
    
    # Extract storage path for tuner
    _storage_path = Path("../Log").resolve().as_posix()

    # -------- Set Parameters -------------

    trainable_parameters = {
        "exp_name":             args.exp_name,
        "n_class":              args.n_class,
        "in_size":              args.in_size,
        "quantization":         args.quantization,
        "optimizer":            args.optimizer,
        "bptt":                 args.bptt,
        "debug_mode":           args.debug_mode,
        "debug_max_samples":    args.debug_max_samples,
        "dataset":              args.dataset,
        "print_intermediate":   args.quantization and not args.tune
    }

    # ------------ Tuner ------------------

    if args.tune:   # TUNING
        # Select the appropriate search space based on dataset and quantization
        if args.quantization:
            search_space = search_spaces[args.dataset]["quantized"]
            cpu_per_trial = MACHINE_CONFIGS[args.machine]["CPU_PER_TRIAL_QUANT"]
            gpu_per_trial = MACHINE_CONFIGS[args.machine]["GPU_PER_TRIAL_QUANT"]
        else:
            search_space = search_spaces[args.dataset]["default"]
            cpu_per_trial = MACHINE_CONFIGS[args.machine]["CPU_PER_TRIAL"]
            gpu_per_trial = MACHINE_CONFIGS[args.machine]["GPU_PER_TRIAL"]
        num_samples = args.exploration_samples

    else:           # TRAINING  
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
        num_samples = 1
        cpu_per_trial = MACHINE_CONFIGS[args.machine]["TOTAL_CPUS"]
        gpu_per_trial = MACHINE_CONFIGS[args.machine]["TOTAL_GPUS"]

    # add exp name to search space for logging
    print(f"CPU per trial: {cpu_per_trial} | GPU per trial: {gpu_per_trial}")
    # make per-trial CPU visible inside trainable for thread heuristic
    trainable_parameters["cpu_per_trial"] = cpu_per_trial
    # Scheduler for Tuner trials
    # scheduler = FIFOScheduler()
    scheduler = ASHAScheduler(
        grace_period=2,        # wait a couple of epochs before pruning
        reduction_factor=3     # typical value; adjust if epochs are tiny
    )

    # Add parameters to the trainable function
    trainable_with_args = tune.with_parameters(trainable, params=trainable_parameters)
    # Resources for each trial
    trainable_with_resources = tune.with_resources(
        trainable=trainable_with_args,                          # This is the training function to run (one trial)
        resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial}  # Resources to allocate for each trial
    )
    # Launch experiment without possibility to restore, if want to add check previous versions
    tuner = tune.Tuner(
        trainable_with_resources,       # The trainable function + resource config
        param_space=search_space,       # Hyperparameter config to use for the trial
        tune_config=tune.TuneConfig(
            num_samples=num_samples,    # Number of samples to try (1 for static, >1 for search)
            scheduler=scheduler,        # Use FIFO scheduler to manage the trial
            metric="testing_accuracy",  # <-- tell Ray Tune which metric to optimize
            mode="max"                  # <-- maximize testing accuracy
        ),
        run_config=tune.RunConfig(
            storage_path=_storage_path,     # Where to store logs and checkpoints
            name=args.exp_name              # Experiment folder name (used under local_dir)
        )
    )
    # Start the tuning process
    tuner.fit()



###############
# ENTRY POINT # -----------------------------------------------------------------------
###############

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize Ray with proper error handling
    try:
        ray.init(
            num_cpus=MACHINE_CONFIGS[args.machine]["TOTAL_CPUS"],
            num_gpus=MACHINE_CONFIGS[args.machine]["TOTAL_GPUS"],
            include_dashboard=False,
            ignore_reinit_error=True,
            _temp_dir=os.environ.get("TMPDIR", None)
        )

        print("Ray initialized.")
        print(f"Machines config: {args.machine}\nCPUs: {ray.available_resources().get('CPU', 0)} | GPUs: {ray.available_resources().get('GPU', 0)}")
        # Run main experiment
        main(args)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise

    finally:
        # Ensure proper cleanup
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown complete.")
