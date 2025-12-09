"""
SNN Training Benchmarking - Main Entry Point.

Supports:
- YAML/JSON configuration files
- CLI argument overrides
- Checkpointing and resume
- Reproducibility logging
- TensorBoard integration
"""

import sys
import os
from collections import deque
from pathlib import Path
import statistics

import torch
from torch.optim import Adam

# Add the src folder to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Core imports
from utils.parameters import parse_args
from utils.config import (
    Config,
    load_config,
    merge_config_with_args,
    validate_config,
    print_config,
)
from utils.checkpoint import CheckpointManager, resume_training
from utils.experiment_logger import (
    ExperimentLogger,
    set_all_seeds,
    print_experiment_info,
)
from datasets.get_loader import get_loader
from networks.fc_network import FCNetwork
from networks.decolle_network import DecolleNetwork
from trainers.stsf_trainer import STSFTrainer
from trainers.bptt_trainer import BPTTTrainer
from trainers.decolle_trainer import DECOLLETrainer
from LearningAlgorithms import LearningAlgorithms


############
# TRAINING #
############


def get_device(config: Config) -> torch.device:
    """Determine the appropriate device based on config and availability."""
    device_str = config.hardware.device

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def trainable(config: Config, trainer_class, logger: ExperimentLogger, checkpoint_manager: CheckpointManager, start_epoch: int = 0):
    """
    Main training function.

    Args:
        config: Experiment configuration
        trainer_class: Trainer class to use
        logger: Experiment logger for metrics
        checkpoint_manager: Checkpoint manager for saving/loading
        start_epoch: Starting epoch (for resume)
    """
    # Get device
    device = get_device(config)
    print(f"Using device: {device}")

    # Get data loaders
    trainloader, testloader = get_loader(
        config.data.dataset,
        config.training.batch_size,
        config.data.timesteps,
    )

    # Create the network (DECOLLE requires explicit stateful network)
    if trainer_class is DECOLLETrainer:
        network = DecolleNetwork(layer_sizes=config.model.layer_sizes)
    else:
        network = FCNetwork(
            layer_sizes=config.model.layer_sizes,
            beta=config.model.beta,
            quant=config.model.quantization,
        )

    # Optimizer
    if config.training.optimizer == "adam":
        optimizer = Adam(
            network.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    else:
        optimizer = None

    # Create the trainer
    # Enable gradients for BPTT (gradient-based), disable for local learners (STSF/DECOLLE)
    requires_grad = config.trainer.name == "bptt"
    torch.set_grad_enabled(requires_grad)
    trainer = trainer_class(
        network=network,
        lr=config.training.learning_rate,
        batch_size=config.training.batch_size,
        quant=config.model.quantization,
        use_optimizer=config.training.optimizer is not None,
        optimizer=optimizer,
        update_last=config.trainer.update_last,
        update_every=config.trainer.update_every,
        seq_batch_size=config.trainer.seq_batch,
    ).to(device)

    trainer.network.train()

    # Setup graceful exit (save checkpoint on Ctrl+C)
    checkpoint_manager.setup_graceful_exit(
        trainer.network, optimizer, config.to_dict()
    )

    # Tracking
    rolling_acc = deque(maxlen=5)
    prev_test = None
    best_accuracy = 0.0

    # Training loop
    for epoch in range(start_epoch, config.training.epochs):
        # TRAINING STEP
        training_metrics = LearningAlgorithms.train_epoch(
            trainer, trainloader, device=device, print_every=1000
        )
        training_loss = training_metrics["loss"]
        training_accuracy = training_metrics["accuracy"]

        # TESTING STEP
        testing_metrics = LearningAlgorithms.evaluate(
            network, testloader, device=device, print_every=1000
        )
        testing_accuracy = testing_metrics["accuracy"]

        # Compute stability metrics
        rolling_acc.append(testing_accuracy)
        std_last5 = statistics.pstdev(rolling_acc) if len(rolling_acc) > 1 else 0.0
        delta = (testing_accuracy - prev_test) if prev_test is not None else 0.0
        prev_test = testing_accuracy

        # Update best
        if testing_accuracy > best_accuracy:
            best_accuracy = testing_accuracy

        # Prepare metrics dictionary
        metrics = {
            "accuracy": testing_accuracy,
            "training_accuracy": training_accuracy,
            "training_loss": training_loss,
            "test_acc_std_last5": std_last5,
            "test_acc_delta": delta,
        }

        # Log to TensorBoard
        logger.log_metrics(metrics, step=epoch, prefix="train")

        # Print progress
        print({
            "epoch": epoch + 1,
            "testing_accuracy": testing_accuracy,
            "training_accuracy": training_accuracy,
            "training_loss": training_loss,
            "test_acc_std_last5": std_last5,
            "test_acc_delta": delta,
        })

        # Save checkpoint if needed
        checkpoint_manager.save_if_needed(
            model=trainer.network,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config.to_dict(),
        )

    # Log final hyperparameters with results
    logger.log_hyperparameters(
        config.to_flat_dict(),
        {"final_accuracy": testing_accuracy, "best_accuracy": best_accuracy},
    )

    # Close logger
    logger.close()

    print(f"\nTraining complete! Best accuracy: {best_accuracy:.4f}")
    return best_accuracy


# Trainer factory
def get_trainer(trainer_name: str):
    """Get trainer class by name."""
    trainers = {
        "stsf": STSFTrainer,
        "bptt": BPTTTrainer,
        "decolle": DECOLLETrainer,
        # Future trainers will be added here:
        # "eprop": EpropTrainer,
        # "stdp": STDPTrainer,
    }
    if trainer_name not in trainers:
        raise ValueError(
            f"Unknown trainer: {trainer_name}. Available: {list(trainers.keys())}"
        )
    return trainers[trainer_name]


########
# MAIN #
########


def main(args=None):
    """Main entry point."""
    # Parse CLI arguments
    if args is None:
        args = parse_args()

    # Load configuration
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
        # Merge with CLI args (CLI takes precedence)
        config = merge_config_with_args(config, args)
    else:
        # Create config from CLI args only (backward compatible)
        config = Config()
        config = merge_config_with_args(config, args)

    # Validate configuration
    issues = validate_config(config)
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        if any("must" in issue for issue in issues):
            sys.exit(1)

    # Print configuration
    print_config(config)

    # Setup experiment logger
    logger = ExperimentLogger(
        experiment_name=config.experiment.name,
        config=config.to_dict(),
        seed=config.experiment.seed,
        log_dir=f"{config.experiment.log_dir}/{config.experiment.name}",
        deterministic=config.experiment.deterministic,
    )

    # Setup experiment (sets seeds, captures environment)
    device = get_device(config)
    context = logger.setup(device)

    # Print experiment info
    print_experiment_info(context)

    # Save experiment context
    logger.save_context()

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(logger.log_dir) / "checkpoints",
        save_best=config.checkpoint.save_best,
        save_latest=config.checkpoint.save_latest,
        save_every=config.checkpoint.save_every,
        max_keep=config.checkpoint.max_keep,
        metric_name="accuracy",
        metric_mode="max",
    )

    # Handle resume
    start_epoch = 0
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        checkpoint = resume_training(
            args.resume_from,
            model=None,  # We'll create model in trainable()
            restore_rng=True,
        )
        start_epoch = checkpoint.epoch + 1
        print(f"Resuming from epoch {start_epoch}")
        # Note: Full resume with model loading happens in trainable()
    elif args.resume and checkpoint_manager.has_checkpoint():
        checkpoint = checkpoint_manager.load_latest()
        if checkpoint:
            start_epoch = checkpoint.epoch + 1
            print(f"Auto-resuming from epoch {start_epoch}")

    # Get trainer class
    trainer_class = get_trainer(config.trainer.name)

    # Run training
    trainable(
        config=config,
        trainer_class=trainer_class,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        start_epoch=start_epoch,
    )


###############
# ENTRY POINT #
###############

if __name__ == "__main__":
    main()
