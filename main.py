"""
SNN Training Benchmarking - Main Entry Point.

Supports:
- YAML/JSON configuration files
- CLI argument overrides
- Checkpointing and resume
- Reproducibility logging
- TensorBoard integration
"""

import os
import statistics
import sys
from collections import deque
from pathlib import Path

import torch
import torch.optim as optim

# Add the src folder to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Core imports
from datasets.get_loader import get_loader
from LearningAlgorithms import LearningAlgorithms
from networks.conv_network import ConvFCNetwork
from networks.get_network import get_network
from trainers.bptt_trainer import BPTTTrainer
from trainers.decolle_trainer import DECOLLETrainer
from trainers.drtp_trainer import DRTPTrainer
from trainers.ell_trainer import ELLTrainer
from trainers.eprop_trainer import EpropTrainer
from trainers.etlp_trainer import ETLPTrainer
from trainers.fell_trainer import FELLTrainer
from trainers.bell_trainer import BELLTrainer
from trainers.ottt_trainer import OTTTTrainer
from trainers.stsf_trainer import STSFTrainer
from trainers.esd_rtrl_trainer import ESDRTRLTrainer
from utils.checkpoint import CheckpointManager, set_rng_state
from utils.config import (
    Config,
    load_config,
    merge_config_with_args,
    print_config,
    validate_config,
)
from utils.experiment_logger import (
    ExperimentLogger,
    print_experiment_info,
    set_all_seeds,
)
from utils.parameters import parse_args

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


def trainable(
    config: Config,
    trainer_class,
    logger: ExperimentLogger,
    checkpoint_manager: CheckpointManager,
    start_epoch: int = 0,
    resume_checkpoint=None,
):
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
    flatten_inputs = config.model.architecture != "conv"
    trainloader, testloader = get_loader(
        config.data.dataset,
        config.training.batch_size,
        config.data.timesteps,
        flatten=flatten_inputs,
        device=device,
    )

    # Create the network (supports fc and recurrent architectures)
    if config.model.architecture == "recurrent":
        from networks.recurrent_srnn import RecurrentSRNN

        # Match original e-prop defaults for comparison
        n_in = config.model.layer_sizes[0]
        n_rec = (
            config.model.layer_sizes[1]
            if len(config.model.layer_sizes) > 2
            else config.model.layer_sizes[1]
            if len(config.model.layer_sizes) > 1
            else 100
        )
        n_out = config.model.layer_sizes[-1]
        network = RecurrentSRNN(
            n_in=n_in,
            n_rec=n_rec,
            n_out=n_out,
            threshold=config.model.threshold,
            tau_mem=2.0,
            tau_out=0.02,
            bias_out=0.0,
            gamma=0.3,
            dt=1e-3,
        )
        # Attach for compatibility
        network.n_classes = n_out
        network.hidden_size = [n_rec]
    elif config.model.architecture == "conv":
        if config.data.dataset == "MNIST":
            input_shape = (1, 28, 28)
        elif config.data.dataset == "CIFAR10":
            input_shape = (3, 32, 32)
        else:
            raise ValueError(
                "Conv architecture currently supports MNIST and CIFAR10 only."
            )

        network = ConvFCNetwork(
            input_shape=input_shape,
            conv_layers=config.model.conv_layers,
            layer_sizes=config.model.layer_sizes,
            beta=config.model.beta,
            threshold=config.model.threshold,
            quant=config.model.quantization,
        )
    else:
        network = FCNetwork(
            layer_sizes=config.model.layer_sizes,
            beta=config.model.beta,
            quant=config.model.quantization,
            threshold=config.model.threshold,
        )

    if config.training.freeze_conv:
        for module in network.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

    # Optimizer (BPTT only; ELL/FELL/BELL use per-layer optimizers in trainer)
    optimizer = None
    if config.training.optimizer is not None:
        optimizer_name = str(config.training.optimizer).lower()
        if optimizer_name == "adam" and config.trainer.name == "bptt":
            optimizer = optim.Adam(
                network.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(
                network.parameters(),
                lr=config.training.learning_rate,
                momentum=0.9,
                weight_decay=config.training.weight_decay,
                nesterov=False,
            )
        elif optimizer_name == "nag":
            optimizer = optim.SGD(
                network.parameters(),
                lr=config.training.learning_rate,
                momentum=0.9,
                weight_decay=config.training.weight_decay,
                nesterov=True,
            )
        elif optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(
                network.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer '{config.training.optimizer}'. "
                "Use one of: adam, sgd, nag, rmsprop, or null."
            )

    # Create the trainer
    # Enable gradients for BPTT and ELL/FELL/BELL; disable for STSF/DECOLLE/OTTT/E-prop
    requires_grad = config.trainer.name in ("bptt", "ell", "fell", "bell")
    torch.set_grad_enabled(requires_grad)
    trainer_kwargs = {
        "network": network,
        "lr": config.training.learning_rate,
        "batch_size": config.training.batch_size,
        "quant": config.model.quantization,
        "use_optimizer": config.training.optimizer is not None,
        "optimizer": optimizer,
    }

    if issubclass(trainer_class, STSFTrainer):
        trainer_kwargs.update(
            update_last=config.trainer.update_last,
            update_every=config.trainer.update_every,
            seq_batch_size=config.trainer.seq_batch,
        )
    if issubclass(trainer_class, DRTPTrainer):
        trainer_kwargs.update(
            feedback_distribution=config.drtp.feedback_distribution,
            feedback_scale=config.drtp.feedback_scale,
            fixed_feedback=config.drtp.fixed_feedback,
            loss_type=config.drtp.loss,
            freeze_conv=config.training.freeze_conv,
            update_last=config.trainer.update_last,
            update_every=config.trainer.update_every,
        )
    if issubclass(trainer_class, ETLPTrainer):
        trainer_kwargs.update(
            trace_decay=config.etlp.trace_decay,
            surrogate_scale=config.etlp.surrogate_scale,
            voltage_reg=config.etlp.voltage_reg,
            weight_l1=config.etlp.weight_l1,
            weight_l2=config.etlp.weight_l2,
            update_rate_hz=config.etlp.update_rate_hz,
            dt_ms=config.etlp.dt_ms,
            feedback_distribution=config.etlp.feedback_distribution,
            feedback_scale=config.etlp.feedback_scale,
            update_last=config.trainer.update_last,
            update_every=config.trainer.update_every,
        )

    trainer = trainer_class(**trainer_kwargs).to(device)

    if resume_checkpoint is not None:
        trainer.network.load_state_dict(resume_checkpoint.model_state_dict)
        if optimizer is not None and resume_checkpoint.optimizer_state_dict is not None:
            optimizer.load_state_dict(resume_checkpoint.optimizer_state_dict)
        if (
            hasattr(trainer, "load_checkpoint_state")
            and resume_checkpoint.trainer_state_dict
        ):
            trainer.load_checkpoint_state(resume_checkpoint.trainer_state_dict)
        start_epoch = max(start_epoch, resume_checkpoint.epoch + 1)

    trainer.network.train()

    # Setup graceful exit (save checkpoint on Ctrl+C)
    checkpoint_manager.setup_graceful_exit(trainer.network, optimizer, config.to_dict())

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
        print(
            {
                "epoch": epoch + 1,
                "testing_accuracy": testing_accuracy,
                "training_accuracy": training_accuracy,
                "training_loss": training_loss,
                "test_acc_std_last5": std_last5,
                "test_acc_delta": delta,
            }
        )
        # Save checkpoint if needed
        trainer_state = None
        if hasattr(trainer, "checkpoint_state"):
            trainer_state = trainer.checkpoint_state()
        checkpoint_manager.save_if_needed(
            model=trainer.network,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config.to_dict(),
            trainer_state_dict=trainer_state,
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
        "eprop": EpropTrainer,
        "decolle": DECOLLETrainer,
        "ottt": OTTTTrainer,
        "drtp": DRTPTrainer,
        "etlp": ETLPTrainer,
        "ell": ELLTrainer,
        "fell": FELLTrainer,
        "bell": BELLTrainer,
        "esd_rtrl": ESDRTRLTrainer,
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
    resume_checkpoint = None
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        resume_checkpoint = checkpoint_manager.load(args.resume_from)
        if resume_checkpoint.rng_state:
            set_rng_state(resume_checkpoint.rng_state)
            print("RNG state restored for exact reproducibility")
        start_epoch = resume_checkpoint.epoch + 1
        print(f"Resuming from epoch {start_epoch}")
    elif args.resume and checkpoint_manager.has_checkpoint():
        resume_checkpoint = checkpoint_manager.load_latest()
        if resume_checkpoint:
            if resume_checkpoint.rng_state:
                set_rng_state(resume_checkpoint.rng_state)
                print("RNG state restored for exact reproducibility")
            start_epoch = resume_checkpoint.epoch + 1
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
        resume_checkpoint=resume_checkpoint,
    )


###############
# ENTRY POINT #
###############

if __name__ == "__main__":
    main()
