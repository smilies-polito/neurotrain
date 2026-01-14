"""
Configuration system for SNN Training Benchmarking.

Supports YAML and JSON config files with CLI override capability.
Uses dataclasses for typed, validated configuration.
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ExperimentConfig:
    """Experiment identification and reproducibility settings."""

    name: str = "experiment"
    seed: int = 42
    deterministic: bool = True
    log_dir: str = "./experiments"


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""

    architecture: str = "fc"  # "fc", "conv", "recurrent"
    layer_sizes: List[int] = field(default_factory=lambda: [784, 200, 10])
    beta: float = 0.9375
    threshold: float = 1.0
    quantization: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.01
    optimizer: Optional[str] = None  # None for manual updates, "adam", "sgd"
    weight_decay: float = 0.0


@dataclass
class TrainerConfig:
    """Trainer-specific configuration."""

    name: str = "stsf"  # "stsf", "bptt", "decolle", "eprop", "drtp", "stdp"
    update_last: bool = False
    update_every: int = 1
    seq_batch: int = 1


@dataclass
class DRTPConfig:
    """Direct Random Target Projection configuration."""

    feedback_distribution: str = "kaiming_uniform"  # "kaiming_uniform", "uniform", "normal"
    feedback_scale: float = 1.0
    fixed_feedback: bool = True


@dataclass
class DataConfig:
    """Dataset configuration."""

    dataset: str = "MNIST"
    timesteps: int = 10
    data_dir: str = "./src/Data"
    num_workers: int = 4


@dataclass
class HardwareConfig:
    """Hardware and device configuration."""

    device: str = "auto"  # "auto", "cuda", "cuda:0", "cpu", "mps"
    mixed_precision: bool = False


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    save_every: int = 0  # 0 = only best and latest
    save_best: bool = True
    save_latest: bool = True
    max_keep: int = 2  # Maximum checkpoints to keep (0 = keep all)


@dataclass
class Config:
    """
    Complete experiment configuration.

    Combines all sub-configurations into a single config object.
    """

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    drtp: DRTPConfig = field(default_factory=DRTPConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to nested dictionary."""
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert config to flat dictionary (for logging)."""
        flat = {}
        for section_name, section in asdict(self).items():
            for key, value in section.items():
                flat[f"{section_name}.{key}"] = value
        return flat

    def save(self, path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save configuration to file.

        Args:
            path: Output file path
            format: "yaml" or "json"
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        if format == "yaml":
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'yaml' or 'json'.")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create Config from nested dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        return cls(
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            trainer=TrainerConfig(**config_dict.get("trainer", {})),
            drtp=DRTPConfig(**config_dict.get("drtp", {})),
            data=DataConfig(**config_dict.get("data", {})),
            hardware=HardwareConfig(**config_dict.get("hardware", {})),
            checkpoint=CheckpointConfig(**config_dict.get("checkpoint", {})),
        )


def load_config(path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML or JSON file.

    Args:
        path: Path to config file

    Returns:
        Config instance
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        if path.suffix in [".yaml", ".yml"]:
            config_dict = yaml.safe_load(f)
        elif path.suffix == ".json":
            config_dict = json.load(f)
        else:
            raise ValueError(
                f"Unknown config format: {path.suffix}. Use .yaml, .yml, or .json"
            )

    return Config.from_dict(config_dict or {})


def merge_config_with_args(config: Config, args) -> Config:
    """
    Merge config with CLI arguments (CLI takes precedence).

    Args:
        config: Base configuration
        args: Parsed CLI arguments (argparse Namespace)

    Returns:
        Updated Config with CLI overrides applied
    """
    config_dict = config.to_dict()

    def _flag_passed(flag_name: str) -> bool:
        """
        Return True if a given CLI flag (long form) was explicitly provided.
        We only override config values when the user actually passed the flag.
        """
        flag = f"--{flag_name.replace('_', '-')}"
        for raw in sys.argv[1:]:
            if raw.split("=")[0] == flag:
                return True
        return False

    # Map CLI args to config sections
    cli_mappings = {
        # experiment
        "exp_name": ("experiment", "name"),
        "seed": ("experiment", "seed"),
        # model
        "layer_size": ("model", "layer_sizes"),  # needs special handling
        "n_layers": ("model", "layer_sizes"),  # needs special handling
        "beta": ("model", "beta"),
        "threshold": ("model", "threshold"),
        "quantization": ("model", "quantization"),
        # training
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "lr": ("training", "learning_rate"),
        # Note: "optimizer" handled specially below (it's a boolean flag)
        # trainer
        "update_last": ("trainer", "update_last"),
        "update_every": ("trainer", "update_every"),
        "seq_batch": ("trainer", "seq_batch"),
        # data
        "dataset": ("data", "dataset"),
        "T": ("data", "timesteps"),
    }

    # Apply overrides from CLI
    for cli_arg, (section, key) in cli_mappings.items():
        if not hasattr(args, cli_arg):
            continue
        value = getattr(args, cli_arg)

        # When a config file is provided, only override if the user explicitly passed the flag
        if args.config and not _flag_passed(cli_arg):
            continue

        # Avoid overriding experiment name with legacy defaults unless explicitly set
        if cli_arg == "exp_name" and args.config:
            if value and not str(value).startswith("STSF_"):
                config_dict[section][key] = value
            continue

        if value is not None:
            config_dict[section][key] = value

    # Special handling for layer_sizes construction (respect config unless flags provided)
    if hasattr(args, "in_size") and hasattr(args, "n_class"):
        # Only rebuild layer_sizes if:
        # - no config file is used, or
        # - the user explicitly passed --n-layers or --layer-size
        rebuild_layers = False
        if not args.config:
            rebuild_layers = True
        else:
            if _flag_passed("n_layers") or _flag_passed("layer_size"):
                rebuild_layers = True
        if rebuild_layers:
            n_layers = getattr(args, "n_layers", 1)
            layer_size = getattr(args, "layer_size", 200)
            config_dict["model"]["layer_sizes"] = (
                [args.in_size] + [layer_size] * n_layers + [args.n_class]
            )

    # Handle optimizer flag (only override if explicitly set to True)
    # When --optimizer is passed, use adam; otherwise keep config file value
    if hasattr(args, "optimizer") and args.optimizer is True:
        config_dict["training"]["optimizer"] = "adam"

    return Config.from_dict(config_dict)


def create_default_config() -> Config:
    """Create a default configuration."""
    return Config()


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.

    Args:
        config: Configuration to validate

    Returns:
        List of warning/error messages (empty if valid)
    """
    issues = []

    # Model validation
    if len(config.model.layer_sizes) < 2:
        issues.append("model.layer_sizes must have at least 2 elements (input, output)")

    if config.model.beta <= 0 or config.model.beta >= 1:
        issues.append("model.beta should be in (0, 1)")

    # Training validation
    if config.training.epochs <= 0:
        issues.append("training.epochs must be positive")

    if config.training.batch_size <= 0:
        issues.append("training.batch_size must be positive")

    if config.training.learning_rate <= 0:
        issues.append("training.learning_rate must be positive")

    # Data validation
    if config.data.timesteps <= 0:
        issues.append("data.timesteps must be positive")

    valid_datasets = [
        # Rate-coded image classification
        "MNIST", "CIFAR10", "FashionMNIST", "SVHN",
        # Event-based neuromorphic (ideal for DECOLLE)
        "NMNIST", "DVSGesture",
        # NeuroBench official benchmarks
        "SpeechCommands", "WISDM",  # Classification
        "PrimateReaching", "MackeyGlass",  # Regression
    ]
    if config.data.dataset not in valid_datasets:
        issues.append(f"data.dataset must be one of {valid_datasets}")

    # Trainer validation
    valid_trainers = ["stsf", "bptt", "decolle", "eprop", "drtp", "stdp"]
    if config.trainer.name not in valid_trainers:
        issues.append(f"trainer.name must be one of {valid_trainers}")

    # DRTP validation
    valid_drtp_distributions = ["kaiming_uniform", "uniform", "normal"]
    if config.drtp.feedback_distribution not in valid_drtp_distributions:
        issues.append(
            f"drtp.feedback_distribution must be one of {valid_drtp_distributions}"
        )
    if config.drtp.feedback_scale <= 0:
        issues.append("drtp.feedback_scale must be positive")

    return issues


def print_config(config: Config) -> None:
    """Print configuration in a formatted way."""
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)

    for section_name, section in config.to_dict().items():
        print(f"\n[{section_name}]")
        for key, value in section.items():
            print(f"  {key}: {value}")

    print("=" * 60 + "\n")
