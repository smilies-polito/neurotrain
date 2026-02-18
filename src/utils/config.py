"""
Configuration system for SNN Training Benchmarking.

Supports YAML and JSON config files with CLI override capability.
Uses dataclasses for typed, validated configuration.
"""

import json
import sys
from dataclasses import asdict, dataclass, field
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

    architecture: str = "fc"  # "fc", "fc_snn", "r_snn", "conv_snn", "vg11_snn", "local_classifier", "recurrent", "ottt_conv_net"
    layer_sizes: List[int] = field(default_factory=lambda: [784, 200, 10])
    conv_layers: List[Dict[str, int]] = field(default_factory=list)
    beta: float = 0.9375
    tau: Optional[float] = None  # for ELL/FELL/BELL: decay = exp(-1/tau) when set
    threshold: float = 1.0
    recurrent_type: str = "standard"  # "standard"/"srnn", "snu"/"ssnu"
    quantization: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.01
    optimizer: Optional[str] = (
        None  # None for manual updates, "adam", "sgd", "nag", "rmsprop"
    )
    weight_decay: float = 0.0
    freeze_conv: bool = False


@dataclass
class TrainerConfig:
    """Trainer-specific configuration."""

    name: str = "stsf"  # "stsf", "bptt", "decolle", "eprop", "drtp", "etlp", "ostl", "osttp", "stdp"
    update_last: bool = False
    update_every: int = 1
    seq_batch: int = 1


@dataclass
class DRTPConfig:
    """Direct Random Target Projection configuration."""

    loss: str = "mse"  # "mse", "bce"
    output_mode: str = "mem"  # "mem", "spike"
    # If true, enforce paper-specific topology checks (e.g., MNIST conv setup).
    paper_reproduction: bool = False
    surrogate_scale: float = 5.0
    surrogate_type: str = "logistic"
    feedback_distribution: str = (
        "kaiming_uniform"  # "kaiming_uniform", "uniform", "normal"
    )
    feedback_scale: float = 1.0
    fixed_feedback: bool = True


@dataclass
class ETLPConfig:
    """Event-based Three-factor Local Plasticity (ETLP) configuration."""

    trace_decay: float = 0.9
    surrogate_scale: float = 0.3
    voltage_reg: float = 0.0
    weight_l1: float = 0.0
    weight_l2: float = 0.0
    update_rate_hz: float = 100.0
    dt_ms: float = 1.0
    feedback_distribution: str = (
        "kaiming_uniform"  # "kaiming_uniform", "uniform", "normal"
    )
    feedback_scale: float = 1.0


@dataclass
class OSTLConfig:
    """Online Spatio-Temporal Learning (OSTL) configuration."""

    surrogate_scale: float = 5.0
    grad_clip: float = 0.0
    output_mode: str = "spike"  # "spike", "mem"


@dataclass
class STOPConfig:
    """STOP (SpatioTemporal Orthogonal Propagation) configuration."""

    loss: str = "ce"  # "ce", "mse"
    surrogate: str = "exp"  # "exp", "rational"
    learn_weights: bool = True
    learn_thresholds: bool = True
    learn_leakage: bool = True
    lr_weight: Optional[float] = None
    lr_threshold: Optional[float] = None
    lr_leakage: Optional[float] = None
    threshold_min: float = 1e-3
    momentum: float = 0.0
    cosine_schedule: bool = False
    cosine_t_max: int = 0
    static_input_timesteps: int = 1


@dataclass
class OSTTPConfig:
    """OSTTP (Online Spatio-Temporal Learning with Target Projection)."""

    pseudo_derivative: str = "tanh"  # "tanh", "fast_sigmoid"
    output_loss: str = "ce"  # "ce", "mse", "bce_logits", "bce_probs" (or "bce" alias)
    output_readout: str = "mem"  # "spk", "mem", "logits", "probs"
    feedback_scale: float = 1.0
    feedback_seed: int = 42
    target_dim: Optional[int] = None
    grad_clip: float = 0.0
    debug: bool = False


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
    etlp: ETLPConfig = field(default_factory=ETLPConfig)
    ostl: OSTLConfig = field(default_factory=OSTLConfig)
    stop: STOPConfig = field(default_factory=STOPConfig)
    osttp: OSTTPConfig = field(default_factory=OSTTPConfig)
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
            etlp=ETLPConfig(**config_dict.get("etlp", {})),
            ostl=OSTLConfig(**config_dict.get("ostl", {})),
            stop=STOPConfig(**config_dict.get("stop", {})),
            osttp=OSTTPConfig(**config_dict.get("osttp", {})),
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

    if config.model.architecture == "conv":
        if not config.model.conv_layers:
            issues.append("model.conv_layers must be provided for conv architecture")
        else:
            required_keys = {"out_channels", "kernel_size"}
            for idx, layer in enumerate(config.model.conv_layers):
                missing = required_keys - set(layer.keys())
                if missing:
                    issues.append(
                        f"model.conv_layers[{idx}] missing keys: {sorted(missing)}"
                    )

    # Training validation
    if config.training.epochs <= 0:
        issues.append("training.epochs must be positive")

    if config.training.batch_size <= 0:
        issues.append("training.batch_size must be positive")

    if config.training.learning_rate <= 0:
        issues.append("training.learning_rate must be positive")

    if config.training.optimizer is not None:
        valid_optimizers = ["adam", "sgd", "nag", "rmsprop"]
        if str(config.training.optimizer).lower() not in valid_optimizers:
            issues.append(
                f"training.optimizer must be one of {valid_optimizers} or null"
            )

    # Data validation
    if config.data.timesteps <= 0:
        issues.append("data.timesteps must be positive")

    valid_datasets = [
        # Rate-coded image classification
        "MNIST",
        "CIFAR10",
        "FashionMNIST",
        "SVHN",
        # Event-based neuromorphic (ideal for DECOLLE)
        "NMNIST",
        "DVSGesture",
        # NeuroBench official benchmarks
        "SpeechCommands",
        "WISDM",  # Classification
        "PrimateReaching",
        "MackeyGlass",  # Regression
    ]
    if config.data.dataset not in valid_datasets:
        issues.append(f"data.dataset must be one of {valid_datasets}")

    # Model architecture validation
    valid_architectures = [
        "fc",
        "fc_snn",
        "r_snn",
        "conv",
        "conv_snn",
        "local_classifier",
        "recurrent",
        "stllr",
        "vgg11",
        "vg11_snn",
        "resnet18",
        "ottt_conv_net",
    ]
    if config.model.architecture not in valid_architectures:
        issues.append(f"model.architecture must be one of {valid_architectures}")
    valid_recurrent_types = ["standard", "srnn", "snu", "ssnu"]
    if config.model.recurrent_type not in valid_recurrent_types:
        issues.append(f"model.recurrent_type must be one of {valid_recurrent_types}")

    if (
        config.trainer.name in ("eprop", "esd_rtrl")
        and config.model.architecture == "recurrent"
        and config.model.recurrent_type not in ("standard", "srnn")
    ):
        issues.append(
            "eprop/esd_rtrl require model.recurrent_type in ['standard', 'srnn']"
        )

    # Trainer validation
    # valid_trainers = ["stsf", "bptt", "decolle", "eprop", "drtp", "etlp", "stdp"]
    valid_trainers = [
        "stsf",
        "bptt",
        "decolle",
        "eprop",
        "ostl",
        "osttp",
        "ottt",
        "ell",
        "fell",
        "bell",
        "stllr",
        "esd_rtrl",
        "stdp",
        "tp",
        "etlp",
        "drtp",
        "stop",
    ]
    if config.trainer.name not in valid_trainers:
        issues.append(f"trainer.name must be one of {valid_trainers}")

    # DRTP validation
    valid_drtp_losses = ["mse", "bce"]
    loss_name = str(config.drtp.loss).lower()
    if loss_name not in valid_drtp_losses:
        issues.append(f"drtp.loss must be one of {valid_drtp_losses}")
    valid_drtp_output_modes = ["mem", "spike"]
    drtp_output_mode = str(config.drtp.output_mode).lower()
    if drtp_output_mode not in valid_drtp_output_modes:
        issues.append(f"drtp.output_mode must be one of {valid_drtp_output_modes}")
    if config.drtp.surrogate_scale <= 0:
        issues.append("drtp.surrogate_scale must be positive")
    valid_drtp_surrogates = ["logistic"]
    if str(config.drtp.surrogate_type).lower() not in valid_drtp_surrogates:
        issues.append(f"drtp.surrogate_type must be one of {valid_drtp_surrogates}")

    valid_drtp_distributions = ["kaiming_uniform", "uniform", "normal"]
    if config.drtp.feedback_distribution not in valid_drtp_distributions:
        issues.append(
            f"drtp.feedback_distribution must be one of {valid_drtp_distributions}"
        )
    if config.drtp.feedback_scale <= 0:
        issues.append("drtp.feedback_scale must be positive")

    # ETLP validation
    if config.etlp.trace_decay <= 0:
        issues.append("etlp.trace_decay must be positive")
    if config.etlp.surrogate_scale <= 0:
        issues.append("etlp.surrogate_scale must be positive")
    if config.etlp.dt_ms <= 0:
        issues.append("etlp.dt_ms must be positive")
    if config.etlp.update_rate_hz < 0:
        issues.append("etlp.update_rate_hz must be non-negative")
    valid_etlp_distributions = ["kaiming_uniform", "uniform", "normal"]
    if config.etlp.feedback_distribution not in valid_etlp_distributions:
        issues.append(
            f"etlp.feedback_distribution must be one of {valid_etlp_distributions}"
        )
    if config.etlp.feedback_scale <= 0:
        issues.append("etlp.feedback_scale must be positive")
    if config.trainer.name == "etlp" and config.model.architecture != "fc":
        issues.append("ETLP currently supports model.architecture == 'fc' only")

    # OSTL validation
    if config.ostl.surrogate_scale <= 0:
        issues.append("ostl.surrogate_scale must be positive")
    if config.ostl.grad_clip < 0:
        issues.append("ostl.grad_clip must be non-negative")
    if str(config.ostl.output_mode).lower() not in ("spike", "mem"):
        issues.append("ostl.output_mode must be one of ['spike', 'mem']")
    if config.trainer.name == "ostl":
        if config.model.architecture not in ("fc", "fc_snn"):
            issues.append(
                "OSTL currently supports model.architecture in {'fc', 'fc_snn'} only"
            )

    # OSTTP validation
    valid_pseudo = ["tanh", "fast_sigmoid"]
    if config.osttp.pseudo_derivative not in valid_pseudo:
        issues.append(f"osttp.pseudo_derivative must be one of {valid_pseudo}")
    valid_output_losses = ["ce", "mse", "bce", "bce_logits", "bce_probs"]
    if config.osttp.output_loss not in valid_output_losses:
        issues.append(f"osttp.output_loss must be one of {valid_output_losses}")
    valid_output_readouts = ["spk", "mem", "logits", "probs"]
    if config.osttp.output_readout not in valid_output_readouts:
        issues.append(f"osttp.output_readout must be one of {valid_output_readouts}")
    if (
        config.osttp.output_loss == "bce_logits"
        and config.osttp.output_readout != "logits"
    ):
        issues.append(
            "osttp.output_loss='bce_logits' requires osttp.output_readout='logits'"
        )
    if (
        config.osttp.output_loss == "bce_probs"
        and config.osttp.output_readout != "probs"
    ):
        issues.append(
            "osttp.output_loss='bce_probs' requires osttp.output_readout='probs'"
        )
    if config.osttp.feedback_scale <= 0:
        issues.append("osttp.feedback_scale must be positive")
    if config.osttp.grad_clip < 0:
        issues.append("osttp.grad_clip must be non-negative")
    if config.osttp.target_dim is not None and config.osttp.target_dim <= 0:
        issues.append("osttp.target_dim must be positive when provided")
    if config.trainer.name == "osttp" and config.model.architecture != "fc":
        issues.append("OSTTP currently supports model.architecture == 'fc' only")

    # STOP validation
    valid_stop_losses = ["ce", "mse"]
    if str(config.stop.loss).lower() not in valid_stop_losses:
        issues.append(f"stop.loss must be one of {valid_stop_losses}")

    valid_stop_surrogates = ["exp", "rational"]
    if str(config.stop.surrogate).lower() not in valid_stop_surrogates:
        issues.append(f"stop.surrogate must be one of {valid_stop_surrogates}")

    if not (
        config.stop.learn_weights
        or config.stop.learn_thresholds
        or config.stop.learn_leakage
    ):
        issues.append(
            "stop requires at least one of learn_weights / learn_thresholds / learn_leakage"
        )

    for key, value in (
        ("stop.lr_weight", config.stop.lr_weight),
        ("stop.lr_threshold", config.stop.lr_threshold),
        ("stop.lr_leakage", config.stop.lr_leakage),
    ):
        if value is not None and value <= 0:
            issues.append(f"{key} must be positive when set")

    if config.stop.threshold_min <= 0:
        issues.append("stop.threshold_min must be positive")
    if config.stop.momentum < 0 or config.stop.momentum >= 1:
        issues.append("stop.momentum must be in [0, 1)")
    if config.stop.static_input_timesteps <= 0:
        issues.append("stop.static_input_timesteps must be positive")
    if config.stop.cosine_schedule and config.stop.cosine_t_max <= 0:
        issues.append("stop.cosine_t_max must be > 0 when stop.cosine_schedule is true")
    if config.trainer.name == "stop" and config.model.architecture not in (
        "fc",
        "conv",
        "vgg11",
        "resnet18",
    ):
        issues.append(
            "STOP currently supports model.architecture in "
            "{'fc', 'conv', 'vgg11', 'resnet18'} only"
        )

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
