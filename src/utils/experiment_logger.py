"""
Experiment Logger - Comprehensive logging for reproducibility.

Captures and logs:
- All random seeds (torch, numpy, random, CUDA)
- Environment details (Python, PyTorch, device info)
- Git commit hash and repository state
- Full hyperparameters and configuration
- RNG states for exact reproducibility
"""

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class ExperimentContext:
    """Complete experiment context for reproducibility."""

    # Experiment identification
    experiment_name: str
    run_id: str
    timestamp: str

    # Seeds
    seed: int
    torch_seed: Optional[int] = None
    numpy_seed: Optional[int] = None
    random_seed: Optional[int] = None
    cuda_seed: Optional[int] = None

    # Environment
    python_version: str = ""
    torch_version: str = ""
    numpy_version: str = ""
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    device: str = ""
    hostname: str = ""
    platform_info: str = ""

    # Code versioning
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Paths
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None


def get_git_info() -> Dict[str, Any]:
    """Get git repository information for reproducibility."""
    git_info = {"commit": None, "branch": None, "dirty": False}

    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()

        # Get current branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["dirty"] = len(result.stdout.strip()) > 0

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # Git not available or not a git repository
        pass

    return git_info


def get_environment_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Collect environment information for reproducibility."""
    env_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "hostname": platform.node(),
        "platform_info": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["cudnn_version"] = str(torch.backends.cudnn.version())
        env_info["cuda_device_count"] = torch.cuda.device_count()
        if device is not None and device.type == "cuda":
            env_info["cuda_device_name"] = torch.cuda.get_device_name(device)
            env_info["cuda_device_capability"] = torch.cuda.get_device_capability(
                device
            )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        env_info["mps_available"] = True

    return env_info


def set_all_seeds(seed: int, deterministic: bool = True) -> Dict[str, int]:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Base seed value
        deterministic: If True, enables deterministic algorithms (may impact performance)

    Returns:
        Dictionary of all seeds that were set
    """
    seeds = {
        "base_seed": seed,
        "random_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
    }

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        seeds["cuda_seed"] = seed

        if deterministic:
            # Enable deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seeds


def get_rng_state() -> Dict[str, Any]:
    """
    Capture current RNG state for all random number generators.

    Returns:
        Dictionary containing RNG states that can be saved and restored
    """
    rng_state = {
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        rng_state["cuda_state"] = torch.cuda.get_rng_state_all()

    return rng_state


def set_rng_state(rng_state: Dict[str, Any]) -> None:
    """
    Restore RNG state from a previously captured state.

    Args:
        rng_state: Dictionary containing RNG states
    """
    if "random_state" in rng_state:
        random.setstate(rng_state["random_state"])

    if "numpy_state" in rng_state:
        np.random.set_state(rng_state["numpy_state"])

    if "torch_state" in rng_state:
        torch.set_rng_state(rng_state["torch_state"])

    if "cuda_state" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda_state"])


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ExperimentLogger:
    """
    Comprehensive experiment logger for tracking reproducibility.

    Usage:
        logger = ExperimentLogger(
            experiment_name="my_experiment",
            config={"lr": 0.01, "epochs": 100},
            seed=42
        )
        logger.setup()  # Sets seeds and captures environment
        logger.save_context()  # Save experiment context to disk

        # During training...
        logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)

        # Save checkpoint
        logger.save_checkpoint(model, optimizer, epoch, metrics)
    """

    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        seed: int = 42,
        log_dir: Optional[str] = None,
        deterministic: bool = True,
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary (hyperparameters, etc.)
            seed: Random seed for reproducibility
            log_dir: Directory for logs and checkpoints
            deterministic: Enable deterministic mode for exact reproducibility
        """
        self.experiment_name = experiment_name
        self.config = config
        self.seed = seed
        self.deterministic = deterministic
        self.run_id = generate_run_id()

        # Set up directories
        if log_dir is None:
            log_dir = Path("./experiments") / experiment_name / self.run_id
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = self.log_dir / "checkpoints"

        self.context: Optional[ExperimentContext] = None
        self.device: Optional[torch.device] = None
        self._tensorboard_writer = None

    def setup(self, device: Optional[torch.device] = None) -> ExperimentContext:
        """
        Set up the experiment: set seeds, capture environment, create directories.

        Args:
            device: PyTorch device being used

        Returns:
            ExperimentContext with all captured information
        """
        self.device = device

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set all seeds
        seeds = set_all_seeds(self.seed, self.deterministic)

        # Capture environment info
        env_info = get_environment_info(device)

        # Capture git info
        git_info = get_git_info()

        # Create experiment context
        self.context = ExperimentContext(
            experiment_name=self.experiment_name,
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            seed=self.seed,
            torch_seed=seeds.get("torch_seed"),
            numpy_seed=seeds.get("numpy_seed"),
            random_seed=seeds.get("random_seed"),
            cuda_seed=seeds.get("cuda_seed"),
            python_version=env_info["python_version"],
            torch_version=env_info["torch_version"],
            numpy_version=env_info["numpy_version"],
            cuda_version=env_info.get("cuda_version"),
            cudnn_version=env_info.get("cudnn_version"),
            device=str(device) if device else "cpu",
            hostname=env_info["hostname"],
            platform_info=env_info["platform_info"],
            git_commit=git_info["commit"],
            git_branch=git_info["branch"],
            git_dirty=git_info["dirty"],
            config=self.config,
            hyperparameters=self.config,
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
        )

        return self.context

    def save_context(self) -> Path:
        """
        Save experiment context to a JSON file.

        Returns:
            Path to the saved context file
        """
        if self.context is None:
            raise RuntimeError("Call setup() before save_context()")

        context_path = self.log_dir / "experiment_context.json"
        context_dict = asdict(self.context)

        with open(context_path, "w") as f:
            json.dump(context_dict, f, indent=2, default=str)

        print(f"Experiment context saved to: {context_path}")
        return context_path

    @staticmethod
    def load_context(context_path: str) -> ExperimentContext:
        """
        Load experiment context from a JSON file.

        Args:
            context_path: Path to the context file

        Returns:
            ExperimentContext loaded from file
        """
        with open(context_path, "r") as f:
            context_dict = json.load(f)

        return ExperimentContext(**context_dict)

    def get_tensorboard_writer(self):
        """Get or create TensorBoard SummaryWriter."""
        if self._tensorboard_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tensorboard_writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")
                return None
        return self._tensorboard_writer

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """
        Log metrics to TensorBoard and/or file.

        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch
            prefix: Optional prefix for metric names
        """
        writer = self.get_tensorboard_writer()
        if writer is not None:
            for name, value in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                writer.add_scalar(tag, value, step)

    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters with associated metrics to TensorBoard."""
        writer = self.get_tensorboard_writer()
        if writer is not None:
            # Sanitize hparams: TensorBoard only accepts int, float, str, bool, or Tensor
            sanitized_hparams = {}
            for key, value in hparams.items():
                if value is None:
                    sanitized_hparams[key] = "None"
                elif isinstance(value, (list, tuple)):
                    sanitized_hparams[key] = str(value)
                elif isinstance(value, (int, float, str, bool)):
                    sanitized_hparams[key] = value
                else:
                    sanitized_hparams[key] = str(value)
            writer.add_hparams(sanitized_hparams, metrics)

    def close(self):
        """Close any open resources."""
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()


def print_experiment_info(context: ExperimentContext) -> None:
    """Print experiment information in a formatted way."""
    print("\n" + "=" * 60)
    print("EXPERIMENT INFORMATION")
    print("=" * 60)
    print(f"Experiment: {context.experiment_name}")
    print(f"Run ID: {context.run_id}")
    print(f"Timestamp: {context.timestamp}")
    print("-" * 60)
    print("SEEDS:")
    print(f"  Base seed: {context.seed}")
    print(f"  CUDA seed: {context.cuda_seed}")
    print("-" * 60)
    print("ENVIRONMENT:")
    print(f"  Python: {context.python_version.split()[0]}")
    print(f"  PyTorch: {context.torch_version}")
    print(f"  Device: {context.device}")
    if context.cuda_version:
        print(f"  CUDA: {context.cuda_version}")
    print("-" * 60)
    print("CODE VERSION:")
    if context.git_commit:
        print(f"  Git commit: {context.git_commit[:8]}")
        print(f"  Git branch: {context.git_branch}")
        print(f"  Uncommitted changes: {context.git_dirty}")
    else:
        print("  Git info: Not available")
    print("-" * 60)
    print(f"Log directory: {context.log_dir}")
    print("=" * 60 + "\n")

