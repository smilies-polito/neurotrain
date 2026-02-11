"""
Checkpointing system for SNN Training Benchmarking.

Provides save/load functionality for:
- Model state
- Optimizer state
- Training state (epoch, metrics)
- RNG states (for exact reproducibility)
- Configuration
"""

import multiprocessing as mp
import random
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CheckpointData:
    """Container for all checkpoint data."""

    # Model and optimizer
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Optional[Dict[str, Any]]
    trainer_state_dict: Optional[Dict[str, Any]]

    # Training state
    epoch: int
    global_step: int
    best_metric: float
    best_epoch: int

    # Metrics history
    metrics_history: Dict[str, list]

    # RNG states for reproducibility
    rng_state: Dict[str, Any]

    # Configuration
    config: Dict[str, Any]

    # Metadata
    checkpoint_path: Optional[str] = None


def get_rng_state() -> Dict[str, Any]:
    """
    Capture current RNG state from all random number generators.

    Returns:
        Dictionary containing RNG states
    """
    rng_state = {
        "python_random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    # Guard against CUDA lazy-init in forked worker processes.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()

    return rng_state


def set_rng_state(rng_state: Dict[str, Any]) -> None:
    """
    Restore RNG state to all random number generators.

    Args:
        rng_state: Dictionary containing RNG states
    """
    if "python_random" in rng_state:
        random.setstate(rng_state["python_random"])

    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])

    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])

    if "cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])


class CheckpointManager:
    """
    Manages checkpoint saving and loading for experiments.

    Features:
    - Automatic saving of best and latest checkpoints
    - Configurable checkpoint frequency
    - RNG state preservation for exact reproducibility
    - Graceful interruption handling (saves on Ctrl+C)

    Usage:
        manager = CheckpointManager(
            checkpoint_dir="./experiments/run_001/checkpoints",
            save_best=True,
            save_latest=True,
        )

        # Setup graceful interruption
        manager.setup_graceful_exit(model, optimizer, config)

        # During training
        for epoch in range(epochs):
            train(...)
            metrics = evaluate(...)

            manager.save_if_needed(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"accuracy": acc, "loss": loss},
                config=config,
            )

        # Resume from checkpoint
        checkpoint = manager.load_latest()
        model.load_state_dict(checkpoint.model_state_dict)
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_best: bool = True,
        save_latest: bool = True,
        save_every: int = 0,
        max_keep: int = 2,
        metric_name: str = "accuracy",
        metric_mode: str = "max",
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best checkpoint
            save_latest: Whether to save latest checkpoint
            save_every: Save every N epochs (0 = disabled)
            max_keep: Maximum periodic checkpoints to keep (0 = keep all)
            metric_name: Name of metric to track for best checkpoint
            metric_mode: "max" or "min" - whether higher or lower is better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_best = save_best
        self.save_latest = save_latest
        self.save_every = save_every
        self.max_keep = max_keep
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        # Tracking
        self.best_metric = float("-inf") if metric_mode == "max" else float("inf")
        self.best_epoch = -1
        self.global_step = 0
        self.metrics_history: Dict[str, list] = {}

        # Graceful exit handling
        self._model_ref = None
        self._optimizer_ref = None
        self._config_ref = None
        self._current_epoch = 0
        self._original_sigint_handler = None

    def _is_better(self, metric: float) -> bool:
        """Check if metric is better than current best."""
        if self.metric_mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        filename: str,
        trainer_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer (can be None)
            epoch: Current epoch number
            metrics: Dictionary of current metrics
            config: Configuration dictionary
            filename: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "trainer_state_dict": trainer_state_dict,
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "metrics": metrics,
            "metrics_history": self.metrics_history,
            "rng_state": get_rng_state(),
            "config": config,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def save_if_needed(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        trainer_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Save checkpoint if conditions are met (best, latest, or periodic).

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch number
            metrics: Dictionary of current metrics
            config: Configuration dictionary

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        self._current_epoch = epoch
        saved_path = None

        # Update metrics history
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)

        # Check if this is the best
        current_metric = metrics.get(self.metric_name, 0)
        is_best = self._is_better(current_metric)

        if is_best:
            self.best_metric = current_metric
            self.best_epoch = epoch

        # Save best checkpoint
        if self.save_best and is_best:
            saved_path = self.save(
                model,
                optimizer,
                epoch,
                metrics,
                config,
                "checkpoint_best.pt",
                trainer_state_dict=trainer_state_dict,
            )

        # Save latest checkpoint
        if self.save_latest:
            saved_path = self.save(
                model,
                optimizer,
                epoch,
                metrics,
                config,
                "checkpoint_latest.pt",
                trainer_state_dict=trainer_state_dict,
            )

        # Save periodic checkpoint
        if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
            saved_path = self.save(
                model,
                optimizer,
                epoch,
                metrics,
                config,
                f"checkpoint_epoch_{epoch}.pt",
                trainer_state_dict=trainer_state_dict,
            )
            self._cleanup_old_checkpoints()

        return saved_path

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old periodic checkpoints if max_keep is set."""
        if self.max_keep <= 0:
            return

        # Find periodic checkpoints (not best or latest)
        periodic_checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        # Remove oldest checkpoints beyond max_keep
        while len(periodic_checkpoints) > self.max_keep:
            oldest = periodic_checkpoints.pop(0)
            oldest.unlink()
            print(f"Removed old checkpoint: {oldest}")

    def load(self, checkpoint_path: Union[str, Path]) -> CheckpointData:
        """
        Load a checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            CheckpointData containing all checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # weights_only=False needed for loading RNG states (numpy arrays)
        # This is safe as we're loading our own checkpoints
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Update internal state
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.best_epoch = checkpoint.get("best_epoch", -1)
        self.global_step = checkpoint.get("global_step", 0)
        self.metrics_history = checkpoint.get("metrics_history", {})

        return CheckpointData(
            model_state_dict=checkpoint["model_state_dict"],
            optimizer_state_dict=checkpoint.get("optimizer_state_dict"),
            trainer_state_dict=checkpoint.get("trainer_state_dict"),
            epoch=checkpoint["epoch"],
            global_step=checkpoint.get("global_step", 0),
            best_metric=checkpoint.get("best_metric", 0),
            best_epoch=checkpoint.get("best_epoch", -1),
            metrics_history=checkpoint.get("metrics_history", {}),
            rng_state=checkpoint.get("rng_state", {}),
            config=checkpoint.get("config", {}),
            checkpoint_path=str(checkpoint_path),
        )

    def load_latest(self) -> Optional[CheckpointData]:
        """Load the latest checkpoint if it exists."""
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest_path.exists():
            return self.load(latest_path)
        return None

    def load_best(self) -> Optional[CheckpointData]:
        """Load the best checkpoint if it exists."""
        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        if best_path.exists():
            return self.load(best_path)
        return None

    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        return (self.checkpoint_dir / "checkpoint_latest.pt").exists() or (
            self.checkpoint_dir / "checkpoint_best.pt"
        ).exists()

    def setup_graceful_exit(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        config: Dict[str, Any],
    ) -> None:
        """
        Setup graceful exit handling to save checkpoint on Ctrl+C.

        Args:
            model: PyTorch model to save
            optimizer: PyTorch optimizer to save
            config: Configuration to save
        """
        self._model_ref = model
        self._optimizer_ref = optimizer
        self._config_ref = config

        def signal_handler(signum, frame):
            # DataLoader workers inherit handlers on fork; never checkpoint there.
            if mp.current_process().name != "MainProcess":
                sys.exit(130)

            print("\n\nInterrupted! Saving checkpoint before exit...")
            if self._model_ref is not None:
                try:
                    self.save(
                        self._model_ref,
                        self._optimizer_ref,
                        self._current_epoch,
                        {},
                        self._config_ref,
                        "checkpoint_interrupted.pt",
                    )
                except Exception as exc:  # pragma: no cover - interruption path
                    print(f"Checkpoint save on interrupt failed: {exc}")
            print("Checkpoint saved. Exiting.")

            # Restore original handler and re-raise
            if self._original_sigint_handler:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            sys.exit(130)

        self._original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)


def resume_training(
    checkpoint_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    trainer: Optional[nn.Module] = None,
    restore_rng: bool = True,
) -> CheckpointData:
    """
    Resume training from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to restore (optional)
        optimizer: PyTorch optimizer to restore (optional)
        trainer: Trainer instance to restore (optional)
        restore_rng: Whether to restore RNG state

    Returns:
        CheckpointData with checkpoint information
    """
    manager = CheckpointManager(Path(checkpoint_path).parent)
    checkpoint = manager.load(checkpoint_path)

    # Restore model state
    if model is not None:
        model.load_state_dict(checkpoint.model_state_dict)

    # Restore optimizer state
    if optimizer is not None and checkpoint.optimizer_state_dict is not None:
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

    # Restore trainer state (if provided)
    if trainer is not None and checkpoint.trainer_state_dict is not None:
        if hasattr(trainer, "load_checkpoint_state"):
            trainer.load_checkpoint_state(checkpoint.trainer_state_dict)
        else:
            trainer.load_state_dict(checkpoint.trainer_state_dict, strict=False)

    # Restore RNG state for exact reproducibility
    if restore_rng and checkpoint.rng_state:
        set_rng_state(checkpoint.rng_state)
        print("RNG state restored for exact reproducibility")

    print(f"Resumed from epoch {checkpoint.epoch}")
    print(
        f"Best {manager.metric_name}: {checkpoint.best_metric:.4f} (epoch {checkpoint.best_epoch})"
    )

    return checkpoint
