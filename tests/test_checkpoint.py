"""Tests for checkpointing system."""

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.checkpoint import (
    CheckpointManager,
    CheckpointData,
    get_rng_state,
    set_rng_state,
    resume_training,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class TestRNGState:
    """Test RNG state capture and restore."""

    def test_get_rng_state(self):
        """Test capturing RNG state."""
        state = get_rng_state()

        assert "python_random" in state
        assert "numpy" in state
        assert "torch" in state

    def test_set_rng_state_reproducibility(self):
        """Test that setting RNG state produces same random numbers."""
        import random
        import numpy as np

        # Capture state
        state = get_rng_state()

        # Generate some random numbers
        py_rand1 = random.random()
        np_rand1 = np.random.rand()
        torch_rand1 = torch.rand(1).item()

        # Restore state
        set_rng_state(state)

        # Generate again - should be same
        py_rand2 = random.random()
        np_rand2 = np.random.rand()
        torch_rand2 = torch.rand(1).item()

        assert py_rand1 == py_rand2
        assert np_rand1 == np_rand2
        assert torch_rand1 == torch_rand2


class TestCheckpointManager:
    """Test CheckpointManager."""

    def test_save_and_load_checkpoint(self):
        """Test saving and loading a checkpoint."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            # Save checkpoint
            metrics = {"accuracy": 0.95, "loss": 0.1}
            config = {"seed": 42}
            manager.save(model, optimizer, epoch=5, metrics=metrics, config=config, filename="test.pt")

            # Load checkpoint
            checkpoint = manager.load(Path(tmpdir) / "test.pt")

            assert checkpoint.epoch == 5
            assert checkpoint.config["seed"] == 42
            assert "model_state_dict" in checkpoint.__dict__

    def test_save_if_needed_best(self):
        """Test automatic best checkpoint saving."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, save_best=True, save_latest=False)

            # Save with lower accuracy
            manager.save_if_needed(model, None, 0, {"accuracy": 0.8}, {})
            assert (Path(tmpdir) / "checkpoint_best.pt").exists()

            # Save with higher accuracy - should update best
            manager.save_if_needed(model, None, 1, {"accuracy": 0.9}, {})
            assert manager.best_metric == 0.9
            assert manager.best_epoch == 1

    def test_save_if_needed_latest(self):
        """Test automatic latest checkpoint saving."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, save_best=False, save_latest=True)

            manager.save_if_needed(model, None, 0, {"accuracy": 0.8}, {})
            assert (Path(tmpdir) / "checkpoint_latest.pt").exists()

    def test_has_checkpoint(self):
        """Test checkpoint existence check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            assert not manager.has_checkpoint()

            model = SimpleModel()
            manager.save(model, None, 0, {}, {}, "checkpoint_latest.pt")
            assert manager.has_checkpoint()

    def test_load_latest(self):
        """Test loading latest checkpoint."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, save_latest=True)
            manager.save_if_needed(model, None, 5, {"accuracy": 0.9}, {})

            checkpoint = manager.load_latest()
            assert checkpoint is not None
            assert checkpoint.epoch == 5

    def test_load_best(self):
        """Test loading best checkpoint."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, save_best=True)
            manager.save_if_needed(model, None, 5, {"accuracy": 0.9}, {})

            checkpoint = manager.load_best()
            assert checkpoint is not None
            assert checkpoint.epoch == 5

    def test_metrics_history(self):
        """Test metrics history tracking."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, save_latest=True)

            for i in range(5):
                manager.save_if_needed(model, None, i, {"accuracy": 0.8 + i * 0.02}, {})

            assert len(manager.metrics_history["accuracy"]) == 5
            assert manager.metrics_history["accuracy"][-1] == 0.88

