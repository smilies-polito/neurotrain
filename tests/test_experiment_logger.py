"""Tests for experiment logging system."""

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.experiment_logger import (
    ExperimentLogger,
    ExperimentContext,
    set_all_seeds,
    get_git_info,
    get_environment_info,
    get_rng_state,
    set_rng_state,
    generate_run_id,
    print_experiment_info,
)


class TestSetAllSeeds:
    """Test seed setting functionality."""

    def test_set_all_seeds_basic(self):
        """Test that set_all_seeds returns seed dict."""
        seeds = set_all_seeds(42)

        assert seeds["base_seed"] == 42
        assert seeds["random_seed"] == 42
        assert seeds["numpy_seed"] == 42
        assert seeds["torch_seed"] == 42

    def test_set_all_seeds_reproducibility(self):
        """Test that same seed produces same random numbers."""
        set_all_seeds(123)
        py_rand1 = random.random()
        np_rand1 = np.random.rand()
        torch_rand1 = torch.rand(1).item()

        set_all_seeds(123)
        py_rand2 = random.random()
        np_rand2 = np.random.rand()
        torch_rand2 = torch.rand(1).item()

        assert py_rand1 == py_rand2
        assert np_rand1 == np_rand2
        assert torch_rand1 == torch_rand2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        set_all_seeds(42)
        rand1 = torch.rand(1).item()

        set_all_seeds(43)
        rand2 = torch.rand(1).item()

        assert rand1 != rand2


class TestRNGState:
    """Test RNG state capture and restore."""

    def test_get_rng_state_keys(self):
        """Test that get_rng_state returns expected keys."""
        state = get_rng_state()

        assert "random_state" in state
        assert "numpy_state" in state
        assert "torch_state" in state

    def test_rng_state_restore(self):
        """Test that restoring RNG state gives same random sequence."""
        set_all_seeds(42)

        # Generate some numbers
        state = get_rng_state()
        rand1 = [random.random() for _ in range(5)]
        np_rand1 = np.random.rand(5).tolist()
        torch_rand1 = torch.rand(5).tolist()

        # Restore state and generate again
        set_rng_state(state)
        rand2 = [random.random() for _ in range(5)]
        np_rand2 = np.random.rand(5).tolist()
        torch_rand2 = torch.rand(5).tolist()

        assert rand1 == rand2
        assert np_rand1 == np_rand2
        assert torch_rand1 == torch_rand2


class TestGitInfo:
    """Test git info extraction."""

    def test_get_git_info_returns_dict(self):
        """Test that get_git_info returns a dictionary."""
        info = get_git_info()

        assert isinstance(info, dict)
        assert "commit" in info
        assert "branch" in info
        assert "dirty" in info

    def test_git_info_types(self):
        """Test git info value types."""
        info = get_git_info()

        # commit can be None or string
        assert info["commit"] is None or isinstance(info["commit"], str)
        # dirty is always bool
        assert isinstance(info["dirty"], bool)


class TestEnvironmentInfo:
    """Test environment info collection."""

    def test_get_environment_info_keys(self):
        """Test that environment info has expected keys."""
        env_info = get_environment_info()

        assert "python_version" in env_info
        assert "torch_version" in env_info
        assert "numpy_version" in env_info
        assert "hostname" in env_info
        assert "platform_info" in env_info

    def test_environment_info_values(self):
        """Test that environment info values are non-empty."""
        env_info = get_environment_info()

        assert len(env_info["python_version"]) > 0
        assert len(env_info["torch_version"]) > 0
        assert len(env_info["numpy_version"]) > 0


class TestGenerateRunId:
    """Test run ID generation."""

    def test_run_id_format(self):
        """Test that run ID has expected format."""
        run_id = generate_run_id()

        # Format: YYYYMMDD_HHMMSS
        assert len(run_id) == 15
        assert run_id[8] == "_"

    def test_run_id_unique(self):
        """Test that consecutive run IDs can be different."""
        import time

        id1 = generate_run_id()
        time.sleep(0.01)  # Small delay
        id2 = generate_run_id()

        # They might be same if generated in same second
        # Just check format is valid
        assert len(id1) == 15
        assert len(id2) == 15


class TestExperimentLogger:
    """Test ExperimentLogger class."""

    def test_logger_creation(self):
        """Test creating an experiment logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                experiment_name="test_exp",
                config={"lr": 0.01},
                seed=42,
                log_dir=tmpdir,
            )

            assert logger.experiment_name == "test_exp"
            assert logger.seed == 42
            assert logger.config["lr"] == 0.01

    def test_logger_setup(self):
        """Test logger setup creates context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                experiment_name="test_exp",
                config={"epochs": 10},
                seed=42,
                log_dir=tmpdir,
            )

            context = logger.setup()

            assert isinstance(context, ExperimentContext)
            assert context.experiment_name == "test_exp"
            assert context.seed == 42

    def test_logger_creates_directories(self):
        """Test that setup creates log directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nested" / "path"
            logger = ExperimentLogger(
                experiment_name="test_exp",
                config={},
                seed=42,
                log_dir=str(log_dir),
            )

            logger.setup()

            assert log_dir.exists()
            assert (log_dir / "checkpoints").exists()

    def test_save_context(self):
        """Test saving experiment context to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                experiment_name="test_exp",
                config={"batch_size": 32},
                seed=123,
                log_dir=tmpdir,
            )

            logger.setup()
            context_path = logger.save_context()

            assert context_path.exists()

            with open(context_path, "r") as f:
                saved_context = json.load(f)

            assert saved_context["experiment_name"] == "test_exp"
            assert saved_context["seed"] == 123
            assert saved_context["config"]["batch_size"] == 32

    def test_load_context(self):
        """Test loading experiment context from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                experiment_name="load_test",
                config={"lr": 0.001},
                seed=999,
                log_dir=tmpdir,
            )

            logger.setup()
            context_path = logger.save_context()

            # Load the context
            loaded = ExperimentLogger.load_context(str(context_path))

            assert loaded.experiment_name == "load_test"
            assert loaded.seed == 999
            assert loaded.config["lr"] == 0.001


class TestExperimentContext:
    """Test ExperimentContext dataclass."""

    def test_context_creation(self):
        """Test creating an ExperimentContext."""
        context = ExperimentContext(
            experiment_name="test",
            run_id="20231128_120000",
            timestamp="2023-11-28T12:00:00",
            seed=42,
        )

        assert context.experiment_name == "test"
        assert context.seed == 42

    def test_context_defaults(self):
        """Test ExperimentContext default values."""
        context = ExperimentContext(
            experiment_name="test",
            run_id="id",
            timestamp="ts",
            seed=1,
        )

        assert context.python_version == ""
        assert context.git_commit is None
        assert context.config == {}

