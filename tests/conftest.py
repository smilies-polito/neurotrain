"""Pytest configuration and shared fixtures."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def configs_dir(project_root):
    """Get configs directory."""
    return project_root / "configs"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_network():
    """Create a simple test network."""
    from networks.fc_network import FCNetwork

    return FCNetwork(
        layer_sizes=[784, 100, 10],
        beta=0.9,
    )


@pytest.fixture
def small_network():
    """Create a small network for faster tests."""
    from networks.fc_network import FCNetwork

    return FCNetwork(
        layer_sizes=[784, 50, 10],
        beta=0.9,
    )


@pytest.fixture
def trainer(simple_network):
    """Create a test trainer."""
    from trainers.stsf_trainer import STSFTrainer

    return STSFTrainer(
        network=simple_network,
        lr=0.01,
        batch_size=32,
    )


@pytest.fixture
def sample_batch():
    """Create a sample batch of data."""
    timesteps = 10
    batch_size = 32
    features = 784

    data = torch.randn(timesteps, batch_size, features)
    target = torch.randint(0, 10, (batch_size,))

    return data, target


@pytest.fixture
def small_sample_batch():
    """Create a small sample batch for faster tests."""
    timesteps = 5
    batch_size = 8
    features = 784

    data = torch.randn(timesteps, batch_size, features)
    target = torch.randint(0, 10, (batch_size,))

    return data, target


@pytest.fixture
def default_config():
    """Create a default config."""
    from utils.config import Config

    return Config()


@pytest.fixture
def test_config():
    """Create a config for testing (minimal settings)."""
    from utils.config import Config

    config = Config()
    config.experiment.name = "test_experiment"
    config.experiment.seed = 42
    config.model.layer_sizes = [784, 50, 10]
    config.training.epochs = 2
    config.training.batch_size = 32
    config.data.timesteps = 5
    return config


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed before each test."""
    torch.manual_seed(42)
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)


@pytest.fixture
def checkpoint_manager(temp_dir):
    """Create a checkpoint manager."""
    from utils.checkpoint import CheckpointManager

    return CheckpointManager(
        checkpoint_dir=temp_dir,
        save_best=True,
        save_latest=True,
    )


@pytest.fixture
def experiment_logger(temp_dir):
    """Create an experiment logger."""
    from utils.experiment_logger import ExperimentLogger

    logger = ExperimentLogger(
        experiment_name="test_exp",
        config={"lr": 0.01},
        seed=42,
        log_dir=str(temp_dir),
    )
    yield logger
    logger.close()

