"""Integration tests for the full training pipeline."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.config import Config, load_config, validate_config
from utils.checkpoint import CheckpointManager
from utils.experiment_logger import ExperimentLogger, set_all_seeds
from networks.fc_network import FCNetwork
from trainers.stsf_trainer import STSFTrainer
from LearningAlgorithms import LearningAlgorithms
from datasets.get_loader import get_loader


class TestFullPipeline:
    """Test the full training pipeline integration."""

    @pytest.fixture
    def config(self):
        """Create a minimal test config."""
        config = Config()
        config.experiment.name = "integration_test"
        config.experiment.seed = 42
        config.model.layer_sizes = [784, 50, 10]  # Small network
        config.training.epochs = 2
        config.training.batch_size = 32
        config.training.learning_rate = 0.01
        config.data.timesteps = 5
        return config

    def test_config_to_training(self, config):
        """Test that config can be used to set up training."""
        set_all_seeds(config.experiment.seed)

        network = FCNetwork(
            layer_sizes=config.model.layer_sizes,
            beta=config.model.beta,
        )

        trainer = STSFTrainer(
            network=network,
            lr=config.training.learning_rate,
            batch_size=config.training.batch_size,
        )

        assert trainer is not None
        assert network.input_size == 784

    def test_training_loop_runs(self, config):
        """Test that a training loop can execute."""
        set_all_seeds(config.experiment.seed)

        # Create network and trainer
        network = FCNetwork(
            layer_sizes=config.model.layer_sizes,
            beta=config.model.beta,
        )

        trainer = STSFTrainer(
            network=network,
            lr=config.training.learning_rate,
            batch_size=config.training.batch_size,
        )

        # Get data
        trainloader, testloader = get_loader(
            config.data.dataset,
            config.training.batch_size,
            config.data.timesteps,
        )

        # Run one epoch
        torch.set_grad_enabled(False)
        metrics = LearningAlgorithms.train_epoch(
            trainer, trainloader, device="cpu", print_every=None
        )

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0
        assert metrics["accuracy"] <= 1

    def test_evaluation_runs(self, config):
        """Test that evaluation works."""
        set_all_seeds(config.experiment.seed)

        network = FCNetwork(
            layer_sizes=config.model.layer_sizes,
            beta=config.model.beta,
        )

        _, testloader = get_loader(
            config.data.dataset,
            config.training.batch_size,
            config.data.timesteps,
        )

        metrics = LearningAlgorithms.evaluate(
            network, testloader, device="cpu", print_every=None
        )

        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0

    def test_checkpoint_save_load_integration(self, config):
        """Test checkpointing during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_all_seeds(config.experiment.seed)

            # Setup
            network = FCNetwork(
                layer_sizes=config.model.layer_sizes,
                beta=config.model.beta,
            )

            checkpoint_manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                save_best=True,
                save_latest=True,
            )

            # Simulate training
            metrics = {"accuracy": 0.85, "loss": 0.5}
            checkpoint_manager.save_if_needed(
                model=network,
                optimizer=None,
                epoch=0,
                metrics=metrics,
                config=config.to_dict(),
            )

            # Load and verify
            checkpoint = checkpoint_manager.load_latest()
            assert checkpoint.epoch == 0
            assert checkpoint.config["experiment"]["seed"] == 42

            # Load weights into new network
            network2 = FCNetwork(
                layer_sizes=config.model.layer_sizes,
                beta=config.model.beta,
            )
            network2.load_state_dict(checkpoint.model_state_dict)

            # Verify weights match
            for (n1, p1), (n2, p2) in zip(
                network.named_parameters(), network2.named_parameters()
            ):
                assert torch.allclose(p1, p2)


class TestConfigFileIntegration:
    """Test loading and using config files."""

    def test_load_mnist_config(self):
        """Test loading the mnist_default.yaml config."""
        config_path = Path(__file__).parent.parent / "configs" / "mnist_default.yaml"

        if config_path.exists():
            config = load_config(config_path)

            assert config.experiment.name == "STSF_MNIST"
            assert config.data.dataset == "MNIST"
            assert validate_config(config) == []

    def test_config_to_network(self):
        """Test creating network from config."""
        config_path = Path(__file__).parent.parent / "configs" / "mnist_default.yaml"

        if config_path.exists():
            config = load_config(config_path)

            network = FCNetwork(
                layer_sizes=config.model.layer_sizes,
                beta=config.model.beta,
            )

            assert network.input_size == 784
            assert network.n_classes == 10


class TestExperimentLoggerIntegration:
    """Test experiment logger integration."""

    def test_logger_with_training(self):
        """Test logger captures training run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.experiment.name = "logger_test"
            config.model.layer_sizes = [784, 50, 10]
            config.training.batch_size = 32
            config.data.timesteps = 5

            logger = ExperimentLogger(
                experiment_name=config.experiment.name,
                config=config.to_dict(),
                seed=config.experiment.seed,
                log_dir=tmpdir,
            )

            context = logger.setup()

            assert context.seed == 42
            assert context.experiment_name == "logger_test"

            # Save context
            context_path = logger.save_context()
            assert context_path.exists()

            # Log some metrics
            logger.log_metrics({"accuracy": 0.9, "loss": 0.1}, step=0)

            logger.close()


class TestReproducibility:
    """Test reproducibility of training."""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        config = Config()
        config.model.layer_sizes = [784, 50, 10]
        config.training.batch_size = 64
        config.data.timesteps = 5

        results = []

        for _ in range(2):
            set_all_seeds(42)

            network = FCNetwork(
                layer_sizes=config.model.layer_sizes,
                beta=config.model.beta,
            )

            trainer = STSFTrainer(
                network=network,
                lr=0.01,
                batch_size=config.training.batch_size,
            )

            # Create deterministic data
            torch.manual_seed(42)
            data = torch.randn(config.data.timesteps, config.training.batch_size, 784)
            target = torch.randint(0, 10, (config.training.batch_size,))

            loss, pred = trainer.train_sample(data, target)
            results.append((loss.item(), pred.sum().item()))

        # Results should be identical
        assert results[0][0] == results[1][0]  # loss
        assert results[0][1] == results[1][1]  # predictions

    def test_different_seed_different_results(self):
        """Test that different seeds produce different results."""
        config = Config()
        config.model.layer_sizes = [784, 50, 10]

        results = []

        for seed in [42, 43]:
            set_all_seeds(seed)

            network = FCNetwork(
                layer_sizes=config.model.layer_sizes,
                beta=config.model.beta,
            )

            # Get first layer weights
            weights = network.layers[0].weight.data.clone()
            results.append(weights)

        # Weights should be different
        assert not torch.allclose(results[0], results[1])

