"""Tests for trainer classes."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from networks.fc_network import FCNetwork
from trainers.base_trainer import BaseTrainer
from trainers.stsf_trainer import STSFTrainer
from trainers.decolle_trainer import DECOLLETrainer


class TestBaseTrainer:
    """Test BaseTrainer abstract class."""

    def test_base_trainer_is_abstract(self):
        """Test that BaseTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTrainer()

    def test_base_trainer_requires_train_sample(self):
        """Test that subclass must implement train_sample."""

        class IncompleteTrainer(BaseTrainer):
            def reset(self):
                pass

        with pytest.raises(TypeError):
            IncompleteTrainer()

    def test_base_trainer_requires_reset(self):
        """Test that subclass must implement reset."""

        class IncompleteTrainer(BaseTrainer):
            def train_sample(self, data, target):
                pass

        with pytest.raises(TypeError):
            IncompleteTrainer()


class TestSTSFTrainer:
    """Test STSFTrainer class."""

    @pytest.fixture
    def network(self):
        """Create a test network."""
        return FCNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
        )

    @pytest.fixture
    def trainer(self, network):
        """Create a test trainer."""
        return STSFTrainer(
            network=network,
            lr=0.01,
            batch_size=32,
            quant=False,
            use_optimizer=False,
        )

    def test_trainer_creation(self, trainer):
        """Test creating an STSF trainer."""
        assert trainer.lr == 0.01
        assert trainer.quant is False
        assert trainer.use_optimizer is False

    def test_trainer_has_network(self, trainer, network):
        """Test that trainer has network reference."""
        assert trainer.network is network

    def test_trainer_has_feedback_matrices(self, trainer):
        """Test that trainer creates feedback matrices."""
        assert len(trainer.feedback) > 0

        # Check feedback matrix shapes
        for fb in trainer.feedback:
            assert fb.shape[0] == 10  # n_classes
            assert fb.requires_grad is False

    def test_trainer_reset(self, trainer):
        """Test trainer reset."""
        # Should not raise
        trainer.reset()

    def test_trainer_train_sample(self, trainer):
        """Test training on a single sample."""
        # Create sample data: [timesteps, batch, features]
        timesteps = 10
        batch_size = 32
        data = torch.randn(timesteps, batch_size, 784)
        target = torch.randint(0, 10, (batch_size,))

        loss, pred = trainer.train_sample(data, target)

        assert loss.shape == ()  # scalar
        assert pred.shape == (batch_size, 1)
        assert not torch.isnan(loss)

    def test_trainer_predictions_valid(self, trainer):
        """Test that predictions are valid class indices."""
        timesteps = 10
        batch_size = 16
        data = torch.randn(timesteps, batch_size, 784)
        target = torch.randint(0, 10, (batch_size,))

        loss, pred = trainer.train_sample(data, target)

        # Predictions should be in [0, 9]
        assert pred.min() >= 0
        assert pred.max() <= 9

    def test_trainer_with_optimizer(self, network):
        """Test trainer with optimizer enabled."""
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

        trainer = STSFTrainer(
            network=network,
            lr=0.01,
            batch_size=32,
            quant=False,
            use_optimizer=True,
            optimizer=optimizer,
        )

        assert trainer.use_optimizer is True
        assert trainer.optimizer is optimizer

    def test_trainer_update_last(self, network):
        """Test trainer with update_last option."""
        trainer = STSFTrainer(
            network=network,
            lr=0.01,
            batch_size=32,
            update_last=True,
        )

        assert trainer.update_last is True

        # Should still work
        data = torch.randn(10, 8, 784)
        target = torch.randint(0, 10, (8,))
        loss, pred = trainer.train_sample(data, target)
        assert not torch.isnan(loss)

    def test_trainer_update_every(self, network):
        """Test trainer with update_every option."""
        trainer = STSFTrainer(
            network=network,
            lr=0.01,
            batch_size=32,
            update_every=5,
        )

        assert trainer.update_every == 5

        # Should still work
        data = torch.randn(10, 8, 784)
        target = torch.randint(0, 10, (8,))
        loss, pred = trainer.train_sample(data, target)
        assert not torch.isnan(loss)

    def test_trainer_device_transfer(self, trainer):
        """Test moving trainer to device."""
        trainer = trainer.to("cpu")

        data = torch.randn(10, 4, 784)
        target = torch.randint(0, 10, (4,))

        loss, pred = trainer.train_sample(data, target)
        assert loss.device.type == "cpu"

    def test_trainer_seq_batch(self, network):
        """Test trainer with seq_batch_size > 1."""
        trainer = STSFTrainer(
            network=network,
            lr=0.01,
            batch_size=1,
            seq_batch_size=4,
        )

        assert trainer.seq_batch_size == 4

        # Should work with batch_size=1
        data = torch.randn(10, 1, 784)
        target = torch.randint(0, 10, (1,))
        loss, pred = trainer.train_sample(data, target)
        assert not torch.isnan(loss)

    def test_trainer_weights_change(self, network):
        """Test that training changes network weights."""
        trainer = STSFTrainer(
            network=network,
            lr=0.1,  # Higher LR to see change
            batch_size=32,
        )

        # Get initial weights
        initial_weights = network.layers[0].weight.data.clone()

        # Train on a batch
        data = torch.randn(10, 32, 784)
        target = torch.randint(0, 10, (32,))
        trainer.train_sample(data, target)

        # Weights should have changed
        assert not torch.allclose(initial_weights, network.layers[0].weight.data)


class TestDECOLLETrainer:
    """Test DECOLLETrainer class (now uses FCNetwork)."""

    @pytest.fixture
    def network(self):
        return FCNetwork(layer_sizes=[32, 16, 4], beta=0.9)

    @pytest.fixture
    def trainer(self, network):
        return DECOLLETrainer(
            network=network,
            lr=0.05,
            batch_size=8,
        )

    def test_trainer_creation(self, trainer):
        assert isinstance(trainer.network, FCNetwork)

    def test_trainer_train_sample(self, trainer):
        data = torch.randint(0, 2, (6, 8, 32)).float()
        target = torch.randint(0, 4, (8,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.shape == ()
        assert pred.shape == (8, 1)

    def test_trainer_updates_weights(self, trainer, network):
        data = torch.randint(0, 2, (4, 8, 32)).float()
        target = torch.randint(0, 4, (8,))
        # FCNetwork uses layers[0::2] for Linear layers
        before = network.layers[0].weight.clone()
        trainer.train_sample(data, target)
        assert not torch.allclose(before, network.layers[0].weight)

