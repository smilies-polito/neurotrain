"""Tests for trainer classes."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from networks.fc_network import FCNetwork
from networks.recurrent_srnn import RecurrentSRNN
from trainers.base_trainer import BaseTrainer
from trainers.bptt_trainer import BPTTTrainer
from trainers.decolle_trainer import DECOLLETrainer
from trainers.drtp_trainer import DRTPTrainer
from trainers.eprop_trainer import EpropTrainer
from trainers.etlp_trainer import ETLPTrainer
from trainers.stsf_trainer import STSFTrainer


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


class TestEpropTrainer:
    """Test EpropTrainer class."""

    @pytest.fixture
    def network(self):
        """Create a test recurrent network."""
        return RecurrentSRNN(
            n_in=784,
            n_rec=100,
            n_out=10,
            threshold=1.0,
            tau_mem=2.0,
            tau_out=0.02,
            dt=1e-3,
        )

    @pytest.fixture
    def trainer(self, network):
        """Create a test trainer."""
        return EpropTrainer(
            network=network,
            lr=0.01,
            batch_size=32,
            use_optimizer=False,
        )

    def test_trainer_creation(self, trainer):
        """Test creating an E-prop trainer."""
        assert trainer.lr == 0.01
        assert trainer.use_optimizer is False
        assert trainer.gamma == 0.3  # default surrogate gradient parameter

    def test_trainer_has_network(self, trainer, network):
        """Test that trainer has network reference."""
        assert trainer.network is network

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

        trainer = EpropTrainer(
            network=network,
            lr=0.01,
            batch_size=32,
            use_optimizer=True,
            optimizer=optimizer,
        )

        assert trainer.use_optimizer is True
        assert trainer.optimizer is optimizer

    def test_trainer_update_last(self, network):
        """Test trainer with update_last option."""
        trainer = EpropTrainer(
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
        trainer = EpropTrainer(
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

    def test_trainer_weights_change(self, network):
        """Test that training changes network weights."""
        trainer = EpropTrainer(
            network=network,
            lr=0.1,  # Higher LR to see change
            batch_size=32,
        )

        # Get initial weights
        initial_weights = network.w_in.data.clone()

        # Train on a batch
        data = torch.randn(10, 32, 784)
        target = torch.randint(0, 10, (32,))
        trainer.train_sample(data, target)

        # Weights should have changed
        assert not torch.allclose(initial_weights, network.w_in.data)

    def test_trainer_surrogate_gradient_parameter(self, network):
        """Test that surrogate gradient parameter can be customized."""
        trainer = EpropTrainer(
            network=network,
            lr=0.01,
            batch_size=32,
            gamma=0.5,  # Custom gamma
        )

        assert trainer.gamma == 0.5

    def test_trainer_layer_lr_norm(self, network):
        """Test that per-layer learning rate can be customized."""
        trainer = EpropTrainer(
            network=network,
            lr=0.01,
            batch_size=32,
            lr_layer_norm=(0.1, 0.5, 1.0),
        )

        assert trainer.lr_layer == (0.1, 0.5, 1.0)

    def test_trainer_inherits_base_trainer(self, trainer):
        """Test that EpropTrainer properly inherits from BaseTrainer."""
        assert isinstance(trainer, BaseTrainer)


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


class TestDRTPTrainer:
    """Test DRTPTrainer class."""

    @pytest.fixture
    def network(self):
        return FCNetwork(layer_sizes=[2, 6, 2], beta=0.9)

    @pytest.fixture
    def trainer(self, network):
        return DRTPTrainer(
            network=network,
            lr=0.05,
            batch_size=4,
            feedback_distribution="kaiming_uniform",
            feedback_scale=0.1,
            fixed_feedback=True,
            use_optimizer=False,
        )

    def test_trainer_has_feedback(self, trainer):
        assert len(trainer.feedback) == 1
        fb = trainer.feedback[0]
        assert fb.shape == (2, 6)
        assert fb.requires_grad is False

    def test_trainer_train_sample(self, trainer):
        data = torch.randn(3, 4, 2)
        target = torch.randint(0, 2, (4,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.shape == ()
        assert pred.shape == (4, 1)

    def test_loss_decreases(self, trainer, network):
        timesteps = 2
        batch_size = 4
        features = 2

        data = torch.zeros(timesteps, batch_size, features)
        data[:, 0, 0] = 5.0
        data[:, 1, 0] = 5.0
        data[:, 2, 1] = 5.0
        data[:, 3, 1] = 5.0
        target = torch.tensor([0, 0, 1, 1])

        def forward_loss():
            network.reset()
            spk_sum = None
            for t in range(timesteps):
                spks, _ = network(data[t])
                spk_sum = spks[-1] if spk_sum is None else spk_sum + spks[-1]
            tgt = torch.zeros(batch_size, 2)
            tgt.scatter_(1, target.unsqueeze(1), 1.0)
            return torch.nn.functional.mse_loss(spk_sum, tgt).item()

        loss_before = forward_loss()

        for _ in range(20):
            trainer.train_sample(data, target)

        loss_after = forward_loss()
        assert loss_after < loss_before


class TestETLPTrainer:
    """Test ETLPTrainer class."""

    @pytest.fixture
    def network(self):
        return FCNetwork(layer_sizes=[16, 8, 4], beta=0.9)

    @pytest.fixture
    def trainer(self, network):
        return ETLPTrainer(
            network=network,
            lr=0.01,
            batch_size=4,
            update_rate_hz=100.0,
        )

    def test_trainer_creation(self, trainer):
        assert trainer.lr == 0.01
        assert trainer.batch_size == 4

    def test_trainer_train_sample(self, trainer):
        timesteps = 5
        batch_size = 4
        data = torch.rand(timesteps, batch_size, 16)
        target = torch.randint(0, 4, (batch_size,))

        loss, pred = trainer.train_sample(data, target)

        assert loss.shape == ()
        assert pred.shape == (batch_size, 1)
        assert not torch.isnan(loss)
