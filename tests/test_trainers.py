"""Tests for trainer classes."""

import os
import sys

import pytest
import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from networks.conv_network import ConvFCNetwork
from networks.fc_network import FCNetwork
from networks.local_classifier_network import LocalClassifierNetwork
from networks.recurrent_fc_network import RecurrentFCNetwork
from networks.recurrent_srnn import RecurrentSRNN

from LearningAlgorithms import LearningAlgorithms
from networks.benchmarking.fc_snn import FCSNN
from networks.reproducibility.DRTP_convolutional_network import DRTPConvMNIST
from trainers.base_trainer import BaseTrainer
from trainers.bell_trainer import BELLTrainer
from trainers.bptt_trainer import BPTTTrainer
from trainers.decolle_trainer import DECOLLETrainer
from trainers.drtp_trainer import DRTPTrainer
from trainers.ell_trainer import ELLTrainer
from trainers.eprop_trainer import EpropTrainer
from trainers.esd_rtrl_trainer import ESDRTRLTrainer
from trainers.etlp_trainer import ETLPTrainer
from trainers.fell_trainer import FELLTrainer
from trainers.ostl_trainer import OSTLTrainer
from trainers.osttp_trainer import OSTTPTrainer
from trainers.stllr_trainer import STLLRTrainer
from trainers.stop_trainer import STOPTrainer
from trainers.stsf_trainer import STSFTrainer


def _make_ostl_temporal_batch(
    batch_size: int = 64,
    timesteps: int = 6,
    in_features: int = 4,
    n_classes: int = 2,
    device: str = "cpu",
):
    """Create a tiny separable temporal task for OSTL smoke/integration tests."""
    target = torch.randint(0, n_classes, (batch_size,), device=device)

    x_static = torch.zeros(batch_size, in_features, device=device)
    x_static[target == 0, :2] = 1.0
    x_static[target == 1, 2:] = 1.0
    x_static += 0.05 * torch.randn_like(x_static)

    data = x_static.unsqueeze(1).repeat(1, timesteps, 1)
    data += 0.01 * torch.randn_like(data)
    return data, target


class _JSBLikeSigmoidNet(nn.Module):
    """Linear->Leaky hidden layer with dense sigmoid readout."""

    def __init__(
        self, in_features: int = 88, hidden: int = 150, out_features: int = 88
    ):
        super().__init__()
        self._n_classes = out_features
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.lif1 = snn.Leaky(beta=0.9, reset_mechanism="zero", reset_delay=True)
        self.fc2 = nn.Linear(hidden, out_features)
        self.sigmoid = nn.Sigmoid()

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset(self) -> None:
        self.lif1.reset_mem()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur = self.fc1(x)
        spk, _mem = self.lif1(cur)
        logits = self.fc2(spk)
        return self.sigmoid(logits)


class _SHDLikeIntegratorNet(nn.Module):
    """Linear->RLeaky hidden layer with Linear->Leaky integrator readout."""

    def __init__(
        self, in_features: int = 64, hidden: int = 450, out_features: int = 20
    ):
        super().__init__()
        self._n_classes = out_features
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.rlif = snn.RLeaky(
            beta=0.9,
            linear_features=hidden,
            all_to_all=True,
            reset_mechanism="subtract",
            reset_delay=True,
        )
        self.fc2 = nn.Linear(hidden, out_features, bias=False)
        self.out_int = snn.Leaky(
            beta=0.95,
            threshold=1e9,
            reset_mechanism="none",
            reset_delay=True,
        )

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset(self) -> None:
        self.rlif.reset_mem()
        self.out_int.reset_mem()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur_h = self.fc1(x)
        spk_h, _mem_h = self.rlif(cur_h)
        cur_o = self.fc2(spk_h)
        _spk_o, mem_o = self.out_int(cur_o)
        return mem_o


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
            loss_type="mse",
            output_mode="spike",
            feedback_distribution="kaiming_uniform",
            feedback_scale=0.1,
            fixed_feedback=True,
            surrogate_scale=5.0,
            surrogate_type="logistic",
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

    def test_trainer_train_sample_mem_mode(self, network):
        trainer = DRTPTrainer(
            network=network,
            lr=0.01,
            batch_size=4,
            loss_type="bce",
            output_mode="mem",
            feedback_distribution="kaiming_uniform",
            feedback_scale=0.1,
            fixed_feedback=True,
            surrogate_scale=5.0,
            surrogate_type="logistic",
            use_optimizer=False,
        )
        data = torch.randn(3, 4, 2)
        target = torch.randint(0, 2, (4,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)
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

    def test_trainer_with_fcsnn(self):
        network = FCSNN(
            in_shape=(2,),
            hidden_sizes=(6,),
            num_classes=2,
            beta=0.9,
            threshold=1.0,
        )
        trainer = DRTPTrainer(
            network=network,
            lr=0.01,
            batch_size=4,
            loss_type="mse",
            output_mode="spike",
            feedback_distribution="kaiming_uniform",
            feedback_scale=0.1,
            fixed_feedback=True,
            surrogate_scale=5.0,
            surrogate_type="logistic",
            use_optimizer=False,
        )
        data = torch.randn(3, 4, 2)
        target = torch.randint(0, 2, (4,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert pred.shape == (4, 1)

    def test_trainer_with_fcsnn_multidim_input(self):
        network = FCSNN(
            in_shape=(1, 2, 2),
            hidden_sizes=(6,),
            num_classes=2,
            beta=0.9,
            threshold=1.0,
        )
        trainer = DRTPTrainer(
            network=network,
            lr=0.01,
            batch_size=4,
            loss_type="mse",
            output_mode="mem",
            feedback_distribution="kaiming_uniform",
            feedback_scale=0.1,
            fixed_feedback=True,
            surrogate_scale=5.0,
            surrogate_type="logistic",
            use_optimizer=False,
        )
        data = torch.randn(3, 4, 1, 2, 2)
        target = torch.randint(0, 2, (4,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert pred.shape == (4, 1)

    def test_trainer_with_drtp_conv_mnist(self):
        network = DRTPConvMNIST(
            in_shape=(1, 28, 28),
            num_classes=10,
            conv_out_channels=8,
            fc_hidden=16,
            beta=0.9,
            threshold=1.0,
        )
        trainer = DRTPTrainer(
            network=network,
            lr=0.01,
            batch_size=2,
            loss_type="bce",
            output_mode="mem",
            feedback_distribution="kaiming_uniform",
            feedback_scale=0.1,
            fixed_feedback=True,
            surrogate_scale=5.0,
            surrogate_type="logistic",
            use_optimizer=False,
        )
        data = torch.randn(2, 2, 1, 28, 28)
        target = torch.randint(0, 10, (2,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert pred.shape == (2, 1)


class TestOSTLTrainer:
    """OSTL trainer tests on synthetic temporal classification."""

    def _make_trainer(self, lr: float = 0.05, output_mode: str = "spike"):
        network = FCSNN(
            in_shape=(4,),
            hidden_sizes=(8,),
            num_classes=2,
            beta=0.9,
            threshold=0.5,
        )
        return OSTLTrainer(
            network=network,
            lr=lr,
            batch_size=64,
            surrogate_scale=5.0,
            grad_clip=1.0,
            use_optimizer=False,
            output_mode=output_mode,
        )

    def test_train_sample_shapes_and_finite(self):
        trainer = self._make_trainer()
        data, target = _make_ostl_temporal_batch(batch_size=32, timesteps=5)

        loss, pred = trainer.train_sample(data.transpose(0, 1), target)

        assert loss.shape == ()
        assert pred.shape == (32, 1)
        assert torch.isfinite(loss)
        assert pred.min().item() >= 0
        assert pred.max().item() <= 1

    def test_loss_decreases_on_tiny_synthetic_task(self):
        trainer = self._make_trainer(lr=0.1)
        data, target = _make_ostl_temporal_batch(batch_size=96, timesteps=6)
        temporal = data.transpose(0, 1)

        losses = []
        for _ in range(30):
            loss, _ = trainer.train_sample(temporal, target)
            losses.append(float(loss.item()))

        first_window = sum(losses[:5]) / 5.0
        last_window = sum(losses[-5:]) / 5.0

        assert all(torch.isfinite(torch.tensor(losses)))
        assert last_window < first_window

    @pytest.mark.parametrize("timesteps", [3, 7, 11])
    def test_timestep_handling(self, timesteps):
        trainer = self._make_trainer()
        data, target = _make_ostl_temporal_batch(batch_size=24, timesteps=timesteps)

        loss, pred = trainer.train_sample(data.transpose(0, 1), target)

        assert torch.isfinite(loss)
        assert pred.shape == (24, 1)

    def test_train_sample_mem_output_mode_shapes_and_finite(self):
        trainer = self._make_trainer(output_mode="mem")
        data, target = _make_ostl_temporal_batch(batch_size=32, timesteps=6)

        loss, pred = trainer.train_sample(data.transpose(0, 1), target)

        assert loss.shape == ()
        assert pred.shape == (32, 1)
        assert torch.isfinite(loss)
        assert pred.min().item() >= 0
        assert pred.max().item() <= 1

    def test_invalid_output_mode_raises(self):
        network = FCSNN(
            in_shape=(4,),
            hidden_sizes=(8,),
            num_classes=2,
            beta=0.9,
            threshold=0.5,
        )
        with pytest.raises(ValueError, match="output_mode"):
            OSTLTrainer(
                network=network,
                lr=0.05,
                batch_size=32,
                surrogate_scale=5.0,
                grad_clip=1.0,
                use_optimizer=False,
                output_mode="invalid",
            )

    def test_learning_algorithms_train_epoch_integration(self):
        trainer = self._make_trainer(lr=0.05)
        data, target = _make_ostl_temporal_batch(batch_size=64, timesteps=6)

        dataset = TensorDataset(data.cpu(), target.cpu())
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        metrics = LearningAlgorithms.train_epoch(
            trainer=trainer,
            train_loader=loader,
            device="cpu",
            print_every=None,
        )

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_recurrent_network_is_rejected(self):
        network = RecurrentFCNetwork(
            layer_sizes=[4, 8, 2],
            beta=0.9,
            threshold=0.5,
            recurrent_type="snu",
        )
        with pytest.raises(TypeError, match="feed-forward only"):
            OSTLTrainer(
                network=network,
                lr=0.05,
                batch_size=32,
                surrogate_scale=5.0,
                grad_clip=1.0,
                use_optimizer=False,
            )

    def test_non_alternating_layers_are_rejected(self):
        class _BadNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(4, 8, bias=False), nn.Linear(8, 2, bias=False)]
                )
                self._n_classes = 2

            @property
            def n_classes(self):
                return self._n_classes

            def reset(self):
                pass

            def forward(self, x: torch.Tensor):
                raise RuntimeError("Not used")

        with pytest.raises(TypeError, match="alternating"):
            OSTLTrainer(
                network=_BadNet(),
                lr=0.05,
                batch_size=16,
                use_optimizer=False,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_train_sample_runs(self):
        trainer = self._make_trainer().to("cuda")
        data, target = _make_ostl_temporal_batch(
            batch_size=16,
            timesteps=5,
            device="cuda",
        )

        loss, pred = trainer.train_sample(data.transpose(0, 1), target)

        assert torch.isfinite(loss)
        assert pred.shape == (16, 1)
        assert pred.device.type == "cuda"


class TestOSTTPTrainer:
    """OSTTP trainer tests on synthetic temporal classification."""

    def _make_trainer(self, lr: float = 0.05):
        network = FCNetwork(
            layer_sizes=[4, 8, 2],
            beta=0.9,
            threshold=0.5,
        )
        return OSTTPTrainer(
            network=network,
            lr=lr,
            batch_size=64,
            pseudo_derivative="tanh",
            output_loss="ce",
            output_readout="spk",
            feedback_scale=1.0,
            feedback_seed=42,
            use_optimizer=False,
        )

    def test_train_sample_shapes_and_finite(self):
        trainer = self._make_trainer()
        data, target = _make_ostl_temporal_batch(batch_size=32, timesteps=5)

        loss, pred = trainer.train_sample(data.transpose(0, 1), target)

        assert loss.shape == ()
        assert pred.shape == (32, 1)
        assert torch.isfinite(loss)
        assert pred.min().item() >= 0
        assert pred.max().item() <= 1

    def test_weights_change(self):
        trainer = self._make_trainer(lr=0.1)
        data, target = _make_ostl_temporal_batch(batch_size=64, timesteps=6)
        temporal = data.transpose(0, 1)

        w0_before = trainer.network.layers[0].weight.detach().clone()
        trainer.train_sample(temporal, target)
        w0_after = trainer.network.layers[0].weight.detach()

        assert not torch.allclose(w0_before, w0_after)

    def test_feedback_matrices_fixed(self):
        trainer = self._make_trainer(lr=0.05)
        data, target = _make_ostl_temporal_batch(batch_size=16, timesteps=4)

        trainer.train_sample(data.transpose(0, 1), target)

        assert trainer._feedback_ready is True
        assert len(trainer._feedback_names) == 1  # one hidden layer for [4, 8, 2]
        fb = getattr(trainer, trainer._feedback_names[0])
        assert fb.requires_grad is False
        assert fb.shape == (2, 8)

    def test_jsb_like_sigmoid_readout_smoke(self):
        network = _JSBLikeSigmoidNet(in_features=88, hidden=150, out_features=88)
        trainer = OSTTPTrainer(
            network=network,
            lr=0.05,
            batch_size=4,
            pseudo_derivative="tanh",
            output_loss="bce_probs",
            output_readout="probs",
            target_dim=88,
            use_optimizer=False,
        )

        timesteps = 5
        batch_size = 4
        data = torch.randn(timesteps, batch_size, 88)
        target = torch.randint(0, 2, (timesteps, batch_size, 88), dtype=torch.float32)

        w_hidden_before = network.fc1.weight.detach().clone()
        w_out_before = network.fc2.weight.detach().clone()

        loss, pred = trainer.train_sample(data, target)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert pred.shape == (batch_size, 1)
        assert not torch.allclose(w_hidden_before, network.fc1.weight.detach())
        assert not torch.allclose(w_out_before, network.fc2.weight.detach())

    def test_jsb_like_logits_readout_smoke(self):
        class _JSBLikeLogitsNet(nn.Module):
            def __init__(self):
                super().__init__()
                self._n_classes = 88
                self.fc1 = nn.Linear(88, 150, bias=False)
                self.lif1 = snn.Leaky(
                    beta=0.9, reset_mechanism="zero", reset_delay=True
                )
                self.fc2 = nn.Linear(150, 88)

            @property
            def n_classes(self):
                return self._n_classes

            def reset(self):
                self.lif1.reset_mem()

            def forward(self, x: torch.Tensor):
                cur = self.fc1(x)
                spk, _mem = self.lif1(cur)
                return self.fc2(spk)

        network = _JSBLikeLogitsNet()
        trainer = OSTTPTrainer(
            network=network,
            lr=0.05,
            batch_size=4,
            pseudo_derivative="tanh",
            output_loss="bce_logits",
            output_readout="logits",
            target_dim=88,
            use_optimizer=False,
        )

        timesteps = 4
        batch_size = 4
        data = torch.randn(timesteps, batch_size, 88)
        target = torch.randint(0, 2, (timesteps, batch_size, 88), dtype=torch.float32)

        w_out_before = network.fc2.weight.detach().clone()
        loss, pred = trainer.train_sample(data, target)

        assert torch.isfinite(loss)
        assert pred.shape == (batch_size, 1)
        assert not torch.allclose(w_out_before, network.fc2.weight.detach())

    def test_shd_like_integrator_readout_smoke(self):
        network = _SHDLikeIntegratorNet(in_features=64, hidden=450, out_features=20)
        trainer = OSTTPTrainer(
            network=network,
            lr=0.02,
            batch_size=3,
            pseudo_derivative="tanh",
            output_loss="ce",
            output_readout="mem",
            target_dim=20,
            use_optimizer=False,
        )

        timesteps = 5
        batch_size = 3
        data = torch.randn(timesteps, batch_size, 64)
        target = torch.randint(0, 20, (batch_size,))

        w_hidden_before = network.fc1.weight.detach().clone()
        w_out_before = network.fc2.weight.detach().clone()

        loss, pred = trainer.train_sample(data, target)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert pred.shape == (batch_size, 1)
        assert not torch.allclose(w_hidden_before, network.fc1.weight.detach())
        assert not torch.allclose(w_out_before, network.fc2.weight.detach())


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


class TestSTOPTrainer:
    """Tests for STOP (spatiotemporal orthogonal propagation) trainer."""

    @pytest.fixture
    def fc_network(self):
        return FCNetwork(layer_sizes=[16, 12, 4], beta=0.9, threshold=0.7)

    @pytest.fixture
    def trainer(self, fc_network):
        return STOPTrainer(
            network=fc_network,
            lr=0.01,
            batch_size=8,
            loss_type="ce",
            surrogate="exp",
        )

    @staticmethod
    def _collect_thresholds(network):
        values = []
        for layer in network.layers:
            if hasattr(layer, "threshold"):
                thr = getattr(layer, "threshold")
                if isinstance(thr, torch.Tensor):
                    values.append(thr.detach().clone())
                else:
                    values.append(torch.tensor(float(thr)))
        return values

    @staticmethod
    def _collect_leaks(network):
        values = []
        for layer in network.layers:
            if hasattr(layer, "beta"):
                beta = getattr(layer, "beta")
                if isinstance(beta, torch.Tensor):
                    values.append(beta.detach().clone())
                else:
                    values.append(torch.tensor(float(beta)))
        return values

    def test_stop_train_sample_smoke(self, trainer):
        data = torch.rand(6, 8, 16)
        target = torch.randint(0, 4, (8,))
        loss, pred = trainer.train_sample(data, target)

        assert loss.shape == ()
        assert pred.shape == (8, 1)
        assert torch.isfinite(loss)
        assert pred.min() >= 0
        assert pred.max() <= 3

    def test_stop_weights_change_when_enabled(self, fc_network):
        trainer = STOPTrainer(
            network=fc_network,
            lr=0.05,
            batch_size=8,
            learn_weights=True,
            learn_thresholds=False,
            learn_leakage=False,
        )
        before = fc_network.layers[0].weight.detach().clone()
        data = torch.rand(6, 8, 16)
        target = torch.randint(0, 4, (8,))
        trainer.train_sample(data, target)
        after = fc_network.layers[0].weight.detach()
        assert not torch.allclose(before, after)

    def test_stop_w_mode_keeps_theta_and_alpha_fixed(self, fc_network):
        trainer = STOPTrainer(
            network=fc_network,
            lr=0.05,
            batch_size=8,
            learn_weights=True,
            learn_thresholds=False,
            learn_leakage=False,
        )

        theta_before = self._collect_thresholds(fc_network)
        alpha_before = self._collect_leaks(fc_network)

        data = torch.rand(5, 8, 16)
        target = torch.randint(0, 4, (8,))
        trainer.train_sample(data, target)

        theta_after = self._collect_thresholds(fc_network)
        alpha_after = self._collect_leaks(fc_network)
        for b, a in zip(theta_before, theta_after):
            assert torch.allclose(b, a)
        for b, a in zip(alpha_before, alpha_after):
            assert torch.allclose(b, a)

    def test_stop_wt_mode_keeps_alpha_fixed(self, fc_network):
        trainer = STOPTrainer(
            network=fc_network,
            lr=0.05,
            batch_size=8,
            learn_weights=True,
            learn_thresholds=True,
            learn_leakage=False,
        )

        alpha_before = self._collect_leaks(fc_network)
        data = torch.rand(5, 8, 16)
        target = torch.randint(0, 4, (8,))
        trainer.train_sample(data, target)
        alpha_after = self._collect_leaks(fc_network)

        for b, a in zip(alpha_before, alpha_after):
            assert torch.allclose(b, a)

    def test_stop_wl_mode_keeps_threshold_fixed(self, fc_network):
        trainer = STOPTrainer(
            network=fc_network,
            lr=0.05,
            batch_size=8,
            learn_weights=True,
            learn_thresholds=False,
            learn_leakage=True,
        )

        theta_before = self._collect_thresholds(fc_network)
        data = torch.rand(5, 8, 16)
        target = torch.randint(0, 4, (8,))
        trainer.train_sample(data, target)
        theta_after = self._collect_thresholds(fc_network)

        for b, a in zip(theta_before, theta_after):
            assert torch.allclose(b, a)

    def test_stop_clamps_threshold_and_leak(self, fc_network):
        trainer = STOPTrainer(
            network=fc_network,
            lr=0.5,
            batch_size=8,
            learn_weights=False,
            learn_thresholds=True,
            learn_leakage=True,
            lr_threshold=1.0,
            lr_leakage=1.0,
            threshold_min=0.05,
        )

        data = torch.rand(8, 8, 16)
        target = torch.randint(0, 4, (8,))
        trainer.train_sample(data, target)

        for layer in fc_network.layers:
            if hasattr(layer, "threshold"):
                thr = getattr(layer, "threshold")
                thr_min = (
                    float(thr.min().item())
                    if isinstance(thr, torch.Tensor)
                    else float(thr)
                )
                assert thr_min >= 0.05 - 1e-8
            if hasattr(layer, "beta"):
                beta = getattr(layer, "beta")
                if isinstance(beta, torch.Tensor):
                    assert float(beta.min().item()) >= -1e-8
                    assert float(beta.max().item()) <= 1.0 + 1e-8
                else:
                    assert 0.0 <= float(beta) <= 1.0

    def test_stop_conv_smoke(self):
        network = ConvFCNetwork(
            input_shape=(1, 8, 8),
            conv_layers=[
                {
                    "out_channels": 2,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "pool_kernel": 2,
                    "pool_stride": 2,
                }
            ],
            layer_sizes=[6, 3],
            beta=0.9,
            threshold=0.7,
            quant=False,
        )
        trainer = STOPTrainer(
            network=network,
            lr=0.01,
            batch_size=4,
            loss_type="mse",
            surrogate="rational",
        )

        data = torch.rand(5, 4, 1, 8, 8)
        target = torch.randint(0, 3, (4,))
        loss, pred = trainer.train_sample(data, target)

        assert loss.shape == ()
        assert pred.shape == (4, 1)
        assert torch.isfinite(loss)


class TestELLTrainer:
    """Test ELLTrainer class."""

    @pytest.fixture
    def network(self):
        return LocalClassifierNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
            mode="ell",
        )

    @pytest.fixture
    def trainer(self, network):
        return ELLTrainer(network=network, lr=0.001, batch_size=32)

    def test_trainer_creation(self, trainer):
        assert trainer.lr == 0.001
        assert len(trainer.optimizers) == 2  # 2 blocks

    def test_trainer_reset(self, trainer):
        trainer.reset()

    def test_trainer_train_sample(self, trainer):
        data = torch.randn(5, 32, 784)
        target = torch.randint(0, 10, (32,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.dim() == 0
        assert pred.shape == (32,)
        assert not torch.isnan(loss)

    def test_trainer_predictions_valid(self, trainer):
        data = torch.randn(5, 16, 784)
        target = torch.randint(0, 10, (16,))
        loss, pred = trainer.train_sample(data, target)
        assert pred.min() >= 0
        assert pred.max() <= 9

    def test_trainer_device_transfer(self, trainer):
        trainer = trainer.to("cpu")
        data = torch.randn(5, 4, 784)
        target = torch.randint(0, 10, (4,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.device.type == "cpu"


class TestFELLTrainer:
    """Test FELLTrainer class."""

    @pytest.fixture
    def network(self):
        return LocalClassifierNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
            mode="fell",
        )

    @pytest.fixture
    def trainer(self, network):
        return FELLTrainer(network=network, lr=0.001, batch_size=32)

    def test_trainer_train_sample(self, trainer):
        data = torch.randn(5, 32, 784)
        target = torch.randint(0, 10, (32,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.dim() == 0
        assert pred.shape == (32,)


class TestBELLTrainer:
    """Test BELLTrainer class."""

    @pytest.fixture
    def network(self):
        return LocalClassifierNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
            mode="bell",
        )

    @pytest.fixture
    def trainer(self, network):
        return BELLTrainer(network=network, lr=0.001, batch_size=32)

    def test_trainer_train_sample(self, trainer):
        data = torch.randn(5, 32, 784)
        target = torch.randint(0, 10, (32,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.dim() == 0
        assert pred.shape == (32,)


class TestSTLLRTrainer:
    """Test STLLRTrainer class."""

    @pytest.fixture
    def network(self):
        from networks.stllr_network import STLLRNetwork

        return STLLRNetwork(
            layer_sizes=[784, 100, 10],
            threshold=0.6,
            leak=2.0,
        )

    @pytest.fixture
    def trainer(self, network):
        return STLLRTrainer(
            network=network,
            lr=0.001,
            batch_size=32,
            delay_ls=5,
        )

    def test_stllr_trainer_smoke(self, trainer):
        """Smoke test: instantiate and run train_sample."""
        data = torch.randn(10, 32, 784)
        target = torch.randint(0, 10, (32,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.dim() == 0
        assert pred.shape == (32,)
        assert not torch.isnan(loss)
        assert pred.min() >= 0
        assert pred.max() <= 9

    def test_stllr_trainer_reset(self, trainer):
        trainer.reset()

    def test_stllr_trainer_device_transfer(self, trainer):
        trainer = trainer.to("cpu")
        data = torch.randn(5, 8, 784)
        target = torch.randint(0, 10, (8,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.device.type == "cpu"


class TestESDRTRLTrainer:
    """Test ESDRTRLTrainer class (ES-D-RTRL, BrainTrace)."""

    @pytest.fixture
    def network(self):
        """Create a small recurrent network for ES-D-RTRL."""
        return RecurrentSRNN(
            n_in=8,
            n_rec=16,
            n_out=4,
            threshold=1.0,
            tau_mem=2.0,
            tau_out=0.02,
            dt=1e-3,
        )

    @pytest.fixture
    def trainer(self, network):
        return ESDRTRLTrainer(
            network=network,
            lr=0.001,
            batch_size=8,
            etrace_decay=0.9,
            use_optimizer=True,
        )

    def test_esd_rtrl_trainer_smoke(self, trainer):
        """Smoke test: run train_sample, check loss and pred shape."""
        T, B, F = 5, 8, 8
        data = torch.randn(T, B, F)
        target = torch.randint(0, 4, (B,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.dim() == 0
        assert pred.shape == (B, 1)
        assert not torch.isnan(loss)
        assert pred.min() >= 0
        assert pred.max() <= 3

    def test_esd_rtrl_trainer_reset(self, trainer):
        trainer.reset()

    def test_esd_rtrl_trainer_device_transfer(self, trainer):
        trainer = trainer.to("cpu")
        data = torch.randn(5, 4, 8)
        target = torch.randint(0, 4, (4,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.device.type == "cpu"


class TestTPTrainer:
    """Test TPTrainer class (Trace Propagation)."""

    @pytest.fixture
    def network(self):
        """Create an FCNetwork for TP."""
        return FCNetwork(layer_sizes=[784, 100, 10], beta=0.98)

    @pytest.fixture
    def trainer(self, network):
        from trainers.tp_trainer import TPTrainer

        return TPTrainer(
            network=network,
            lr=0.001,
            batch_size=32,  # Must be >= 2
            alpha=0.77,
            beta=0.98,
            vth=0.66,
            use_optimizer=True,
        )

    def test_tp_trainer_smoke(self, trainer):
        """Smoke test: run train_sample, check loss and pred shape."""
        T, B, F = 10, 32, 784  # B >= 2 required
        data = torch.randn(T, B, F)
        target = torch.randint(0, 10, (B,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.dim() == 0
        assert pred.shape == (B,)
        assert not torch.isnan(loss)
        assert pred.min() >= 0
        assert pred.max() <= 9

    def test_tp_trainer_reset(self, trainer):
        """Test that reset runs without error."""
        trainer.reset()
        # New implementation is stateless across batches (except network state)
        # So just ensure it runs.

    def test_tp_trainer_device_transfer(self, trainer):
        """Test moving trainer to CPU."""
        trainer = trainer.to("cpu")
        # Check S matrix is on cpu
        assert trainer.S.weight.device.type == "cpu"

        data = torch.randn(5, 8, 784)  # B=8 >= 2
        target = torch.randint(0, 10, (8,))
        loss, pred = trainer.train_sample(data, target)
        assert loss.device.type == "cpu"

    def test_tp_trainer_weights_change(self, network):
        """Test that training modifies weights."""
        from trainers.tp_trainer import TPTrainer

        trainer = TPTrainer(
            network=network,
            lr=0.01,
            batch_size=16,  # Must be >= 2
        )
        initial_weights = network.layers[0].weight.data.clone()

        data = torch.randn(5, 16, 784)
        target = torch.randint(0, 10, (16,))
        trainer.train_sample(data, target)

        assert not torch.allclose(initial_weights, network.layers[0].weight.data)

    def test_tp_trainer_has_target_propagator(self, trainer):
        """Test that trainer has target propagator layer S."""
        assert hasattr(trainer, "S")
        assert isinstance(trainer.S, torch.nn.Linear)
        # Target propagator maps n_classes to first hidden size
        assert trainer.S.in_features == 10  # n_classes
        assert trainer.S.out_features == 100  # first hidden

    def test_tp_trainer_batch_size_check(self, network):
        """Test that batch size < 2 raises error."""
        from trainers.tp_trainer import TPTrainer

        with pytest.raises(ValueError, match="TP requires batch_size >= 2"):
            TPTrainer(network=network, lr=0.001, batch_size=1)
