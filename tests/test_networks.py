"""Tests for neural network architectures."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from networks.etlp_network import ETLPNetwork
from networks.fc_network import FCNetwork
from networks.recurrent_srnn import RecurrentSRNN


class TestFCNetwork:
    """Test FCNetwork class."""

    def test_network_creation(self):
        """Test creating a basic FCNetwork."""
        network = FCNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
        )

        assert network.input_size == 784
        assert network.hidden_size == [100]
        assert network.n_classes == 10

    def test_network_multiple_hidden_layers(self):
        """Test network with multiple hidden layers."""
        network = FCNetwork(
            layer_sizes=[784, 256, 128, 64, 10],
            beta=0.9,
        )

        assert network.input_size == 784
        assert network.hidden_size == [256, 128, 64]
        assert network.n_classes == 10

    def test_network_forward_shape(self):
        """Test forward pass output shapes."""
        network = FCNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
        )

        batch_size = 32
        x = torch.randn(batch_size, 784)

        spk_rec, mem_rec = network(x)

        # Should have one output per layer (hidden + output)
        assert len(spk_rec) == 2  # 1 hidden + 1 output
        assert len(mem_rec) == 2

        # Check output layer shape
        assert spk_rec[-1].shape == (batch_size, 10)
        assert mem_rec[-1].shape == (batch_size, 10)

        # Check hidden layer shape
        assert spk_rec[0].shape == (batch_size, 100)

    def test_network_reset(self):
        """Test network state reset."""
        network = FCNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
        )

        # Run a forward pass
        x = torch.randn(1, 784)
        network(x)

        # Reset should not raise
        network.reset()

    def test_network_parameters(self):
        """Test that network has trainable parameters."""
        network = FCNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
        )

        params = list(network.parameters())
        assert len(params) > 0

        # Check total parameter count
        total_params = sum(p.numel() for p in params)
        # 784*100 + 100*10 = 78400 + 1000 = 79400
        assert total_params == 784 * 100 + 100 * 10

    def test_network_no_bias(self):
        """Test that network layers have no bias."""
        network = FCNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
        )

        for layer in network.layers:
            if hasattr(layer, "bias"):
                assert layer.bias is None

    def test_network_device_transfer(self):
        """Test moving network to different devices."""
        network = FCNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
        )

        # Move to CPU explicitly
        network = network.to("cpu")
        x = torch.randn(1, 784)
        spk, mem = network(x)

        assert spk[-1].device.type == "cpu"

    def test_network_train_eval_mode(self):
        """Test train/eval mode switching."""
        network = FCNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
        )

        network.train()
        assert network.training

        network.eval()
        assert not network.training

    def test_network_state_dict(self):
        """Test saving and loading state dict."""
        network1 = FCNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
        )

        # Save state
        state = network1.state_dict()

        # Create new network and load state
        network2 = FCNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
        )
        network2.load_state_dict(state)

        # Check weights match
        for (n1, p1), (n2, p2) in zip(
            network1.named_parameters(), network2.named_parameters()
        ):
            assert torch.allclose(p1, p2)

    def test_network_quantized(self):
        """Test network with quantization flag."""
        network = FCNetwork(
            layer_sizes=[784, 100, 10],
            beta=0.9,
            quant=True,
        )

        assert network.quant is True

    def test_network_different_betas(self):
        """Test network with different beta values."""
        for beta in [0.5, 0.75, 0.9, 0.99]:
            network = FCNetwork(
                layer_sizes=[784, 100, 10],
                beta=beta,
            )

            x = torch.randn(1, 784)
            spk, mem = network(x)

            # Should produce valid outputs
            assert not torch.isnan(spk[-1]).any()
            assert not torch.isnan(mem[-1]).any()


class TestRecurrentSRNN:
    """Test RecurrentSRNN class."""

    def test_network_creation(self):
        network = RecurrentSRNN(n_in=8, n_rec=16, n_out=4)
        assert network.n_in == 8
        assert network.n_rec == 16
        assert network.n_out == 4

    def test_network_forward_shape(self):
        network = RecurrentSRNN(n_in=8, n_rec=16, n_out=4)
        x = torch.randn(32, 8)
        spk_rec, mem_rec = network(x)
        assert len(spk_rec) == 1
        assert spk_rec[-1].shape == (32, 4)
        assert mem_rec[-1].shape == (32, 4)

    def test_network_reset(self):
        network = RecurrentSRNN(n_in=8, n_rec=16, n_out=4)
        x = torch.randn(1, 8)
        network(x)
        network.reset()


class TestETLPNetwork:
    """Test ETLPNetwork class."""

    def test_network_creation(self):
        network = ETLPNetwork(n_in=12, n_rec=6, n_out=3, dt=1.0)
        assert network.n_in == 12
        assert network.n_rec == 6
        assert network.n_out == 3

    def test_network_forward_shape(self):
        network = ETLPNetwork(n_in=12, n_rec=6, n_out=3, dt=1.0)
        batch_size = 4
        x = torch.rand(batch_size, 12)
        spk_rec, mem_rec = network(x)
        assert len(spk_rec) == 2
        assert spk_rec[-1].shape == (batch_size, 3)
        assert mem_rec[-1].shape == (batch_size, 3)

    def test_network_reset(self):
        network = ETLPNetwork(n_in=12, n_rec=6, n_out=3, dt=1.0)
        x = torch.rand(1, 12)
        network(x)
        network.reset()
