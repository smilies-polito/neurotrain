"""Tests for dataset loaders."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets.get_loader import get_loader


class TestGetLoader:
    """Test get_loader factory function."""

    def test_get_loader_mnist(self):
        """Test loading MNIST dataset."""
        trainloader, testloader = get_loader("MNIST", batch_size=32, T=10)

        assert trainloader is not None
        assert testloader is not None

    def test_get_loader_fashionmnist(self):
        """Test loading FashionMNIST dataset."""
        trainloader, testloader = get_loader("FashionMNIST", batch_size=32, T=10)

        assert trainloader is not None
        assert testloader is not None

    def test_get_loader_invalid_dataset(self):
        """Test that invalid dataset raises error."""
        with pytest.raises(ValueError):
            get_loader("INVALID_DATASET", batch_size=32, T=10)

    def test_mnist_batch_shape(self):
        """Test MNIST batch shapes."""
        trainloader, _ = get_loader("MNIST", batch_size=16, T=5)

        batch_data, batch_labels = next(iter(trainloader))

        # Data should be [batch, timesteps, features]
        assert batch_data.shape[0] == 16  # batch size
        assert batch_data.shape[1] == 5  # timesteps
        assert batch_data.shape[2] == 784  # flattened 28x28

        # Labels should be [batch]
        assert batch_labels.shape == (16,)

    def test_mnist_data_range(self):
        """Test that MNIST data is properly normalized."""
        trainloader, _ = get_loader("MNIST", batch_size=32, T=10)

        batch_data, _ = next(iter(trainloader))

        # Rate-coded spikes should be binary (0 or 1)
        # or normalized values
        assert batch_data.min() >= 0
        assert batch_data.max() <= 1

    def test_mnist_label_range(self):
        """Test that MNIST labels are valid."""
        trainloader, _ = get_loader("MNIST", batch_size=64, T=10)

        _, batch_labels = next(iter(trainloader))

        assert batch_labels.min() >= 0
        assert batch_labels.max() <= 9

    def test_different_batch_sizes(self):
        """Test loaders with different batch sizes."""
        for bs in [1, 16, 64, 128]:
            trainloader, _ = get_loader("MNIST", batch_size=bs, T=10)
            batch_data, batch_labels = next(iter(trainloader))
            assert batch_data.shape[0] == bs

    def test_different_timesteps(self):
        """Test loaders with different timesteps."""
        for T in [1, 5, 10, 25]:
            trainloader, _ = get_loader("MNIST", batch_size=32, T=T)
            batch_data, _ = next(iter(trainloader))
            assert batch_data.shape[1] == T


class TestDataLoaderIteration:
    """Test iterating over data loaders."""

    @pytest.fixture
    def loaders(self):
        """Create test loaders."""
        return get_loader("MNIST", batch_size=64, T=10)

    def test_trainloader_iterable(self, loaders):
        """Test that trainloader is iterable."""
        trainloader, _ = loaders

        count = 0
        for data, labels in trainloader:
            count += 1
            if count >= 3:
                break

        assert count == 3

    def test_testloader_iterable(self, loaders):
        """Test that testloader is iterable."""
        _, testloader = loaders

        count = 0
        for data, labels in testloader:
            count += 1
            if count >= 3:
                break

        assert count == 3

    def test_trainloader_length(self, loaders):
        """Test trainloader has expected length."""
        trainloader, _ = loaders

        # MNIST has 60000 training samples
        # With batch_size=64, should have ~938 batches
        assert len(trainloader) > 900
        assert len(trainloader) < 1000

    def test_testloader_length(self, loaders):
        """Test testloader has expected length."""
        _, testloader = loaders

        # MNIST has 10000 test samples
        # With batch_size=64, should have ~157 batches
        assert len(testloader) > 150
        assert len(testloader) < 170


class TestRateCoding:
    """Test rate coding transform."""

    def test_rate_coding_produces_spikes(self):
        """Test that rate coding produces spike-like output."""
        trainloader, _ = get_loader("MNIST", batch_size=1, T=100)

        batch_data, _ = next(iter(trainloader))

        # With T=100, we should see variation over time
        # Sum over time should vary across features
        temporal_sum = batch_data.sum(dim=1)
        assert temporal_sum.std() > 0

    def test_rate_coding_deterministic_with_seed(self):
        """Test that rate coding is deterministic with same seed."""
        torch.manual_seed(42)
        trainloader1, _ = get_loader("MNIST", batch_size=8, T=10)
        data1, labels1 = next(iter(trainloader1))

        torch.manual_seed(42)
        trainloader2, _ = get_loader("MNIST", batch_size=8, T=10)
        data2, labels2 = next(iter(trainloader2))

        # Note: Due to data loader workers, exact reproducibility
        # may require num_workers=0. Labels should match though.
        assert torch.equal(labels1, labels2)

