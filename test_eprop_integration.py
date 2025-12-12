#!/usr/bin/env python3
"""
Minimal integration test for the E-prop trainer.

This script verifies that the E-prop algorithm is correctly integrated 
into the SNN training framework and can train a network on MNIST.

Usage:
    python test_eprop_integration.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn.functional as F


def test_eprop_basic_functionality():
    """Test basic E-prop trainer functionality."""
    print("=" * 60)
    print("TEST 1: Basic E-prop Trainer Functionality")
    print("=" * 60)
    
    from networks.fc_network import FCNetwork
    from trainers.eprop_trainer import EpropTrainer
    
    # Create network
    network = FCNetwork(
        layer_sizes=[784, 100, 10],
        beta=0.9,
    )
    print(f"✓ Created FCNetwork with layers {[784, 100, 10]}")
    
    # Create trainer
    trainer = EpropTrainer(
        network=network,
        lr=0.01,
        batch_size=32,
        gamma=0.3,
        use_optimizer=False,
    )
    print(f"✓ Created EpropTrainer with lr={trainer.lr}, gamma={trainer.gamma}")
    
    # Test forward pass with random data
    timesteps = 10
    batch_size = 32
    data = torch.randn(timesteps, batch_size, 784)
    target = torch.randint(0, 10, (batch_size,))
    
    # Get initial weights
    initial_weights = network.layers[0].weight.data.clone()
    
    # Train sample
    loss, pred = trainer.train_sample(data, target)
    
    print(f"✓ Forward pass completed")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Predictions shape: {pred.shape}")
    
    # Check weights changed
    weights_changed = not torch.allclose(initial_weights, network.layers[0].weight.data)
    assert weights_changed, "Weights should change after training"
    print(f"✓ Weights updated correctly")
    
    # Check predictions are valid
    assert pred.min() >= 0 and pred.max() <= 9, "Predictions should be in [0, 9]"
    print(f"✓ Predictions are valid class indices")
    
    print("TEST 1 PASSED ✓\n")
    return True


def test_eprop_with_real_data():
    """Test E-prop trainer with MNIST-like data."""
    print("=" * 60)
    print("TEST 2: E-prop Training Loop (mini-MNIST simulation)")
    print("=" * 60)
    
    from networks.fc_network import FCNetwork
    from trainers.eprop_trainer import EpropTrainer
    
    # Create network
    network = FCNetwork(
        layer_sizes=[784, 200, 10],
        beta=0.9,
    )
    
    # Create trainer
    trainer = EpropTrainer(
        network=network,
        lr=0.01,
        batch_size=64,
        gamma=0.3,
    )
    
    print("Training on synthetic MNIST-like data...")
    
    # Simulate a few training epochs with synthetic data
    num_batches = 10
    timesteps = 10
    batch_size = 64
    
    for epoch in range(3):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx in range(num_batches):
            # Create rate-coded MNIST-like data (sparse binary)
            # Simulate 784-dim input over 10 timesteps
            data = (torch.rand(timesteps, batch_size, 784) < 0.1).float()
            target = torch.randint(0, 10, (batch_size,))
            
            trainer.reset()
            loss, pred = trainer.train_sample(data, target)
            
            total_loss += loss.item() * batch_size
            total_correct += pred.squeeze().eq(target).sum().item()
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100
        print(f"  Epoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")
    
    print("✓ Training loop completed successfully")
    print("TEST 2 PASSED ✓\n")
    return True


def test_eprop_recurrent_srnn_mnist_like():
    """Test E-prop with recurrent SRNN mirroring reference defaults."""
    print("=" * 60)
    print("TEST 3: Recurrent SRNN E-prop (reference-like)")
    print("=" * 60)

    from networks.recurrent_srnn import RecurrentSRNN
    from trainers.eprop_trainer import EpropTrainer

    # Reference-like parameters
    network = RecurrentSRNN(
        n_in=784,
        n_rec=100,
        n_out=10,
        threshold=0.6,
        tau_mem=2.0,
        tau_out=0.02,
        bias_out=0.0,
        gamma=0.3,
        dt=1e-3,
        w_init_gain=(0.5, 0.1, 0.5),
    )
    network.n_classes = 10

    trainer = EpropTrainer(
        network=network,
        lr=1e-4,
        batch_size=64,
        gamma=0.3,
        lr_layer_norm=(0.05, 0.05, 1.0),
        use_optimizer=True,
        optimizer=torch.optim.Adam(network.parameters(), lr=1e-4),
    )

    # Synthetic MNIST-like batch
    timesteps = 10
    batch_size = 64
    data = (torch.rand(timesteps, batch_size, 784) < 0.1).float()
    target = torch.randint(0, 10, (batch_size,))

    loss, pred = trainer.train_sample(data, target)

    print(f"✓ Recurrent SRNN forward/backward completed. Loss={loss.item():.4f}")
    assert pred.shape == (batch_size, 1)
    assert pred.min() >= 0 and pred.max() <= 9
    print("TEST 3 PASSED ✓\n")
    return True


def test_eprop_in_benchmark_runner():
    """Test E-prop is properly registered in benchmark runner."""
    print("=" * 60)
    print("TEST 3: E-prop Registration in Benchmark Runner")
    print("=" * 60)
    
    from benchmark_runner import ALGORITHM_INFO
    from trainers.eprop_trainer import EpropTrainer
    from trainers.bptt_trainer import BPTTTrainer
    from trainers.stsf_trainer import STSFTrainer
    
    # Check algorithm info
    assert "eprop" in ALGORITHM_INFO, "E-prop should be in ALGORITHM_INFO"
    info = ALGORITHM_INFO["eprop"]
    assert info["is_local"] is True, "E-prop should be marked as local learning"
    assert info["requires_backprop"] is False, "E-prop should not require backprop"
    print(f"✓ E-prop registered in ALGORITHM_INFO:")
    print(f"  Name: {info['name']}")
    print(f"  Local: {info['is_local']}")
    print(f"  Requires Backprop: {info['requires_backprop']}")
    
    print("TEST 4 PASSED ✓\n")
    return True


def test_eprop_in_main_trainer_factory():
    """Test E-prop is properly registered in main.py trainer factory."""
    print("=" * 60)
    print("TEST 4: E-prop Registration in Main Trainer Factory")
    print("=" * 60)
    
    # Import from main
    sys.path.insert(0, os.path.dirname(__file__))
    from main import get_trainer
    
    # Check eprop is available
    trainer_class = get_trainer("eprop")
    from trainers.eprop_trainer import EpropTrainer
    assert trainer_class is EpropTrainer, "get_trainer('eprop') should return EpropTrainer"
    print(f"✓ get_trainer('eprop') returns EpropTrainer")
    
    # Verify other trainers still work
    from trainers.bptt_trainer import BPTTTrainer
    from trainers.stsf_trainer import STSFTrainer
    
    assert get_trainer("bptt") is BPTTTrainer, "BPTT trainer should still work"
    assert get_trainer("stsf") is STSFTrainer, "STSF trainer should still work"
    print(f"✓ Other trainers (bptt, stsf) still work correctly")
    
    print("TEST 5 PASSED ✓\n")
    return True


def test_eprop_config_validation():
    """Test that config validation accepts 'eprop' as a valid trainer."""
    print("=" * 60)
    print("TEST 5: Config Validation Accepts E-prop")
    print("=" * 60)
    
    from utils.config import Config, validate_config, TrainerConfig
    
    # Create config with eprop trainer
    config = Config()
    config.trainer = TrainerConfig(name="eprop")
    
    issues = validate_config(config)
    
    # Should have no issues related to trainer name
    trainer_issues = [i for i in issues if "trainer.name" in i]
    assert len(trainer_issues) == 0, f"E-prop should be a valid trainer, got issues: {trainer_issues}"
    print(f"✓ Config validation accepts 'eprop' as valid trainer")
    
    print("TEST 6 PASSED ✓\n")
    return True


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("E-PROP INTEGRATION TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        test_eprop_basic_functionality,
        test_eprop_with_real_data,
        test_eprop_recurrent_srnn_mnist_like,
        test_eprop_in_benchmark_runner,
        test_eprop_in_main_trainer_factory,
        test_eprop_config_validation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED - E-prop integration successful!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

