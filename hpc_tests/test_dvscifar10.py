#!/usr/bin/env python3
"""
Quick test script for DVSCifar10Loader.
Tests basic functionality: can we load packages and get a few batches?
"""
import sys
from pathlib import Path

# Add src to path to import datasets
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

def test_dvscifar10_loader():
    print("=" * 60)
    print("Testing DVSCifar10Loader")
    print("=" * 60)

    try:
        from datasets.dvscifar10_loader import DVSCifar10Loader
        print("[✓] Successfully imported DVSCifar10Loader")
    except ImportError as e:
        print(f"[✗] Failed to import DVSCifar10Loader: {e}")
        return False

    try:
        import torch
        print(f"[✓] PyTorch available: {torch.__version__}")
    except ImportError as e:
        print(f"[✗] PyTorch not available: {e}")
        return False

    try:
        import tonic
        print(f"[✓] Tonic available: {tonic.__version__}")
    except ImportError as e:
        print(f"[✗] Tonic not available: {e}")
        return False

    try:
        from tonic.datasets import CIFAR10DVS
        print("[✓] CIFAR10DVS available from tonic")
    except ImportError as e:
        print(f"[✗] CIFAR10DVS not available: {e}")
        return False

    print("\n" + "=" * 60)
    print("Creating DataLoaders (batch_size=4, T=10)...")
    print("=" * 60)

    try:
        trainloader, testloader = DVSCifar10Loader(
            batch_size=4,
            T=10,
            pin_memory=False,
            seed=42,
            num_workers=0,  # Use 0 workers for faster startup
            download=True,
            use_cache=False,  # Skip cache for faster test
            train_fraction=0.9,
        )
        print("[✓] Successfully created train/test DataLoaders")
    except Exception as e:
        print(f"[✗] Failed to create DataLoaders: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("Loading 3 batches from trainloader...")
    print("=" * 60)

    try:
        for batch_idx, (data, target) in enumerate(trainloader):
            if batch_idx >= 3:
                break

            print(f"\nBatch {batch_idx + 1}:")
            print(f"  data shape:   {data.shape}  (expected: [T=10, B=4, 2, 128, 128])")
            print(f"  target shape: {target.shape}  (expected: [B=4])")
            print(f"  data dtype:   {data.dtype}")
            print(f"  target dtype: {target.dtype}")
            print(f"  target values: {target.tolist()}")
            print(f"  data min/max:  [{data.min():.3f}, {data.max():.3f}]")

        print("\n[✓] Successfully loaded batches")
    except Exception as e:
        print(f"[✗] Failed to load batches: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_dvscifar10_loader()
    sys.exit(0 if success else 1)
