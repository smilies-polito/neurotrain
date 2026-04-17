from pathlib import Path
import os
import random

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose, ToTensor, Normalize,
    RandomCrop, RandomHorizontalFlip,
)
from torch.utils.data import DataLoader


# Default on-disk location for the dataset (relative to the repo layout).
# You can override this at runtime with the environment variable CIFAR10_ROOT.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

# CIFAR-10 per-channel mean and std (computed on training set, RGB order).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def _seed_worker(worker_id: int) -> None:
    """
    Seed each DataLoader worker process deterministically.

    Why this matters:
      - With num_workers > 0, PyTorch spawns subprocesses.
      - Each worker has its own RNG state.
      - If you don't seed workers, runs can differ between executions
        (especially important for stochastic transforms like RandomCrop).

    torch.initial_seed() is set by the DataLoader using its generator (if provided).
    We derive a 32-bit seed from it and seed numpy + random so everything is aligned.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def CIFAR10StaticLoader(
    batch_size,
    pin_memory: bool = False,
    seed=None,
    num_workers: int = 4,
    data_root=None,
    download: bool = True,
):
    """
    Build train/test DataLoaders for CIFAR-10 with standard image preprocessing.

    This loader follows the OTTT-SNN recipe: images are kept as static tensors
    [B, 3, 32, 32]. There is NO rate-coding — the time dimension is handled
    externally by the training loop (same frame fed to the SNN at each timestep).

    Output batch shapes:
      - data:   [B, 3, 32, 32]
      - target: [B]

    Training augmentations: RandomCrop(32, padding=4) + RandomHorizontalFlip.
    Both train and test are normalized with CIFAR-10 per-channel mean/std.

    Notes:
      - Use CIFAR10Loader (rate-coded) instead if your model expects spike trains.
      - Setting `seed` makes shuffle order and worker augmentation RNG reproducible.
    """
    # Allow users/HPC scripts to relocate datasets without editing code
    if data_root is None:
        data_root = os.environ.get("CIFAR10_ROOT", str(DEFAULT_DATA_ROOT))

    # Training pipeline: crop + flip augmentation, then normalize to zero mean/unit std.
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Test pipeline: no augmentation, only normalization.
    transform_test = Compose([
        ToTensor(),
        Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # If a seed is provided, use it to make:
    #  - train shuffle order deterministic (via generator)
    #  - stochastic augmentations deterministic across dataloader workers (via worker_init_fn)
    g = None
    worker_init_fn = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        worker_init_fn = _seed_worker

    # Training loader: shuffled each epoch, with data augmentation
    trainloader = DataLoader(
        CIFAR10(str(data_root), train=True, download=download, transform=transform_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,           # useful when transferring batches to GPU
        generator=g,                     # controls deterministic shuffling
        worker_init_fn=worker_init_fn,   # controls deterministic RNG inside workers
        persistent_workers=(num_workers > 0),  # reuse workers between epochs for speed
    )

    # Test loader: no shuffle, no augmentation
    testloader = DataLoader(
        CIFAR10(str(data_root), train=False, download=download, transform=transform_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,   # keeps augmentation RNG deterministic if seed was set
        persistent_workers=(num_workers > 0),
    )

    return trainloader, testloader
