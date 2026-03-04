from pathlib import Path
import os
import random

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

from datasets.rate import Rate  # your rate-coding transform (returns [T, C, H, W] per sample)


# Default on-disk location for the dataset (relative to the repo layout).
# You can override this at runtime with the environment variable CIFAR10_ROOT.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"


def _seed_worker(worker_id: int) -> None:
    """
    Seed each DataLoader worker process deterministically.

    Why this matters:
      - With num_workers > 0, PyTorch spawns subprocesses.
      - Each worker has its own RNG state.
      - If you don't seed workers, runs can differ between executions
        (especially important for stochastic transforms like Rate coding).

    torch.initial_seed() is set by the DataLoader using its generator (if provided).
    We derive a 32-bit seed from it and seed numpy + random so everything is aligned.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def CIFAR10Loader(
    batch_size,
    T,
    pin_memory: bool = False,
    seed=None,
    num_workers: int = 4,
    data_root=None,
    download: bool = True,
):
    """
    Build train/test DataLoaders for CIFAR-10 with rate-coded spike trains.

    Output batch shapes:
      - data:   [B, T, 3, 32, 32]   (because Rate(T) produces [T,3,32,32] per sample)
      - target: [B]

    Notes:
      - We keep images in RGB and do NOT flatten here.
        Models that need flattening (e.g., FC networks) should do it inside forward().
      - Rate coding is stochastic; setting `seed` makes shuffling and worker RNG repeatable.
    """
    # Allow users/HPC scripts to relocate datasets without editing code
    if data_root is None:
        data_root = os.environ.get("CIFAR10_ROOT", str(DEFAULT_DATA_ROOT))

    # Transform pipeline:
    #  1) ToTensor: converts PIL image -> torch.Tensor in [0,1], shape [C,H,W]
    #  2) Rate(T):  converts static image -> spike train, shape [T,C,H,W]
    transform = Compose([ToTensor(), Rate(T)])

    # If a seed is provided, use it to make:
    #  - train shuffle order deterministic (via generator)
    #  - stochastic Rate coding deterministic across dataloader workers (via worker_init_fn)
    g = None
    worker_init_fn = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        worker_init_fn = _seed_worker

    # Training loader: shuffled each epoch
    trainloader = DataLoader(
        CIFAR10(str(data_root), train=True, download=download, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,           # useful when transferring batches to GPU
        generator=g,                     # controls deterministic shuffling
        worker_init_fn=worker_init_fn,   # controls deterministic RNG inside workers
        persistent_workers=(num_workers > 0),  # reuse workers between epochs for speed
    )

    # Test loader: no shuffle
    testloader = DataLoader(
        CIFAR10(str(data_root), train=False, download=download, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,   # keeps transform RNG deterministic if seed was set
        persistent_workers=(num_workers > 0),
    )

    return trainloader, testloader