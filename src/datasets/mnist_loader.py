from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from datasets.rate import Rate, time_major_collate  # returns [T, ...] per sample (time-major)


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"


def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker process deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def MNISTLoader(
    batch_size: int,
    T: int,
    pin_memory: bool = False,
    seed: int | None = None,
    num_workers: int = 4,
    data_root: str | Path | None = None,
    download: bool = True,
):
    """
    Minimal MNIST loader for SNNs using snnTorch rate coding (spikegen.rate).

    Per-sample:
      - ToTensor() -> x in [0,1], shape [1,28,28]
      - Rate(T) -> spikes [T, 1, 28, 28]  (time-major)

    Batched output shapes:
      - data:   [T, B, 1, 28, 28]
      - target: [B]
    """
    if data_root is None:
        data_root = os.environ.get("MNIST_ROOT", str(DEFAULT_DATA_ROOT))

    transform = Compose([ToTensor(), Rate(T)])

    g = None
    worker_init_fn = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        worker_init_fn = _seed_worker

    trainloader = DataLoader(
        MNIST(str(data_root), train=True, download=download, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
        collate_fn=time_major_collate,
    )

    testloader = DataLoader(
        MNIST(str(data_root), train=False, download=download, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
        collate_fn=time_major_collate,
    )

    return trainloader, testloader