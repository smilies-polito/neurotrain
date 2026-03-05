from pathlib import Path
import os
import random

import numpy as np
import torch
from torchvision.datasets import SVHN
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

from datasets.rate import Rate  # your rate-coding transform (returns [T, C, H, W] per sample)


# Default on-disk location for the dataset (relative to the repo layout).
# You can override this at runtime with the environment variable CIFAR10_ROOT.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"


def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker process deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def SVHNLoader(
    batch_size: int,
    T: int,
    pin_memory: bool = False,
    seed: int | None = None,
    num_workers: int = 4,
    data_root: str | Path | None = None,
    download: bool = True,
):
    """
    Build train/test DataLoaders for SVHN with rate-coded spike trains.

    Output batch shapes:
      - data:   [B, T, 3, 32, 32]
      - target: [B]

    Notes:
      - We keep images in RGB and do NOT flatten here.
      - Rate coding is stochastic; setting `seed` makes shuffling and worker RNG repeatable.
    """
    if data_root is None:
        data_root = os.environ.get("SVHN_ROOT", str(DEFAULT_DATA_ROOT))

    transform = Compose([ToTensor(), Rate(T)])

    g = None
    worker_init_fn = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        worker_init_fn = _seed_worker

    trainloader = DataLoader(
        SVHN(str(data_root), split="train", download=download, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
    )

    testloader = DataLoader(
        SVHN(str(data_root), split="test", download=download, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
    )

    return trainloader, testloader

# NOTE (SVHN): torchvision.SVHN has an additional split called "extra" (more labeled training data).
# Some baselines/papers train on train+extra, so results may not be apples-to-apples if we use only "train".
# If you want to include it later, build the training dataset like this:
#
#   from torch.utils.data import ConcatDataset
#   train_ds = ConcatDataset([
#       SVHN(str(data_root), split="train", download=download, transform=transform),
#       SVHN(str(data_root), split="extra", download=download, transform=transform),
#   ])
#
# and then pass `train_ds` to DataLoader instead of SVHN(..., split="train", ...).