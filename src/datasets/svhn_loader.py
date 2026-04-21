from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch
from torchvision.datasets import SVHN
from torchvision.transforms import Compose, Normalize, RandomCrop, ToTensor
from torch.utils.data import DataLoader

from datasets.rate import Rate, DirectCoding, time_major_collate


# Default on-disk location for the dataset (relative to the repo layout).
# You can override this at runtime with the environment variable SVHN_ROOT.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

# Per-channel mean and std computed over the SVHN training set.
_SVHN_MEAN = (0.4377, 0.4438, 0.4728)
_SVHN_STD  = (0.1980, 0.2010, 0.1970)


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
    direct_coding: bool = True,
):
    """
    Build train/test DataLoaders for SVHN with a choice of input encoding.

    Args:
        batch_size (int): Number of samples per batch.
        T (int): Number of SNN timesteps.
        pin_memory (bool): Pin host memory for faster GPU transfers.
        seed (int | None): If set, makes shuffling and worker RNG reproducible.
        num_workers (int): Subprocesses used for data loading.
        data_root (str | Path | None): Dataset root directory.  Falls back to
            the ``SVHN_ROOT`` environment variable, then ``src/Data/``.
        download (bool): Download SVHN if not already present.
        direct_coding (bool): Select the input encoding scheme.

            * ``False`` (default) — **Rate coding** via ``[ToTensor, Rate(T)]``.
            * ``True`` — **Direct coding**.  Normalised pixel values are
              repeated across T timesteps via ``DirectCoding(T)``.  Train
              pipeline adds ``RandomCrop`` for augmentation.
              ``RandomHorizontalFlip`` is intentionally omitted — flipping
              digit images corrupts class semantics (e.g. "2" ↔ mirror image).

    Output batch shapes (both modes):
        - data:   ``[T, B, 3, 32, 32]``
        - target: ``[B]``
    """
    if data_root is None:
        data_root = os.environ.get("SVHN_ROOT", str(DEFAULT_DATA_ROOT))

    if direct_coding:
        train_transform = Compose([
            RandomCrop(32, padding=4),
            ToTensor(),
            Normalize(_SVHN_MEAN, _SVHN_STD),
            DirectCoding(T),
        ])
        test_transform = Compose([
            ToTensor(),
            Normalize(_SVHN_MEAN, _SVHN_STD),
            DirectCoding(T),
        ])
    else:
        train_transform = test_transform = Compose([ToTensor(), Rate(T)])

    g = None
    worker_init_fn = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        worker_init_fn = _seed_worker

    trainloader = DataLoader(
        SVHN(str(data_root), split="train", download=download, transform=train_transform),
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
        SVHN(str(data_root), split="test", download=download, transform=test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
        collate_fn=time_major_collate,
    )

    return trainloader, testloader

# NOTE (SVHN): torchvision.SVHN has an additional split called "extra" (more labeled training data).
# Some baselines/papers train on train+extra, so results may not be apples-to-apples if we use only "train".
# If you want to include it later, build the training dataset like this:
#
#   from torch.utils.data import ConcatDataset
#   train_ds = ConcatDataset([
#       SVHN(str(data_root), split="train", download=download, transform=train_transform),
#       SVHN(str(data_root), split="extra", download=download, transform=train_transform),
#   ])
#
# and then pass `train_ds` to DataLoader instead of SVHN(..., split="train", ...).