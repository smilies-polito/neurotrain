from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Normalize, RandomCrop, ToTensor

from datasets.rate import Rate, DirectCoding, time_major_collate


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

# Per-channel mean and std computed over the Fashion-MNIST training set.
_FMNIST_MEAN = (0.2860,)
_FMNIST_STD  = (0.3530,)


def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker process deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def FashionMNISTLoader(
    batch_size: int,
    T: int,
    pin_memory: bool = False,
    seed: int | None = None,
    num_workers: int = 4,
    data_root: str | Path | None = None,
    download: bool = True,
    direct_coding: bool = False,
):
    """
    Build train/test DataLoaders for Fashion-MNIST with a choice of input encoding.

    Args:
        batch_size (int): Number of samples per batch.
        T (int): Number of SNN timesteps.
        pin_memory (bool): Pin host memory for faster GPU transfers.
        seed (int | None): If set, makes shuffling and worker RNG reproducible.
        num_workers (int): Subprocesses used for data loading.
        data_root (str | Path | None): Dataset root directory.  Falls back to
            the ``FASHIONMNIST_ROOT`` environment variable, then ``src/Data/``.
        download (bool): Download Fashion-MNIST if not already present.
        direct_coding (bool): Select the input encoding scheme.

            * ``False`` (default) — **Rate coding** via
              ``[ToTensor, Normalize, Rate(T)]``.
            * ``True`` — **Direct coding**.  Normalised pixel values are
              repeated across T timesteps via ``DirectCoding(T)``.  Train
              pipeline adds ``RandomCrop`` for augmentation.
              ``RandomHorizontalFlip`` is intentionally omitted — while some
              classes tolerate flipping (T-shirt), others do not (boot, bag).
              Omitting it is the safe, consistent choice across all classes.

    Output batch shapes (both modes):
        - data:   ``[T, B, 1, 28, 28]``
        - target: ``[B]``

    Notes:
        - Fashion-MNIST is grayscale; images remain [1, 28, 28].
          Models that need flattening should do it inside ``forward()``.
    """
    if data_root is None:
        data_root = os.environ.get("FASHIONMNIST_ROOT", str(DEFAULT_DATA_ROOT))

    if direct_coding:
        train_transform = Compose([
            RandomCrop(28, padding=4),
            ToTensor(),
            Normalize(_FMNIST_MEAN, _FMNIST_STD),
            DirectCoding(T),
        ])
        test_transform = Compose([
            ToTensor(),
            Normalize(_FMNIST_MEAN, _FMNIST_STD),
            DirectCoding(T),
        ])
    else:
        train_transform = test_transform = Compose([
            ToTensor(),
            Normalize(_FMNIST_MEAN, _FMNIST_STD),
            Rate(T),
        ])

    g = None
    worker_init_fn = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        worker_init_fn = _seed_worker

    trainloader = DataLoader(
        FashionMNIST(str(data_root), train=True, download=download, transform=train_transform),
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
        FashionMNIST(str(data_root), train=False, download=download, transform=test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
        collate_fn=time_major_collate,
    )

    return trainloader, testloader