from pathlib import Path
import os
import random

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)
from torch.utils.data import DataLoader

from datasets.rate import Rate, DirectCoding, time_major_collate


# Default on-disk location for the dataset (relative to the repo layout).
# You can override this at runtime with the environment variable CIFAR10_ROOT.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

# Per-channel mean and std computed over the CIFAR-10 training set.
# Used to normalise images to approximately zero mean and unit variance.
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def _seed_worker(worker_id: int) -> None:
    """
    Seed each DataLoader worker process deterministically.

    Why this matters:
      - With num_workers > 0, PyTorch spawns subprocesses.
      - Each worker has its own RNG state.
      - If you don't seed workers, runs can differ between executions.
        This affects both stochastic transforms (Rate coding) and random
        augmentations (RandomCrop, RandomHorizontalFlip in direct coding).

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
    direct_coding: bool = True,
):
    """
    Build train/test DataLoaders for CIFAR-10 with a choice of input encoding.

    Args:
        batch_size (int): Number of samples per batch.
        T (int): Number of SNN timesteps.
        pin_memory (bool): Pin host memory for faster GPU transfers.
        seed (int | None): If set, makes shuffling and worker RNG reproducible.
        num_workers (int): Subprocesses used for data loading.
        data_root (str | Path | None): Dataset root directory.  Falls back to
            the ``CIFAR10_ROOT`` environment variable, then ``src/Data/``.
        download (bool): Download CIFAR-10 if not already present.
        direct_coding (bool): Select the input encoding scheme.

            * ``False`` (default) — **Rate coding**.  Images are converted to
              binary Bernoulli spike trains via ``Rate(T)``.  Both train and test
              sets share the same ``[ToTensor, Rate(T)]`` pipeline.

            * ``True`` — **Direct (analog) coding**.  Normalised pixel values
              are repeated across T timesteps via ``DirectCoding(T)``.  The
              train pipeline includes ``RandomCrop`` and ``RandomHorizontalFlip``
              augmentations; the test pipeline uses only normalisation.

    Output batch shapes (both modes):
        - data:   ``[T, B, 3, 32, 32]``  (time-major, via ``time_major_collate``)
        - target: ``[B]``

    Notes:
        - Images are kept in RGB and are NOT flattened here.
          Models that need flattening (e.g., FC networks) should do it inside
          their ``forward()`` method.
        - Rate coding is stochastic; direct coding augmentations are also
          stochastic.  Setting ``seed`` makes both reproducible.
    """
    # Allow users/HPC scripts to relocate datasets without editing code.
    if data_root is None:
        data_root = os.environ.get("CIFAR10_ROOT", str(DEFAULT_DATA_ROOT))

    if direct_coding:
        # Direct coding: continuous normalised values repeated across T timesteps.
        # Separate pipelines for train/test because augmentations only apply during
        # training — applying them at test time would degrade accuracy.
        train_transform = Compose([
            RandomCrop(32, padding=4),      # random 32x32 crop from zero-padded 40x40
            RandomHorizontalFlip(),         # standard left/right flip
            ToTensor(),                     # PIL -> [0,1] float tensor, shape [C,H,W]
            Normalize(_CIFAR10_MEAN, _CIFAR10_STD),  # per-channel normalisation
            DirectCoding(T),                # [C,H,W] -> [T,C,H,W]
        ])
        test_transform = Compose([
            ToTensor(),
            Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
            DirectCoding(T),
        ])
    else:
        # Rate coding: the same stochastic encoding is applied regardless of split.
        # Transform pipeline:
        #  1) ToTensor: converts PIL image -> torch.Tensor in [0,1], shape [C,H,W]
        #  2) Rate(T):  converts static image -> spike train, shape [T,C,H,W]
        train_transform = test_transform = Compose([ToTensor(), Rate(T)])

    # If a seed is provided, use it to make:
    #  - train shuffle order deterministic (via generator)
    #  - stochastic transforms deterministic across dataloader workers (via worker_init_fn)
    g = None
    worker_init_fn = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        worker_init_fn = _seed_worker

    # Training loader: shuffled each epoch
    trainloader = DataLoader(
        CIFAR10(str(data_root), train=True, download=download, transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,           # useful when transferring batches to GPU
        generator=g,                     # controls deterministic shuffling
        worker_init_fn=worker_init_fn,   # controls deterministic RNG inside workers
        persistent_workers=(num_workers > 0),  # reuse workers between epochs for speed
        collate_fn=time_major_collate,
    )

    # Test loader: no shuffle
    testloader = DataLoader(
        CIFAR10(str(data_root), train=False, download=download, transform=test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,   # keeps transform RNG deterministic if seed was set
        persistent_workers=(num_workers > 0),
        collate_fn=time_major_collate,
    )

    return trainloader, testloader