from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from tonic.datasets import CIFAR10DVS
from tonic import transforms as tonic_transforms, DiskCachedDataset

from datasets.rate import time_major_collate


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"


def _seed_worker(worker_id: int) -> None:
    """
    Seed each DataLoader worker process deterministically.

    torch.initial_seed() is set by the DataLoader using its generator (if provided).
    We derive a 32-bit seed from it and seed numpy + random so everything is aligned.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# DVS-CIFAR10 (a.k.a. CIFAR10-DVS) is the neuromorphic counterpart of CIFAR-10.
# It was recorded by pointing a DVS camera at CIFAR-10 images displayed on a monitor
# while the camera was moved in smooth circles, generating ~9000 samples across 10
# classes. Each sample is a raw event stream; we bin it into T frames of shape
# [2, 128, 128] (ON/OFF polarities at the original 128×128 DVS resolution).
#
# NOTE: tonic does NOT split the dataset into train/test sets — all 9000 samples are
# in a single pool. We apply a deterministic 90/10 split (stratification not guaranteed).
# Override `train_fraction` if you prefer a different split ratio.
def DVSCifar10Loader(
    batch_size: int,
    T: int,
    pin_memory: bool = False,
    seed: int | None = None,
    num_workers: int = 4,
    data_root: str | Path | None = None,
    download: bool = True,  # kept for API consistency; tonic downloads as needed
    use_cache: bool = True,  # cache preprocessed frames alongside the dataset
    train_fraction: float = 0.9,  # fraction of data used for training
):
    """
    Build train/test DataLoaders for DVS-CIFAR10 (CIFAR10-DVS) with binned event frames.

    Output batch shapes:
      - data:   [T, B, 2, 128, 128]   (2 channels = ON/OFF polarities)
      - target: [B]

    Notes:
      - tonic's CIFAR10DVS has no official train/test split; we do a random 90/10 split.
        Pass `seed` to make this split reproducible.
      - We do NOT flatten here; models that need flattened inputs should do it in forward().
      - We clamp counts to [0, 1] (binary activity per polarity). Remove clamp_ to use counts.
      - Caching is enabled by default; set use_cache=False to disable.
    """
    if data_root is None:
        data_root = os.environ.get("DVSCIFAR10_ROOT", str(DEFAULT_DATA_ROOT))

    sensor_size = CIFAR10DVS.sensor_size  # (128, 128, 2)

    def to_torch_and_postprocess(frames: np.ndarray) -> torch.Tensor:
        # ToFrame -> [T, 2, 128, 128] (counts per polarity)
        x = torch.from_numpy(frames).to(torch.float32)
        x = x.clamp_(0, 1)  # binarize activity per polarity (optional)
        return x

    transform = tonic_transforms.Compose(
        [
            tonic_transforms.Denoise(filter_time=10000),
            tonic_transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T),
            to_torch_and_postprocess,
        ]
    )

    # Bypass Figshare's bot protection by pointing directly to the S3 backend
    CIFAR10DVS.url = "https://ndownloader.figshare.com/files/38023437"

    full_ds = CIFAR10DVS(
        save_to=str(Path(data_root) / "DVSCIFAR10"),
        transform=transform,
    )

    # Wrap with disk cache before splitting so both subsets share the same cache.
    if use_cache:
        cache_path = Path(data_root) / "DVSCIFAR10" / "cache" / f"T{T}"
        full_ds = DiskCachedDataset(full_ds, cache_path=str(cache_path))

    # Deterministic train/test split.
    n_total = len(full_ds)
    n_train = int(n_total * train_fraction)
    n_test = n_total - n_train

    split_generator = torch.Generator().manual_seed(seed if seed is not None else 0)
    train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=split_generator)

    g = None
    worker_init_fn = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        worker_init_fn = _seed_worker

    trainloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
        collate_fn=time_major_collate,
    )

    testloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
        collate_fn=time_major_collate,
    )

    return trainloader, testloader
