from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from tonic import transforms as tonic_transforms, DiskCachedDataset
from tonic.datasets import NMNIST

from datasets.rate import time_major_collate


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"


def _seed_worker(worker_id: int) -> None:
    """
    Seed each DataLoader worker process deterministically.
    (Same rationale as your CIFAR10 loader.)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def NMNISTLoader(
    batch_size: int,
    T: int,
    pin_memory: bool = False,
    seed: int | None = None,
    num_workers: int = 4,
    data_root: str | Path | None = None,
    download: bool = True,  # kept for API consistency; tonic downloads as needed
    use_cache: bool = True,  # cache preprocessed frames alongside the dataset
):
    """
    Build train/test DataLoaders for N-MNIST with binned event frames (2 polarities).

    Output batch shapes:
      - data:   [T, B, 2, 34, 34]   (2 channels = ON/OFF polarities)
      - target: [B]

    Notes:
      - We do NOT flatten here; models that need flattened inputs should do it in forward().
      - Setting `seed` makes train shuffle and worker RNG deterministic.
      - We clamp counts to [0, 1] (binary activity per polarity). Remove clamp_ if you want counts.
      - Caching is enabled by default; set use_cache=False to disable.
    """
    if data_root is None:
        data_root = os.environ.get("NMNIST_ROOT", str(DEFAULT_DATA_ROOT))

    sensor_size = NMNIST.sensor_size  # (34, 34, 2)

    def to_torch_and_postprocess(frames: np.ndarray) -> torch.Tensor:
        # ToFrame -> [T, 2, 34, 34] (counts per polarity)
        x = torch.from_numpy(frames).to(torch.float32)  # [T,2,H,W]
        x = x.clamp_(0, 1)  # binarize activity per polarity (optional)
        return x

    transform = tonic_transforms.Compose(
        [
            tonic_transforms.Denoise(filter_time=10000),
            tonic_transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T),
            to_torch_and_postprocess,
        ]
    )

    train_ds = NMNIST(save_to=str(Path(data_root) / "NMNIST"), train=True, transform=transform)
    test_ds = NMNIST(save_to=str(Path(data_root) / "NMNIST"), train=False, transform=transform)

    # Wrap with disk cache to avoid re-processing raw events on every access.
    if use_cache:
        cache_path = Path(data_root) / "NMNIST" / "cache"
        train_ds = DiskCachedDataset(train_ds, cache_path=str(cache_path / f"train_T{T}"))
        test_ds = DiskCachedDataset(test_ds, cache_path=str(cache_path / f"test_T{T}"))

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