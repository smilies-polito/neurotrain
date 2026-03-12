from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from tonic.datasets import DVSGesture
from tonic import transforms as tonic_transforms

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


def DVSGestureLoader(
    batch_size: int,
    T: int,
    pin_memory: bool = False,
    seed: int | None = None,
    num_workers: int = 4,
    data_root: str | Path | None = None,
    download: bool = True,  # kept for API consistency; tonic downloads as needed
):
    """
    Build train/test DataLoaders for DVS Gesture with binned event frames (2 polarities).

    Output batch shapes:
      - data:   [T, B, 2, 128, 128]   (2 channels = ON/OFF polarities)
      - target: [B]

    Notes:
      - We do NOT flatten here; models that need flattened inputs should do it in forward().
      - Setting `seed` makes train shuffle and worker RNG deterministic.
      - We clamp counts to [0,1] (binary activity per polarity). Remove clamp_ if you want counts.
    """
    if data_root is None:
        data_root = os.environ.get("DVSGESTURE_ROOT", str(DEFAULT_DATA_ROOT))

    sensor_size = DVSGesture.sensor_size  # typically (128, 128, 2)

    def to_torch_and_postprocess(frames: np.ndarray) -> torch.Tensor:
        # ToFrame -> [T, 2, 128, 128] (counts per polarity)
        x = torch.from_numpy(frames).to(torch.float32)
        x = x.clamp_(0, 1)  # optional binarization
        return x

    transform = tonic_transforms.Compose(
        [
            tonic_transforms.Denoise(filter_time=10000),
            tonic_transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T),
            to_torch_and_postprocess,
        ]
    )

    train_ds = DVSGesture(
        save_to=str(Path(data_root) / "DVSGesture"),
        train=True,
        transform=transform,
    )
    test_ds = DVSGesture(
        save_to=str(Path(data_root) / "DVSGesture"),
        train=False,
        transform=transform,
    )

    g = None
    worker_init_fn = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        worker_init_fn = _seed_worker

    trainloader = DataLoader(
        train_ds,
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
        test_ds,
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