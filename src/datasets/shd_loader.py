from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from tonic import transforms as tonic_transforms
from tonic.datasets import SHD


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"


def _seed_worker(worker_id: int) -> None:
    """
    Seed each DataLoader worker process deterministically.

    torch.initial_seed() is set by the DataLoader using its generator (if provided).
    We derive a 32-bit seed from it and seed numpy + random so everything is aligned.
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# SHD (Spiking Heidelberg Digits) is a neuromorphic audio dataset for spoken-digit
# recognition. It contains utterances of digits 0–9 converted into spike trains,
# typically using 700 input channels that represent a cochlea-like auditory encoding.
# Each sample is an event sequence rather than a dense image: spikes are defined by
# neuron/channel indices and spike times over a fixed recording window (commonly 1 s).
# In practice, the data is often represented either as:
#   - sparse events: (times, units), one variable-length spike list per sample, or
#   - dense tensors after binning: [T, 700], where T is the number of time bins.
# Labels are single digit classes in the range 0–9.
def SHDLoader(
    batch_size: int,
    T: int,
    pin_memory: bool = False,
    seed: int | None = None,
    num_workers: int = 4,
    data_root: str | Path | None = None,
    download: bool = True,  # kept for API consistency; tonic downloads as needed
):
    """
    Build train/test DataLoaders for SHD with fixed-bin spike-count frames.

    Output batch shapes:
      - data:   [B, T, 700]
      - target: [B]

    Notes:
      - SHD samples are native spike trains over 700 cochlear channels.
      - Like the other event loaders in this repo, we discretize them into a fixed
        number of bins `T` so batches stay rectangular.
      - We clamp counts to [0, 1] to keep inputs binary per bin/channel.
    """
    del download

    if data_root is None:
        data_root = os.environ.get("SHD_ROOT", str(DEFAULT_DATA_ROOT))

    sensor_size = SHD.sensor_size  # (700, 1, 1)

    def to_torch_and_postprocess(frames: np.ndarray) -> torch.Tensor:
        # ToFrame -> [T, 1, 700, 1] for SHD. Squeeze singleton dims to [T, 700].
        x = torch.from_numpy(frames).to(torch.float32)
        x = x.clamp_(0, 1).squeeze(1).squeeze(-1)
        return x

    transform = tonic_transforms.Compose(
        [
            tonic_transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T),
            to_torch_and_postprocess,
        ]
    )

    train_ds = SHD(save_to=str(Path(data_root) / "SHD"), train=True, transform=transform)
    test_ds = SHD(save_to=str(Path(data_root) / "SHD"), train=False, transform=transform)

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
    )

    return trainloader, testloader
