from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import h5py
from tonic import transforms as tonic_transforms, DiskCachedDataset
from tonic.datasets import SHD
from tonic.io import make_structured_array

from datasets.rate import time_major_collate


class _SHDFixed(SHD):
    """Thin SHD subclass that fixes a float16 overflow bug in tonic's HSD.__getitem__.

    tonic stores spike times as float16 (seconds) and multiplies by 1e6 to get
    microseconds *while still in float16*, which overflows for any time > ~0.065 s
    (float16 max ≈ 65504).  SHD recordings last up to ~1 s, so almost all events
    are silently dropped, producing all-zero frames.

    Fix: cast times to float64 before the multiply.

    tonic stores data under ``<save_to>/<ClassName>/``, so passing the same
    ``save_to`` as the original SHD class will resolve to ``<save_to>/_SHDFixed/``.
    If the original ``SHD`` data directory exists, copy or symlink it there once;
    otherwise tonic will download the files automatically on first use.
    """

    def __getitem__(self, index):
        import os
        file = h5py.File(
            os.path.join(self.location_on_system, self.data_filename), "r"
        )
        events = make_structured_array(
            file["spikes/times"][index].astype(np.float64) * 1e6,  # cast BEFORE *1e6
            file["spikes/units"][index],
            1,
            dtype=self.dtype,
        )
        target = file["labels"][index].astype(int)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target


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
    use_cache: bool = True,  # cache preprocessed frames alongside the dataset
):
    """
    Build train/test DataLoaders for SHD with fixed-bin spike-count frames.

    Output batch shapes:
      - data:   [T, B, 700]
      - target: [B]

    Notes:
      - SHD samples are native spike trains over 700 cochlear channels.
      - Like the other event loaders in this repo, we discretize them into a fixed
        number of bins `T` so batches stay rectangular.
      - We clamp counts to [0, 1] to keep inputs binary per bin/channel.
      - Caching is enabled by default; set use_cache=False to disable.
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

    train_ds = _SHDFixed(save_to=str(Path(data_root) / "SHD"), train=True, transform=transform)
    test_ds = _SHDFixed(save_to=str(Path(data_root) / "SHD"), train=False, transform=transform)

    # Wrap with disk cache to avoid re-processing raw events on every access.
    if use_cache:
        cache_path = Path(data_root) / "SHD" / "cache"
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
