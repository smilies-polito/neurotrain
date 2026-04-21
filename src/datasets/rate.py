# datasets/rate.py
from __future__ import annotations
import torch
from torch.utils.data import default_collate
from snntorch import spikegen


class Rate:
    """Rate-code a static tensor in [0,1] into spikes [T, ...] (time-major)."""
    def __init__(self, T: int):
        self.T = T

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is typically [C,H,W] for images
        # x is typically [D] for fully-connected layers
        return spikegen.rate(x, num_steps=self.T)


class DirectCoding:
    """Temporal expansion for SNN direct (analog) coding.

    Replicates a normalised frame across T timesteps, producing [T, C, H, W].
    The SNN processes the same continuous values at every timestep; temporal
    integration is handled by the network's membrane dynamics rather than by
    stochastic input encoding.

    Args:
        T (int): Number of timesteps to repeat the frame across.
    """
    def __init__(self, T: int) -> None:
        self.T = T

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0).repeat(self.T, 1, 1, 1)


def time_major_collate(batch):
    """Collate a batch and transpose data from [B, T, ...] -> [T, B, ...]."""
    data, targets = default_collate(batch)
    return data.transpose(0, 1), targets