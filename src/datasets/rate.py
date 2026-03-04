# datasets/rate.py
from __future__ import annotations
import torch
from snntorch import spikegen

class Rate:
    """Rate-code a static tensor in [0,1] into spikes [T, ...] (time-major)."""
    def __init__(self, T: int):
        self.T = T

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is typically [C,H,W] for images
        # x is typiically [D] for fully-connected layers
        return spikegen.rate(x, num_steps=self.T)