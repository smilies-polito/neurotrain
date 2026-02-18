"""
Base class for all SNN models in the framework.

Provides unified interface for evaluation, NeuroBench, and algorithm trainers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn


class BaseSNN(nn.Module, ABC):
    """
    Abstract base class for SNN models.

    All networks must implement forward(x), reset(), and n_classes.
    """
    net_tags: frozenset[str] = frozenset()

    @property
    def fully_connected(self) -> bool: return "fully_connected" in self.net_tags

    @property
    def convolutional(self) -> bool: return "convolutional" in self.net_tags

    @property
    def recurrent(self) -> bool: return "recurrent" in self.net_tags

    @property
    def single_layer(self) -> bool: return "single_layer" in self.net_tags

    @property
    def vgg(self) -> bool: return "vgg" in self.net_tags

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Single timestep forward pass.

        Args:
            x: Input tensor of shape [B, F].

        Returns:
            (spk_rec, mem_rec) where spk_rec[-1] has shape [B, n_classes].
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state (membrane, spikes)."""
        pass

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Number of output classes."""
        pass
