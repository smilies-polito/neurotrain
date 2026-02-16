"""Single-step feedforward spiking network built with snnTorch layers."""

from __future__ import annotations

from math import prod
from typing import Sequence, Tuple

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from networks.base_snn import BaseSNN


class FCSNN(BaseSNN):
    """Fully connected SNN with no recurrent connections.

    `forward` performs exactly one timestep on `(B, *in_shape)` input.
    State reset is external via `reset()`.
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...] = (1, 28, 28),    # Shape of the input to the network (excluding batch dimension).        
        num_classes: int = 10,                      # Number of output classes.
        hidden_sizes: Sequence[int] = (256, 128),   # List with the sizes of hidden layers. Empty tuple means no hidden layers.
        beta: float = 0.9,                          # Decay factor for the leaky integrate-and-fire neurons.
        threshold: float = 1.0,                     # Firing threshold for the leaky integrate-and-fire neurons.
        spike_grad=None,                            # Surrogate gradient function for the spiking neurons.     
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = ()
            
        self.in_shape = tuple(int(v) for v in in_shape)
        self.input_size = int(prod(self.in_shape))
        self.hidden_size = [int(v) for v in hidden_sizes]
        if any(v <= 0 for v in self.hidden_size):
            raise ValueError("All hidden layer sizes must be positive integers.")
        self._n_classes = int(num_classes)
        
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        layer_sizes = [self.input_size, *self.hidden_size, self._n_classes]
        self.synapses = nn.ModuleList()
        self.neurons = nn.ModuleList()
        
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.synapses.append(nn.Linear(int(n_in), int(n_out), bias=False))
            self.neurons.append(
                snn.Leaky(
                    beta=float(beta),
                    threshold=float(threshold),
                    spike_grad=spike_grad,
                    init_hidden=True,
                    output=True,
                )
            )

    def forward(self, x: torch.Tensor):
        if x.dim() != len(self.in_shape) + 1 or tuple(x.shape[1:]) != self.in_shape:
            raise ValueError(
                f"Expected one-step input shape (B, {self.in_shape}), got {tuple(x.shape)}."
            )

        spk = x.reshape(x.shape[0], -1)
        spk_rec = []
        mem_rec = []
        for fc, lif in zip(self.synapses, self.neurons):
            cur = fc(spk)
            spk, mem = lif(cur)
            spk_rec.append(spk)
            mem_rec.append(mem)
        return spk_rec, mem_rec

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset(self) -> None:
        for lif in self.neurons:
            lif.reset_mem()
