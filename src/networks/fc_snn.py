"""Single-step feedforward spiking network built with snnTorch layers."""

from __future__ import annotations

from math import prod
from typing import Sequence, Tuple

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from networks.base_snn import BaseSNN
is_recurrent = False

class FCSNN(BaseSNN):
    """Fully connected SNN (no recurrence), single-step.

    `forward` performs exactly one timestep on `(B, *in_shape)` input.
    State reset is external via `reset()`.
    Returns: (spk_rec, mem_rec) with one entry per spiking layer.

    Args:
        out_integrator: If True, the output layer neuron is configured as a
            pure Leaky Integrator (beta=1.0, threshold=1e9 so it never fires).
            The output neuron computes mem += W·spk at each step without spiking.
            Use mem_rec[-1] at the final timestep for prediction (do not accumulate
            over timesteps — the membrane already accumulates internally).
            If False (default), the output uses the same beta/threshold as hidden
            layers (standard Leaky — fires spikes at the output).
    """
    net_tags = frozenset({"fully_connected", "baseline"})

    def __init__(
        self,
        in_shape: Tuple[int, ...] | None = None,
        num_classes: int = 10,
        hidden_sizes: Sequence[int] = (256,),
        beta: float = 0.9,
        threshold: float = 1.0,
        spike_grad=None,
        reset_mechanism: str = "subtract",
        out_integrator: bool = False,
    ) -> None:
        super().__init__()

        if in_shape is None:
            in_shape = (784,)

        if hidden_sizes is None:
            hidden_sizes = ()

        self.in_shape = tuple(int(v) for v in in_shape)
        self._n_classes = int(num_classes)

        input_size = int(prod(self.in_shape))
        hidden_sizes = [int(v) for v in hidden_sizes]
        if any(v <= 0 for v in hidden_sizes):
            raise ValueError("All hidden layer sizes must be positive integers.")

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        layer_sizes = [input_size, *hidden_sizes, self._n_classes]
        self.input_size = input_size
        self.hidden_size = list(hidden_sizes)

        self.synapses = nn.ModuleList()
        self.neurons = nn.ModuleList()

        n_layers = len(layer_sizes) - 1
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.synapses.append(nn.Linear(int(n_in), int(n_out), bias=False))
            is_output = (i == n_layers - 1)
            if is_output and out_integrator:
                # Pure Leaky Integrator: no decay (beta=1), never fires (threshold=1e9).
                # Matches the TP algorithm's output layer (Sec 3.1): mem += W*spk.
                # Use mem_rec[-1] at the final timestep for prediction during eval.
                self.neurons.append(
                    snn.Leaky(
                        beta=1.0,
                        threshold=1e9,
                        spike_grad=spike_grad,
                        reset_mechanism=reset_mechanism,
                        init_hidden=True,
                        output=True,
                    )
                )
            else:
                self.neurons.append(
                    snn.Leaky(
                        beta=float(beta),
                        threshold=float(threshold),
                        spike_grad=spike_grad,
                        reset_mechanism=reset_mechanism,
                        init_hidden=True,
                        output=True,
                    )
                )

        # Minimal alternating list (Linear, LIF, Linear, LIF, ...)
        self.layers = nn.ModuleList()
        for fc, lif in zip(self.synapses, self.neurons):
            self.layers.append(fc)
            self.layers.append(lif)

        # Print initialization summary
        print(f"\n{'='*60}")
        print(f"  FCSNN")
        print(f"{'='*60}")
        print(f"  {'Input Shape':<25} {self.in_shape}")
        print(f"  {'Layer Sizes':<25} {layer_sizes}")
        print(f"  {'Num Classes':<25} {self._n_classes}")
        print(f"  {'Beta':<25} {beta}")
        print(f"  {'Threshold':<25} {threshold}")
        print(f"  {'Reset Mechanism':<25} {reset_mechanism}")
        print(f"  {'Out Integrator':<25} {out_integrator}")
        print(f"{'='*60}\n")

    def forward(self, x: torch.Tensor):
        if x.dim() < 2 or prod(x.shape[1:]) != prod(self.in_shape):
            raise ValueError(f"Expected input (B,{self.in_shape}) [{prod(self.in_shape)} elements], got {tuple(x.shape)}.")

        # The network flattens the input since it's fc
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
