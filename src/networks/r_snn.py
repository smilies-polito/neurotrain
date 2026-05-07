"""Single-step recurrent spiking network built with snnTorch layers."""

from __future__ import annotations

from math import prod
from typing import Sequence, Tuple

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from networks.base_snn import BaseSNN


class RSNN(BaseSNN):
    """Recurrent SNN with explicit recurrent dynamics (snnTorch RLeaky), single-step.

    `forward` performs exactly one timestep on `(B, *in_shape)` input.
    State reset is external via `reset()`.
    Returns: (spk_rec, mem_rec) with one entry per spiking layer (hidden rec layers + output).

    Args:
        out_integrator: If True, the output layer neuron (lif_out) is configured as a
            pure Leaky Integrator (beta=1.0, threshold=1e9 so it never fires).
            The output neuron computes mem += W·spk at each step without spiking.
            Use mem_rec[-1] at the final timestep for prediction (do not accumulate
            over timesteps — the membrane already accumulates internally).
            If False (default), lif_out uses the same beta/threshold as hidden layers.
    """
    net_tags = frozenset({"fully_connected", "recurrent", "baseline"})
    is_recurrent = True

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
            hidden_sizes = (256,)

        self.in_shape = tuple(int(v) for v in in_shape)
        self._n_classes = int(num_classes)
        self.out_integrator = bool(out_integrator)

        hidden_sizes = [int(v) for v in hidden_sizes]
        if not hidden_sizes:
            raise ValueError("RSNN requires at least one recurrent hidden layer.")
        if any(v <= 0 for v in hidden_sizes):
            raise ValueError("All hidden layer sizes must be positive integers.")

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        input_size = int(prod(self.in_shape))
        self.input_size = input_size
        self.hidden_size = list(hidden_sizes)

        self.input_layers = nn.ModuleList()
        self.recurrent_layers = nn.ModuleList()

        prev_features = input_size
        for layer_features in hidden_sizes:
            self.input_layers.append(nn.Linear(prev_features, layer_features, bias=False))
            self.recurrent_layers.append(
                snn.RLeaky(
                    beta=float(beta),
                    linear_features=int(layer_features),
                    threshold=float(threshold),
                    spike_grad=spike_grad,
                    all_to_all=True,
                    learn_recurrent=True,
                    reset_delay=False,
                    reset_mechanism=reset_mechanism,
                    init_hidden=True,
                    output=True,
                )
            )
            prev_features = int(layer_features)

        # Output head — pure Leaky Integrator when out_integrator=True
        self.fc_out = nn.Linear(prev_features, self.n_classes, bias=False)
        if out_integrator:
            # beta=1.0 (no decay), threshold=1e9 (never fires) → mem += W*spk each step.
            # Use mem_rec[-1] at the final timestep for prediction during eval.
            self.lif_out = snn.Leaky(
                beta=1.0,
                threshold=1e9,
                spike_grad=spike_grad,
                reset_mechanism=reset_mechanism,
                init_hidden=True,
                output=True,
            )
        else:
            self.lif_out = snn.Leaky(
                beta=float(beta),
                threshold=float(threshold),
                spike_grad=spike_grad,
                reset_mechanism=reset_mechanism,
                init_hidden=True,
                output=True,
            )

        # Minimal alternating list (Linear, RLeaky, ..., Linear, Leaky)
        self.layers = nn.ModuleList()
        for fc, rlif in zip(self.input_layers, self.recurrent_layers):
            self.layers.append(fc)
            self.layers.append(rlif)
        self.layers.append(self.fc_out)
        self.layers.append(self.lif_out)

        # Print initialization summary
        print(f"\n{'='*60}")
        print(f"  RSNN")
        print(f"{'='*60}")
        print(f"  {'Input Shape':<25} {self.in_shape}")
        print(f"  {'Recurrent Hidden':<25} {hidden_sizes}")
        print(f"  {'Num Classes':<25} {self._n_classes}")
        print(f"  {'Beta':<25} {beta}")
        print(f"  {'Threshold':<25} {threshold}")
        print(f"  {'Reset Mechanism':<25} {reset_mechanism}")
        print(f"  {'Out Integrator':<25} {out_integrator}")
        print(f"{'='*60}\n")

    def forward(self, x: torch.Tensor):
        if x.dim() < 2 or prod(x.shape[1:]) != prod(self.in_shape):
            raise ValueError(f"Expected input (B,{self.in_shape}) [{prod(self.in_shape)} elements], got {tuple(x.shape)}.")

        spk = x.reshape(x.shape[0], -1)

        spk_rec = []
        mem_rec = []

        # Recurrent hidden stack
        for fc, rlif in zip(self.input_layers, self.recurrent_layers):
            cur = fc(spk)
            spk, mem = rlif(cur)
            spk_rec.append(spk)
            mem_rec.append(mem)

        # Output layer
        cur_out = self.fc_out(spk)
        spk_out, mem_out = self.lif_out(cur_out)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

        return spk_rec, mem_rec

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset(self, device: torch.device | None = None) -> None:
        for rlif in self.recurrent_layers:
            rlif.reset_mem()
        self.lif_out.reset_mem()
