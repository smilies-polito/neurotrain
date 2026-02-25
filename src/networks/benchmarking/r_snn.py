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
    """Recurrent SNN with explicit recurrent dynamics.

    `forward` performs exactly one timestep on `(B, *in_shape)` input.
    State reset is external via `reset()`.
    """
    net_tags = frozenset({"fully_connected", "recurrent", "single_layer"})

    def __init__(
        self,
        in_shape: Tuple[int, ...] = (1, 28, 28),      # Shape of the input to the network (excluding batch dimension).
        num_classes: int = 10,                        # Number of output classes.
        hidden_sizes: Sequence[int] = (256,),         # List with the sizes of recurrent hidden layers.
        beta: float = 0.9,                            # Decay factor for recurrent and output LIF neurons.
        threshold: float = 1.0,                       # Firing threshold for recurrent and output LIF neurons.
        spike_grad=None,                              # Surrogate gradient function for the spiking neurons.
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = (256,)
        self.hidden_sizes = [int(v) for v in hidden_sizes]
        if not self.hidden_sizes:
            raise ValueError("RSNN requires at least one recurrent hidden layer.")
        if any(v <= 0 for v in self.hidden_sizes):
            raise ValueError("All hidden layer sizes must be positive integers.")

        self.in_shape = tuple(int(v) for v in in_shape)
        self.input_size = int(prod(self.in_shape))
        self.hidden_size = (
            self.hidden_sizes[0] if len(self.hidden_sizes) == 1 else self.hidden_sizes
        )
        self._n_classes = int(num_classes)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.n_in = self.input_size
        self.n_rec = int(self.hidden_sizes[0])
        self.n_out = self._n_classes
        self.alpha = self.beta
        self.kappa = self.beta
        self.is_recurrent = True
        self.register_buffer("vo", torch.zeros(0), persistent=False)

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        self.input_layers = nn.ModuleList()
        self.recurrent_layers = nn.ModuleList()
        prev_features = self.input_size
        for layer_features in self.hidden_sizes:
            self.input_layers.append(nn.Linear(prev_features, layer_features, bias=False))
            self.recurrent_layers.append(
                snn.RLeaky(
                    beta=self.beta,
                    linear_features=layer_features,
                    threshold=self.threshold,
                    spike_grad=spike_grad,
                    all_to_all=True,
                    learn_recurrent=True,
                    reset_delay=False,
                    init_hidden=True,
                    output=True,
                )
            )
            prev_features = layer_features

        self.fc_out = nn.Linear(prev_features, self._n_classes, bias=False)
        self.lif_out = snn.Leaky(
            beta=self.beta,
            threshold=self.threshold,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
        )

        self.synapses = nn.ModuleList([*self.input_layers, self.fc_out])
        self.neurons = nn.ModuleList([*self.recurrent_layers, self.lif_out])
        self.trainable_layers = list(self.synapses)
        self.trainable_types = ["linear"] * len(self.synapses)

        # Legacy compatibility path for trainers that expect alternating
        # [Linear, (R)Leaky, ...] layout on `network.layers`.
        self.layers = nn.ModuleList()
        for syn, neu in zip(self.synapses, self.neurons):
            self.layers.append(syn)
            self.layers.append(neu)

        print(
            f"[Net][RSNN] in_shape={self.in_shape} "
            f"recurrent_hidden={self.hidden_sizes} out={self._n_classes}"
        )

    def forward(self, x: torch.Tensor):
        if x.dim() != len(self.in_shape) + 1 or tuple(x.shape[1:]) != self.in_shape:
            raise ValueError(
                f"Expected one-step input shape (B, {self.in_shape}), got {tuple(x.shape)}."
            )

        spk = x.reshape(x.shape[0], -1)
        spk_rec = []
        mem_rec = []
        for fc, rlif in zip(self.input_layers, self.recurrent_layers):
            cur = fc(spk)
            spk, mem = rlif(cur)
            spk_rec.append(spk)
            mem_rec.append(mem)

        cur_out = self.fc_out(spk)
        spk_out, mem_out = self.lif_out(cur_out)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
        return spk_rec, mem_rec

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def w_in(self) -> torch.nn.Parameter:
        return self.input_layers[0].weight

    @property
    def w_rec(self) -> torch.nn.Parameter:
        return self.recurrent_layers[0].recurrent.weight

    @property
    def w_out(self) -> torch.nn.Parameter:
        return self.fc_out.weight

    def step(self, x: torch.Tensor, vo: torch.Tensor | None = None):
        if len(self.input_layers) != 1 or len(self.recurrent_layers) != 1:
            raise NotImplementedError(
                "RSNN.step supports exactly one recurrent hidden layer."
            )

        if x.dim() != len(self.in_shape) + 1 or tuple(x.shape[1:]) != self.in_shape:
            raise ValueError(
                f"Expected one-step input shape (B, {self.in_shape}), got {tuple(x.shape)}."
            )

        flat = x.reshape(x.shape[0], -1)
        cur = self.input_layers[0](flat)
        z_t, v_t = self.recurrent_layers[0](cur)

        vo_prev = self.vo if vo is None else vo
        if vo_prev.shape != (x.shape[0], self.n_out):
            vo_prev = torch.zeros(x.shape[0], self.n_out, device=x.device)
        vo_t = self.kappa * vo_prev + z_t @ self.w_out.t()
        self.vo = vo_t
        return z_t, v_t, vo_t

    def reset(self, device: torch.device | None = None) -> None:
        for rlif in self.recurrent_layers:
            rlif.reset_mem()
        self.lif_out.reset_mem()
        ref = next(self.parameters()).device
        if device is not None:
            ref = device
        self.vo = torch.zeros(1, self.n_out, device=ref)
