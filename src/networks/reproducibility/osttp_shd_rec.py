"""Paper-style recurrent SNN used for the OSTTP SHD reproduction test."""

from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from networks.base_snn import BaseSNN


class OSTTPSHDRec(BaseSNN):
    """
    Single recurrent hidden layer + leaky output integrator for SHD.

    This mirrors the architecture reported in the OSTTP paper:
    - 700-dimensional SHD input
    - one recurrent hidden layer of 450 soft-reset SNUs
    - one leaky-integrating output layer of 20 units

    The network itself is training-rule agnostic. Trainers decide whether the
    output is interpreted via spikes or membrane values.
    """

    net_tags = frozenset({"fully_connected", "recurrent", "single_layer"})

    def __init__(
        self,
        input_size: int = 700,
        hidden_size: int = 450,
        num_classes: int = 20,
        beta: float = 0.95,
        output_beta: float = 0.99,
        threshold: float = 1.0,
        output_threshold: float = 1e9,
        spike_grad=None,
    ) -> None:
        super().__init__()

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self._n_classes = int(num_classes)

        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self._n_classes <= 0:
            raise ValueError("num_classes must be positive.")

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=100.0)

        self.fc_in = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.rlif = snn.RLeaky(
            beta=float(beta),
            linear_features=self.hidden_size,
            threshold=float(threshold),
            spike_grad=spike_grad,
            all_to_all=True,
            learn_recurrent=True,
            reset_mechanism="subtract",
            init_hidden=True,
            output=True,
            reset_delay=True,
        )

        self.fc_out = nn.Linear(self.hidden_size, self._n_classes, bias=False)
        self.readout = snn.Leaky(
            beta=float(output_beta),
            threshold=float(output_threshold),
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
            reset_mechanism="none",
            reset_delay=True,
        )

        self.layers = nn.ModuleList([self.fc_in, self.rlif, self.fc_out, self.readout])

        print(
            "[Net][OSTTPSHDRec] "
            f"in={self.input_size} hidden={self.hidden_size} out={self._n_classes} "
            f"beta={float(beta):.3f} output_beta={float(output_beta):.3f}"
        )

    def forward(self, x: torch.Tensor):
        if x.dim() != 2 or x.size(1) != self.input_size:
            raise ValueError(
                f"Expected input (B, {self.input_size}), got {tuple(x.shape)}."
            )

        cur_hidden = self.fc_in(x)
        spk_hidden, mem_hidden = self.rlif(cur_hidden)

        cur_out = self.fc_out(spk_hidden)
        spk_out, mem_out = self.readout(cur_out)

        return [spk_hidden, spk_out], [mem_hidden, mem_out]

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset(self) -> None:
        self.rlif.reset_mem()
        self.readout.reset_mem()
