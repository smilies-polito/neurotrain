"""
Local classifier network (passive).

Feedforward SNN with per-layer encoder + decoder_y (local classifier).
Inherits from BaseSNN. Used by ELL, FELL, BELL trainers.
"""

import math
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from networks.base_snn import BaseSNN
from networks.local_classifier_block import LocalClassifierBlock


class LocalClassifierNetwork(BaseSNN):
    """
    Feedforward SNN with per-layer local classifiers.

    Stack of LocalClassifierBlock. forward(x) returns (spk_rec, mem_rec) per BaseSNN.
    forward_step_all(x_t) for trainers returns List[(spike_out, y_hat_spike)] per block.

    When tau is set, decay = exp(-1/tau) (paper-identical); else decay = beta.
    constant_input_per_timestep = True so evaluation uses same input every step (paper logic).
    """

    def __init__(
        self,
        layer_sizes: List[int],
        beta: float = 0.9,
        tau: Optional[float] = None,
        mode: Literal["ell", "fell", "bell"] = "ell",
        threshold: float = 1.0,
        bias: bool = False,
        fa: bool = False,
    ):
        super().__init__()
        self._layer_sizes = layer_sizes
        self._n_classes = layer_sizes[-1]
        self._mode = mode
        self.constant_input_per_timestep = True

        # decay: paper-identical uses tau -> decay = exp(-1/tau); else decay = beta
        if tau is not None:
            decay = math.exp(-1.0 / tau)
        else:
            decay = float(beta)

        self.blocks = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.blocks.append(
                LocalClassifierBlock(
                    num_in=layer_sizes[i],
                    num_out=layer_sizes[i + 1],
                    num_classes=self._n_classes,
                    threshold=threshold,
                    decay=decay,
                    mode=mode,
                    bias=bias,
                    fa=fa,
                )
            )

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset(self) -> None:
        for block in self.blocks:
            block.reset()

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Single timestep forward. Returns (spk_rec, mem_rec) per BaseSNN.

        spk_rec[i] = encoder spike of block i [B, hidden]; mem_rec[i] = decoder y_hat_spike [B, n_classes].
        For readout use mem_rec[-1], not spk_rec[-1].
        """
        spk_rec: List[torch.Tensor] = []
        mem_rec: List[torch.Tensor] = []

        out = x
        for block in self.blocks:
            spike_out, y_hat_spike = block.forward_step(out)
            spk_rec.append(spike_out)
            mem_rec.append(y_hat_spike)
            out = spike_out

        return spk_rec, mem_rec

    def forward_step_all(
        self, x_t: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        For ELL/FELL/BELL trainers.

        Returns [(spike_0, y_hat_0), (spike_1, y_hat_1), ...] per block.
        """
        result: List[Tuple[torch.Tensor, torch.Tensor]] = []
        out = x_t
        for block in self.blocks:
            spike_out, y_hat_spike = block.forward_step(out)
            result.append((spike_out, y_hat_spike))
            out = spike_out
        return result
