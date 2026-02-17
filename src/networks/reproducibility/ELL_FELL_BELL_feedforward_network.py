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
from utils.linear_fa import LinearFA
from utils.surrogate_gradient import ExponentialSurroGrad


class LocalClassifierNetwork(BaseSNN):
    """
    Feedforward SNN with per-layer local classifiers.

    Stack of LocalClassifierBlock. forward(x) returns (spk_rec, mem_rec) per BaseSNN.
    forward_step_all(x_t) for trainers returns List[(spike_out, y_hat_spike)] per block.

    When tau is set, decay = exp(-1/tau) (paper-identical); else decay = beta.
    constant_input_per_timestep = True so evaluation uses same input every step (paper logic).
    """

    class LocalClassifierBlock(nn.Module):
        """
        Passive local classifier block: encoder + LIF + decoder_y.

        Performs forward only: encoder + LIF dynamics + decoder_y (local readout).
        No optimizers, no backward. Learning logic is in the trainer.
        Implements forward_step(x_in) -> (spike_out, y_hat_spike).
        Mode controls detach in recurrence: ELL detaches, FELL/BELL do not.
        """
        """
        Passive local classifier block: encoder + LIF + decoder_y.

        Implements forward_step(x_in) -> (spike_out, y_hat_spike).
        Mode controls detach in recurrence: ELL detaches, FELL/BELL do not.
        """

        def __init__(
            self,
            num_in: int,
            num_out: int,
            num_classes: int,
            threshold: float = 1.0,
            decay: float = 0.9,
            mode: Literal["ell", "fell", "bell"] = "ell",
            bias: bool = False,
            fa: bool = False,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.decay = decay
            self.thresh = threshold
            self.mode = mode

            self.encoder = nn.Linear(num_in, num_out, bias=bias)
            if fa:
                self.decoder_y = LinearFA(num_out, num_classes, bias=bias)
            else:
                self.decoder_y = nn.Linear(num_out, num_classes, bias=bias)

            # State (cleared by reset)
            self.register_buffer("_mem", torch.zeros(0), persistent=False)
            self.register_buffer("_spike", torch.zeros(0), persistent=False)
            self.register_buffer("_y_hat_mem", torch.zeros(0), persistent=False)
            self.register_buffer("_y_hat_spike", torch.zeros(0), persistent=False)

        def reset(self) -> None:
            """Clear membrane and spike state."""
            self._mem = torch.zeros(0)
            self._spike = torch.zeros(0)
            self._y_hat_mem = torch.zeros(0)
            self._y_hat_spike = torch.zeros(0)

        def forward_step(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Single timestep forward.

            Args:
                x_in: Input [B, F_in].

            Returns:
                (spike_out, y_hat_spike): spike_out [B, F_out] for next layer,
                    y_hat_spike [B, n_classes] for local loss.
            """
            batch_size = x_in.shape[0]
            device = x_in.device

            if self._mem.numel() == 0 or self._mem.shape[0] != batch_size:
                self._mem = torch.zeros(batch_size, self.encoder.out_features, device=device)
                self._spike = torch.zeros(batch_size, self.encoder.out_features, device=device)
                self._y_hat_mem = torch.zeros(batch_size, self.num_classes, device=device)
                self._y_hat_spike = torch.zeros(batch_size, self.num_classes, device=device)

            h = self.encoder(x_in)

            # Encoder LIF
            if self.mode == "ell":
                self._mem = (
                    self._mem.detach() * self.decay
                    + h
                    - self._spike.detach() * self.thresh * self.decay
                )
                self._spike = ExponentialSurroGrad.apply(self._mem, self.thresh)
                spike_for_decoder = self._spike
            elif self.mode == "fell":
                # FELL: per-step backward + step; detach recurrence to avoid in-place error when
                # next step's backward traverses previous step's graph (params already updated).
                prev_mem = self._mem.detach()
                prev_spike = self._spike.detach()
                mem_new = prev_mem * self.decay + h - prev_spike * self.thresh * self.decay
                self._mem = mem_new.detach()
                self._spike = ExponentialSurroGrad.apply(mem_new, self.thresh).detach()
                spike_for_decoder = ExponentialSurroGrad.apply(mem_new, self.thresh)
            else:
                # BELL: one backward at end; keep recurrence in graph for BPTT.
                mem_new = (
                    self._mem * self.decay
                    + h
                    - self._spike * self.thresh * self.decay
                )
                self._mem = mem_new
                self._spike = ExponentialSurroGrad.apply(mem_new, self.thresh)
                spike_for_decoder = self._spike

            # Decoder (local classifier) with LIF-like readout
            y_dec = self.decoder_y(spike_for_decoder)
            if self.mode == "ell":
                self._y_hat_mem = (
                    self._y_hat_mem.detach() * self.decay
                    + y_dec
                    - self._y_hat_spike.detach() * self.thresh * self.decay
                )
            elif self.mode == "fell":
                prev_mem = self._y_hat_mem.detach()
                prev_spike = self._y_hat_spike.detach()
                y_hat_mem_new = prev_mem * self.decay + y_dec - prev_spike * self.thresh * self.decay
                self._y_hat_mem = y_hat_mem_new.detach()
                self._y_hat_spike = ExponentialSurroGrad.apply(y_hat_mem_new, self.thresh).detach()
                y_hat_spike_out = ExponentialSurroGrad.apply(y_hat_mem_new, self.thresh)
                return spike_for_decoder, y_hat_spike_out
            else:
                # BELL: no detach — keep decoder recurrence in graph for BPTT.
                y_hat_mem_new = (
                    self._y_hat_mem * self.decay
                    + y_dec
                    - self._y_hat_spike * self.thresh * self.decay
                )
                self._y_hat_mem = y_hat_mem_new
                self._y_hat_spike = ExponentialSurroGrad.apply(y_hat_mem_new, self.thresh)
                return spike_for_decoder, self._y_hat_spike

            self._y_hat_spike = ExponentialSurroGrad.apply(self._y_hat_mem, self.thresh)
            return spike_for_decoder, self._y_hat_spike

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
                LocalClassifierNetwork.LocalClassifierBlock(
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
