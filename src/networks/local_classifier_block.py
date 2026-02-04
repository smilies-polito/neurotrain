"""
Local classifier block (passive).

Performs forward only: encoder + LIF dynamics + decoder_y (local readout).
No optimizers, no backward. Learning logic is in the trainer.
"""

from typing import Literal, Tuple

import torch
import torch.nn as nn

from utils.linear_fa import LinearFA
from utils.surrogate_gradient import ExponentialSurroGrad


class LocalClassifierBlock(nn.Module):
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
        else:
            self._mem = (
                self._mem * self.decay + h - self._spike * self.thresh * self.decay
            )
        self._spike = ExponentialSurroGrad.apply(self._mem, self.thresh)

        # Decoder (local classifier) with LIF-like readout
        y_dec = self.decoder_y(self._spike)
        if self.mode == "ell":
            self._y_hat_mem = (
                self._y_hat_mem.detach() * self.decay
                + y_dec
                - self._y_hat_spike.detach() * self.thresh * self.decay
            )
        else:
            self._y_hat_mem = (
                self._y_hat_mem * self.decay
                + y_dec
                - self._y_hat_spike * self.thresh * self.decay
            )
        self._y_hat_spike = ExponentialSurroGrad.apply(self._y_hat_mem, self.thresh)

        return self._spike, self._y_hat_spike
