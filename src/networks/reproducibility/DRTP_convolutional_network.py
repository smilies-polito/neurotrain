"""
DRTP-specific ConvSNN for MNIST.

Designed to be *trainer-friendly* for DRTPTrainer:
- Exposes `trainable_layers` and `trainable_types`
- Exposes `layer_output_shapes()` for feedback matrix sizing
- Stores `_last_layer_inputs` and `_last_layer_mems` aligned with trainable layers
- `forward()` returns (spk_list, mem_list) with length == num_layers (trainable layers)
- Feed-forward only (no recurrence)

Paper-aligned topology (MNIST):
Conv(32, 5x5, stride=1, padding=2) -> MaxPool(2x2, stride=2) -> FC(1000) -> FC(10)

Important convention (kept consistent for DRTP):
- Conv "layer output" is defined AFTER pooling and AFTER LIF (i.e., post-pool spk/mem).
  This ensures the feedback projection shape matches `mem_k.shape` and the forward input
  to the next stage (flatten) is exactly the spiking output of the conv block.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

try:
    # If you have a common base class in your framework
    from networks.base_snn import BaseSNN
except Exception:
    BaseSNN = nn.Module


class DRTPConvMNIST(BaseSNN):
    def __init__(
        self,
        in_shape: Tuple[int, int, int] = (1, 28, 28),
        num_classes: int = 10,
        conv_out_channels: int = 32,
        conv_kernel_size: int = 5,
        conv_stride: int = 1,
        conv_padding: int = 2,
        pool_kernel: int = 2,
        pool_stride: int = 2,
        fc_hidden: int = 1000,
        beta: float = 0.9,
        threshold: float = 1.0,
        spike_grad=None,
        quant: bool = False,
    ) -> None:
        super().__init__()
        if len(in_shape) != 3:
            raise ValueError("in_shape must be (C, H, W).")
        self.in_shape = tuple(int(v) for v in in_shape)
        self._n_classes = int(num_classes)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.quant = bool(quant)

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        # --- Conv block (paper-like) ---
        self.conv = nn.Conv2d(
            in_channels=self.in_shape[0],
            out_channels=int(conv_out_channels),
            kernel_size=int(conv_kernel_size),
            stride=int(conv_stride),
            padding=int(conv_padding),
            bias=False,
        )
        self.pool = nn.MaxPool2d(kernel_size=int(pool_kernel), stride=int(pool_stride))
        # Convention: Conv -> Pool -> LIF
        self.conv_lif = snn.Leaky(
            beta=self.beta,
            threshold=self.threshold,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
        )

        # Infer flatten dimension after conv->pool (before LIF or after is same shape)
        with torch.no_grad():
            dummy = torch.zeros(1, *self.in_shape)
            dummy = self.pool(self.conv(dummy))
            flat_dim = int(dummy.flatten(1).shape[1])

        # --- FC classifier ---
        self.fc1 = nn.Linear(flat_dim, int(fc_hidden), bias=False)
        self.fc1_lif = snn.Leaky(
            beta=self.beta,
            threshold=self.threshold,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
        )

        self.fc2 = nn.Linear(int(fc_hidden), self._n_classes, bias=False)
        self.fc2_lif = snn.Leaky(
            beta=self.beta,
            threshold=self.threshold,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
        )

        # --- DRTPTrainer interface ---
        # Trainable layers in forward order:
        #   0: conv, 1: fc1, 2: fc2
        self.trainable_layers = [self.conv, self.fc1, self.fc2]
        self.trainable_types = ["conv", "linear", "linear"]

        # Optional convenience for threshold resolution (some trainers look at these)
        self.conv_blocks = nn.ModuleList(
            [nn.ModuleList([self.conv, self.conv_lif, self.pool])]
        )
        self.fc_blocks = nn.ModuleList(
            [
                nn.ModuleList([self.fc1, self.fc1_lif]),
                nn.ModuleList([self.fc2, self.fc2_lif]),
            ]
        )

        self.reset_parameters()

        # Buffers populated every forward() for DRTPTrainer
        self._last_layer_inputs: List[torch.Tensor] = []
        self._last_layer_mems: List[torch.Tensor] = []

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def reset(self) -> None:
        """Reset neuron states (must be called at start of each sequence/batch)."""
        self.conv_lif.reset_mem()
        self.fc1_lif.reset_mem()
        self.fc2_lif.reset_mem()

    def forward(self, x: torch.Tensor):
        """
        One-timestep forward pass.

        Args:
            x: (B, C, H, W) for MNIST (B,1,28,28)

        Returns:
            spk_rec: list of spikes per trainable layer [conv_out, fc1_out, fc2_out]
            mem_rec: list of membranes per trainable layer [conv_mem, fc1_mem, fc2_mem]
        """
        if x.dim() != 4 or tuple(x.shape[1:]) != self.in_shape:
            raise ValueError(
                f"Expected one-step input shape (B, {self.in_shape}), got {tuple(x.shape)}."
            )

        spk_rec: List[torch.Tensor] = []
        mem_rec: List[torch.Tensor] = []
        self._last_layer_inputs = []
        self._last_layer_mems = []

        # --- Layer 0: Conv ---
        # Input to the conv weights is the image tensor
        self._last_layer_inputs.append(x)
        cur0 = self.conv(x)
        cur0 = self.pool(cur0)
        spk0, mem0 = self.conv_lif(cur0)
        if self.quant:
            self.conv_lif.mem.copy_(torch.trunc(self.conv_lif.mem))
        spk_rec.append(spk0)
        mem_rec.append(mem0)
        self._last_layer_mems.append(mem0)

        # --- Flatten for FC ---
        flat = spk0.flatten(1)

        # --- Layer 1: FC1 ---
        self._last_layer_inputs.append(flat)
        cur1 = self.fc1(flat)
        spk1, mem1 = self.fc1_lif(cur1)
        if self.quant:
            self.fc1_lif.mem.copy_(torch.trunc(self.fc1_lif.mem))
        spk_rec.append(spk1)
        mem_rec.append(mem1)
        self._last_layer_mems.append(mem1)

        # --- Layer 2: FC2 (output) ---
        self._last_layer_inputs.append(spk1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.fc2_lif(cur2)
        if self.quant:
            self.fc2_lif.mem.copy_(torch.trunc(self.fc2_lif.mem))
        spk_rec.append(spk2)
        mem_rec.append(mem2)
        self._last_layer_mems.append(mem2)

        return spk_rec, mem_rec

    def layer_output_shapes(self) -> List[Tuple[int, ...]]:
        """
        Output shapes per trainable layer (excluding batch dimension).
        Must align with forward's (spk_rec/mem_rec) entries:
          0: conv lif output (B, C, H, W)
          1: fc1 output (B, F)
          2: fc2 output (B, num_classes)
        """
        dev = self.trainable_layers[0].weight.device
        with torch.no_grad():
            self.reset()
            dummy = torch.zeros(1, *self.in_shape, device=dev)
            spk_rec, _ = self.forward(dummy)
            self.reset()
        return [tuple(s.shape[1:]) for s in spk_rec]
