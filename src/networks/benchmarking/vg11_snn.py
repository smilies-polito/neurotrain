"""Single-step VGG-11-ish convolutional SNN built with explicit snnTorch blocks."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from networks.base_snn import BaseSNN


class VG11SNN(BaseSNN):
    """Deeper VGG-11-style SNN with repeated conv blocks and pooling.

    `forward` performs exactly one timestep on `(B, C, H, W)` input.
    State reset is external via `reset()`.
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int] = (3, 32, 32),           # Shape of the input to the network (excluding batch dimension).
        num_classes: int = 10,                                  # Number of output classes.
        feature_cfg: Sequence[Union[int, str]] | None = None,   # VGG-style feature layout using ints and "M" pooling tokens.
        classifier_hidden_sizes: Sequence[int] | None = None,   # List with sizes of classifier hidden layers.
        base_channels: int = 16,                                # Base channel multiplier used when scaling integer cfg tokens.
        cfg_scale_with_base_channels: bool = True,              # If True, int tokens in feature_cfg are scaled by base_channels.
        use_batch_norm: bool = True,                            # If True, apply BatchNorm2d after each conv layer.
        pool_kernel: int = 2,                                   # Kernel size for MaxPool layers triggered by "M" tokens.
        pool_stride: int = 2,                                   # Stride for MaxPool layers triggered by "M" tokens.
        beta: float = 0.9,                                      # Decay factor for all LIF neurons.
        threshold: float = 1.0,                                 # Firing threshold for all LIF neurons.
        spike_grad=None,                                        # Surrogate gradient function for the spiking neurons.
    ) -> None:
        super().__init__()
        if len(in_shape) != 3:
            raise ValueError("in_shape must be (C, H, W) for VG11SNN.")

        self.in_shape = tuple(int(v) for v in in_shape)
        self._n_classes = int(num_classes)
        self.use_batch_norm = bool(use_batch_norm)

        if feature_cfg is None:
            feature_cfg = [1, "M", 2, "M", 4, 4, "M", 8, 8, "M", 8, 8, "M"]
        self.feature_cfg = [token if isinstance(token, str) else int(token) for token in feature_cfg]
        if not any(isinstance(token, int) for token in self.feature_cfg):
            raise ValueError("feature_cfg must contain at least one conv token.")

        if classifier_hidden_sizes is None:
            classifier_hidden_sizes = []
        self.classifier_hidden_sizes = [int(v) for v in classifier_hidden_sizes]
        if any(v <= 0 for v in self.classifier_hidden_sizes):
            raise ValueError("All classifier_hidden_sizes values must be positive integers.")

        if pool_kernel <= 0 or pool_stride <= 0:
            raise ValueError("pool_kernel and pool_stride must be positive.")

        # Quick guide:
        # - feature_cfg controls VGG-style conv/pool stages using ints and "M".
        # - if cfg_scale_with_base_channels=True, int tokens are multipliers of base_channels.
        # - set use_batch_norm=True (default) for the common VGG-style Conv+BN+Pool+LIF stack.
        # - classifier_hidden_sizes controls classifier depth.
        # - feature pipeline follows snnTorch-style ordering: Conv -> Pool -> LIF.
        #   classifier_hidden_sizes=() -> direct output layer after conv stack.
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        in_channels = self.in_shape[0]
        for token in self.feature_cfg:
            if token == "M":
                if not self.pool_layers:
                    raise ValueError("Invalid VGG-11 config: pool before first conv.")
                self.pool_layers[-1] = nn.MaxPool2d(
                    kernel_size=int(pool_kernel), stride=int(pool_stride)
                )
                continue
            if isinstance(token, str):
                raise ValueError(f"Unknown feature_cfg token '{token}'. Use ints or 'M'.")

            out_channels = int(token)
            if cfg_scale_with_base_channels:
                out_channels = int(base_channels) * out_channels
            if out_channels <= 0:
                raise ValueError("Resolved conv out_channels must be positive.")
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            )
            if self.use_batch_norm:
                self.conv_norms.append(nn.BatchNorm2d(out_channels))
            else:
                self.conv_norms.append(nn.Identity())
            self.lif_layers.append(
                snn.Leaky(
                    beta=float(beta),
                    threshold=float(threshold),
                    spike_grad=spike_grad,
                    init_hidden=True,
                    output=True,
                )
            )
            self.pool_layers.append(nn.Identity())
            in_channels = out_channels

        with torch.no_grad():
            dummy = torch.zeros(2, *self.in_shape)
            for conv, norm, pool in zip(self.conv_layers, self.conv_norms, self.pool_layers):
                dummy = pool(norm(conv(dummy)))
            flat_features = int(dummy.flatten(1).shape[1])

        classifier_sizes = [*self.classifier_hidden_sizes, self._n_classes]
        self.classifier_layers = nn.ModuleList()
        self.classifier_neurons = nn.ModuleList()
        prev_features = flat_features
        for out_features in classifier_sizes:
            self.classifier_layers.append(
                nn.Linear(prev_features, int(out_features), bias=False)
            )
            self.classifier_neurons.append(
                snn.Leaky(
                    beta=float(beta),
                    threshold=float(threshold),
                    spike_grad=spike_grad,
                    init_hidden=True,
                    output=True,
                )
            )
            prev_features = int(out_features)

    def forward(self, x: torch.Tensor):
        if x.dim() != 4 or tuple(x.shape[1:]) != self.in_shape:
            raise ValueError(
                f"Expected one-step input shape (B, {self.in_shape}), got {tuple(x.shape)}."
            )

        out = x
        spk_rec = []
        mem_rec = []
        # LIF state is persistent across forward calls and reset by reset().
        for conv, norm, lif, pool in zip(
            self.conv_layers, self.conv_norms, self.lif_layers, self.pool_layers
        ):
            cur = pool(norm(conv(out)))
            spk, mem = lif(cur)
            spk_rec.append(spk)
            mem_rec.append(mem)
            out = spk

        out = out.flatten(1)
        for fc, lif in zip(self.classifier_layers, self.classifier_neurons):
            spk, mem = lif(fc(out))
            spk_rec.append(spk)
            mem_rec.append(mem)
            out = spk

        return spk_rec, mem_rec

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset(self) -> None:
        for lif in self.lif_layers:
            lif.reset_mem()
        for lif in self.classifier_neurons:
            lif.reset_mem()
