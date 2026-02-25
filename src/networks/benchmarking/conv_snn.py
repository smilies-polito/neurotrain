"""Single-step convolutional spiking network built with snnTorch layers."""

from __future__ import annotations

from typing import Sequence, Tuple

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from networks.base_snn import BaseSNN


class ConvSNN(BaseSNN):
    """Modest ConvSNN with explicit conv/pool/classifier blocks.

    `forward` performs exactly one timestep on `(B, C, H, W)` input.
    State reset is external via `reset()`.
    """
    net_tags = frozenset({"convolutional"})

    def __init__(
        self,
        in_shape: Tuple[int, int, int] = (1, 28, 28),   # Shape of the input to the network (excluding batch dimension).
        num_classes: int = 10,                          # Number of output classes.
        conv_channels: Sequence[int] = (32, 64),        # List with output channels for each convolutional layer.
        fc_hidden_sizes: Sequence[int] = (128,),        # List with sizes of classifier hidden layers.
        use_batch_norm: bool = True,                    # If True, apply BatchNorm2d after each conv layer.
        pool_after: Sequence[bool] | None = None,       # Whether to apply pooling after each conv layer.
        pool_kernel: int = 2,                           # Kernel size for MaxPool layers.
        pool_stride: int = 2,                           # Stride for MaxPool layers.
        conv_kernel_size: int = 3,                      # Kernel size for Conv2d layers.
        conv_stride: int = 1,                           # Stride for Conv2d layers.
        conv_padding: int = 1,                          # Padding for Conv2d layers.
        beta: float = 0.9,                              # Decay factor for all LIF neurons.
        threshold: float = 1.0,                         # Firing threshold for all LIF neurons.
        spike_grad=None,                                # Surrogate gradient function for the spiking neurons.
    ) -> None:
        super().__init__()
        if len(in_shape) != 3:
            raise ValueError("in_shape must be (C, H, W) for ConvSNN.")

        self.in_shape = tuple(int(v) for v in in_shape)
        self._n_classes = int(num_classes)
        self.use_batch_norm = bool(use_batch_norm)
        self.beta = float(beta)
        self.threshold = float(threshold)

        self.conv_channels = [int(v) for v in conv_channels]
        if not self.conv_channels:
            raise ValueError("conv_channels must contain at least one value.")
        if any(v <= 0 for v in self.conv_channels):
            raise ValueError("All conv_channels values must be positive integers.")

        self.fc_hidden_sizes = [int(v) for v in fc_hidden_sizes]
        if any(v <= 0 for v in self.fc_hidden_sizes):
            raise ValueError("All fc_hidden_sizes values must be positive integers.")

        if pool_after is None:
            pool_after = tuple(True for _ in self.conv_channels)
        self.pool_after = [bool(v) for v in pool_after]
        if len(self.pool_after) != len(self.conv_channels):
            raise ValueError("pool_after must have the same length as conv_channels.")

        if pool_kernel <= 0 or pool_stride <= 0:
            raise ValueError("pool_kernel and pool_stride must be positive.")

        # Quick guide:
        # - conv depth/dimensions are set by conv_channels.
        # - set use_batch_norm=True (default) for the common Conv+BN+Pool+LIF style.
        # - pooling placement is set by pool_after (one bool per conv layer).
        # - classifier depth/dimensions are set by fc_hidden_sizes.
        # - conv pipeline follows snnTorch-style ordering: Conv -> Pool -> LIF.
        #   fc_hidden_sizes=() -> conv features go directly to output.
        classifier_sizes = [*self.fc_hidden_sizes, self._n_classes]

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        self.conv_neurons = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        in_channels = self.in_shape[0]
        for out_channels, use_pool in zip(self.conv_channels, self.pool_after):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=int(conv_kernel_size),
                    stride=int(conv_stride),
                    padding=int(conv_padding),
                    bias=False,
                )
            )
            if self.use_batch_norm:
                self.conv_norms.append(nn.BatchNorm2d(out_channels))
            else:
                self.conv_norms.append(nn.Identity())
            self.conv_neurons.append(
                snn.Leaky(
                    beta=self.beta,
                    threshold=self.threshold,
                    spike_grad=spike_grad,
                    init_hidden=True,
                    output=True,
                )
            )
            if use_pool:
                self.pool_layers.append(
                    nn.MaxPool2d(kernel_size=int(pool_kernel), stride=int(pool_stride))
                )
            else:
                self.pool_layers.append(nn.Identity())
            in_channels = out_channels

        with torch.no_grad():
            dummy = torch.zeros(1, *self.in_shape)
            self._layer_output_shapes = []
            for conv, norm, pool in zip(self.conv_layers, self.conv_norms, self.pool_layers):
                dummy = pool(norm(conv(dummy)))
                self._layer_output_shapes.append(tuple(int(v) for v in dummy.shape[1:]))
            flat_features = int(dummy.flatten(1).shape[1])

        self.classifier_layers = nn.ModuleList()
        self.classifier_neurons = nn.ModuleList()
        prev_features = flat_features
        for out_features in classifier_sizes:
            self.classifier_layers.append(
                nn.Linear(prev_features, int(out_features), bias=False)
            )
            self.classifier_neurons.append(
                snn.Leaky(
                    beta=self.beta,
                    threshold=self.threshold,
                    spike_grad=spike_grad,
                    init_hidden=True,
                    output=True,
                )
            )
            prev_features = int(out_features)

        self.trainable_layers = list(self.conv_layers) + list(self.classifier_layers)
        self.trainable_types = ["conv"] * len(self.conv_layers) + ["linear"] * len(
            self.classifier_layers
        )

        # Legacy compatibility path for trainers expecting alternating
        # [synapse, neuron, ...] via network.layers.
        self.layers = nn.ModuleList()
        self.stop_layer_specs = []
        for conv, lif, pool in zip(
            self.conv_layers, self.conv_neurons, self.pool_layers
        ):
            self.layers.append(conv)
            self.layers.append(lif)
            self.stop_layer_specs.append(
                {
                    "synapse": conv,
                    "neuron": lif,
                    "layer_type": "conv",
                    "pool": pool,
                }
            )
        for fc, lif in zip(self.classifier_layers, self.classifier_neurons):
            self.layers.append(fc)
            self.layers.append(lif)
            self.stop_layer_specs.append(
                {
                    "synapse": fc,
                    "neuron": lif,
                    "layer_type": "linear",
                    "pool": None,
                }
            )

        with torch.no_grad():
            dummy = torch.zeros(1, *self.in_shape)
            for conv, norm, pool in zip(
                self.conv_layers, self.conv_norms, self.pool_layers
            ):
                dummy = pool(norm(conv(dummy)))
            dummy = dummy.flatten(1)
            for fc in self.classifier_layers:
                dummy = fc(dummy)
                self._layer_output_shapes.append((int(dummy.shape[1]),))

        print(
            f"[Net][ConvSNN] in_shape={self.in_shape} "
            f"conv={self.conv_channels} pool_after={self.pool_after} "
            f"classifier={[*self.fc_hidden_sizes, self._n_classes]}"
        )

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
            self.conv_layers, self.conv_norms, self.conv_neurons, self.pool_layers
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
        for lif in self.conv_neurons:
            lif.reset_mem()
        for lif in self.classifier_neurons:
            lif.reset_mem()

    def layer_output_shapes(self) -> list[tuple[int, ...]]:
        return list(self._layer_output_shapes)
