from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import snntorch as snn
import torch
import torch.nn as nn


class ConvFCNetwork(nn.Module):
    """
    Convolutional front-end + FC spiking network with LIF neurons.

    Conv blocks: Conv2d -> LIF -> MaxPool2d
    FC blocks: Linear -> LIF
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        conv_layers: List[Dict[str, int]],
        layer_sizes: List[int],
        beta: float,
        threshold: float = 1.0,
        quant: bool = False,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (channels, height, width).")
        if not conv_layers:
            raise ValueError("conv_layers must contain at least one conv layer.")
        if len(layer_sizes) < 1:
            raise ValueError("layer_sizes must contain at least the output size.")

        self.input_shape = tuple(int(v) for v in input_shape)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.quant = quant
        self.fc_layer_sizes = [int(v) for v in layer_sizes]
        self.n_classes = int(self.fc_layer_sizes[-1])
        self.hidden_size = [int(v) for v in self.fc_layer_sizes[:-1]]

        self.conv_blocks = nn.ModuleList()
        in_channels = self.input_shape[0]
        for cfg in conv_layers:
            out_channels = int(cfg["out_channels"])
            kernel_size = int(cfg["kernel_size"])
            stride = int(cfg.get("stride", 1))
            padding = int(cfg.get("padding", 0))
            pool_kernel = int(cfg.get("pool_kernel", 2))
            pool_stride = int(cfg.get("pool_stride", 2))

            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            lif = snn.Leaky(beta=self.beta, threshold=self.threshold)
            pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
            self.conv_blocks.append(nn.ModuleList([conv, lif, pool]))
            in_channels = out_channels

        self.fc_blocks = nn.ModuleList()

        conv_out_shape = self._infer_conv_output_shape()
        self.conv_output_shape = conv_out_shape
        self.conv_flat_features = int(torch.prod(torch.tensor(conv_out_shape)))

        fc_sizes = [self.conv_flat_features] + self.fc_layer_sizes
        for in_features, out_features in zip(fc_sizes[:-1], fc_sizes[1:]):
            fc = nn.Linear(int(in_features), int(out_features), bias=False)
            lif = snn.Leaky(beta=self.beta, threshold=self.threshold)
            self.fc_blocks.append(nn.ModuleList([fc, lif]))

        self.trainable_layers = []
        self.trainable_types = []
        for conv, _, _ in self.conv_blocks:
            self.trainable_layers.append(conv)
            self.trainable_types.append("conv")
        for fc, _ in self.fc_blocks:
            self.trainable_layers.append(fc)
            self.trainable_types.append("linear")

        self.reset_parameters()

        print(f"\n\nNetwork: {self.__class__.__name__}")
        print(f"Input shape: {self.input_shape}")
        print(f"Conv layers: {len(self.conv_blocks)}")
        print(f"FC sizes: {self.fc_layer_sizes}")
        print(f"Beta: {self.beta}")
        print(f"Threshold: {self.threshold}")

    def _infer_conv_output_shape(self) -> Tuple[int, int, int]:
        with torch.no_grad():
            dummy = torch.zeros((1, *self.input_shape))
            self.reset()
            out = dummy
            for conv, lif, pool in self.conv_blocks:
                cur = conv(out)
                spk, _ = lif(cur)
                out = pool(spk)
            self.reset()
            return out.shape[1:]

    def reset_parameters(self):
        for conv, _, _ in self.conv_blocks:
            nn.init.kaiming_uniform_(conv.weight)
        for fc, _ in self.fc_blocks:
            nn.init.xavier_uniform_(fc.weight)

    def forward(self, x: torch.Tensor):
        spk_rec, mem_rec = [], []
        self._last_layer_inputs = []
        self._last_layer_spks = []
        self._last_layer_mems = []

        out = x
        for conv, lif, pool in self.conv_blocks:
            self._last_layer_inputs.append(out)
            cur = conv(out)
            spk, mem = lif(cur)
            if self.quant:
                lif.mem.copy_(torch.trunc(lif.mem))
            spk_rec.append(spk)
            mem_rec.append(mem)
            self._last_layer_spks.append(spk)
            self._last_layer_mems.append(mem)
            out = pool(spk)

        out = out.view(out.size(0), -1)
        for fc, lif in self.fc_blocks:
            self._last_layer_inputs.append(out)
            cur = fc(out)
            spk, mem = lif(cur)
            if self.quant:
                lif.mem.copy_(torch.trunc(lif.mem))
            spk_rec.append(spk)
            mem_rec.append(mem)
            self._last_layer_spks.append(spk)
            self._last_layer_mems.append(mem)
            out = spk

        return spk_rec, mem_rec

    def reset(self):
        for _, lif, _ in self.conv_blocks:
            lif.reset_mem()
        for _, lif in self.fc_blocks:
            lif.reset_mem()

    def layer_output_shapes(self) -> List[Tuple[int, ...]]:
        with torch.no_grad():
            dummy = torch.zeros(
                (1, *self.input_shape), device=self.trainable_layers[0].weight.device
            )
            self.reset()
            spk_rec, _ = self.forward(dummy)
            self.reset()
        return [tuple(spk.shape[1:]) for spk in spk_rec]
