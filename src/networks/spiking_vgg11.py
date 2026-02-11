from __future__ import annotations

from typing import List, Tuple

import snntorch as snn
import torch
import torch.nn as nn

from networks.base_snn import BaseSNN
from trainers.stop_trainer import get_stop_spike_grad


class SpikingVGG11(BaseSNN):
    """
    VGG-11 style spiking network (CIFAR-like input).

    Single-step forward returns (spk_rec, mem_rec) over all spiking layers.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        base_channels: int = 64,
        surrogate: str = "exp",
    ):
        super().__init__()
        self._n_classes = int(num_classes)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.has_residual_connections = False

        c1 = int(base_channels)
        c2 = int(base_channels * 2)
        c3 = int(base_channels * 4)
        c4 = int(base_channels * 8)
        cfg = [c1, "M", c2, "M", c3, c3, "M", c4, c4, "M", c4, c4, "M"]

        spike_grad = get_stop_spike_grad(surrogate)
        in_channels = int(input_channels)

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for token in cfg:
            if token == "M":
                if len(self.pool_layers) == 0:
                    raise ValueError("Invalid VGG cfg: pooling before any conv layer.")
                self.pool_layers[-1] = nn.MaxPool2d(kernel_size=2, stride=2)
                continue

            out_channels = int(token)
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            self.lif_layers.append(
                snn.Leaky(
                    beta=self.beta,
                    threshold=self.threshold,
                    learn_beta=True,
                    learn_threshold=True,
                    spike_grad=spike_grad,
                )
            )
            self.pool_layers.append(nn.Identity())
            in_channels = out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, self._n_classes, bias=False)
        self.fc_lif = snn.Leaky(
            beta=self.beta,
            threshold=self.threshold,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
        )

        self.stop_layer_specs = []
        for conv, lif, pool in zip(self.conv_layers, self.lif_layers, self.pool_layers):
            self.stop_layer_specs.append(
                {
                    "synapse": conv,
                    "neuron": lif,
                    "layer_type": "conv",
                    "pool": pool if isinstance(pool, nn.MaxPool2d) else None,
                }
            )
        self.stop_layer_specs.append(
            {
                "synapse": self.fc,
                "neuron": self.fc_lif,
                "layer_type": "linear",
                "pool": None,
            }
        )

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        spk_rec: List[torch.Tensor] = []
        mem_rec: List[torch.Tensor] = []

        out = x
        for conv, bn, lif, pool in zip(
            self.conv_layers, self.bn_layers, self.lif_layers, self.pool_layers
        ):
            cur = bn(conv(out))
            spk, mem = lif(cur)
            spk_rec.append(spk)
            mem_rec.append(mem)
            out = pool(spk)

        out = self.avgpool(out).flatten(1)
        out = self.fc(out)
        spk_out, mem_out = self.fc_lif(out)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
        return spk_rec, mem_rec

    def reset(self) -> None:
        for lif in self.lif_layers:
            lif.reset_mem()
        self.fc_lif.reset_mem()
