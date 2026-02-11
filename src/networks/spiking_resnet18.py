from __future__ import annotations

from typing import List, Optional, Tuple

import snntorch as snn
import torch
import torch.nn as nn

from networks.base_snn import BaseSNN
from trainers.stop_trainer import get_stop_spike_grad


class _SpikingBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        beta: float,
        threshold: float,
        spike_grad,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
        )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
        )

        self.shortcut_conv: Optional[nn.Conv2d] = None
        self.shortcut_bn: Optional[nn.BatchNorm2d] = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

        self.stop_layer_specs = [
            {"synapse": self.conv1, "neuron": self.lif1, "layer_type": "conv", "pool": None},
        ]
        # Skip projection shares output local error (lif2 membrane) with the main path.
        # This matches residual addition before spiking nonlinearity.
        if self.shortcut_conv is not None:
            self.stop_layer_specs.append(
                {
                    "synapse": self.shortcut_conv,
                    "neuron": self.lif2,
                    "layer_type": "conv",
                    "pool": None,
                    "update_neuron_params": False,
                }
            )
        self.stop_layer_specs.append(
            {"synapse": self.conv2, "neuron": self.lif2, "layer_type": "conv", "pool": None}
        )

    def forward(self, x: torch.Tensor):
        spk_rec: List[torch.Tensor] = []
        mem_rec: List[torch.Tensor] = []

        out = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(out)
        spk_rec.append(spk1)
        mem_rec.append(mem1)

        identity = x
        if self.shortcut_conv is not None and self.shortcut_bn is not None:
            identity = self.shortcut_bn(self.shortcut_conv(x))

        out = self.bn2(self.conv2(spk1))
        out = out + identity
        spk2, mem2 = self.lif2(out)
        spk_rec.append(spk2)
        mem_rec.append(mem2)

        return spk2, spk_rec, mem_rec

    def reset(self) -> None:
        self.lif1.reset_mem()
        self.lif2.reset_mem()


class SpikingResNet18(BaseSNN):
    """
    ResNet-18 style spiking network with residual skip connections.
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
        self.has_residual_connections = True

        spike_grad = get_stop_spike_grad(surrogate)

        c1 = int(base_channels)
        self.stem_conv = nn.Conv2d(
            input_channels, c1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(c1)
        self.stem_lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
        )

        self.layer1 = self._make_stage(c1, c1, num_blocks=2, stride=1, beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.layer2 = self._make_stage(c1, c1 * 2, num_blocks=2, stride=2, beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.layer3 = self._make_stage(c1 * 2, c1 * 4, num_blocks=2, stride=2, beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.layer4 = self._make_stage(c1 * 4, c1 * 8, num_blocks=2, stride=2, beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c1 * 8, self._n_classes, bias=False)
        self.fc_lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
        )

        self.stop_layer_specs = [
            {"synapse": self.stem_conv, "neuron": self.stem_lif, "layer_type": "conv", "pool": None}
        ]
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in stage:
                self.stop_layer_specs.extend(block.stop_layer_specs)
        self.stop_layer_specs.append(
            {"synapse": self.fc, "neuron": self.fc_lif, "layer_type": "linear", "pool": None}
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        beta: float,
        threshold: float,
        spike_grad,
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()
        blocks.append(
            _SpikingBasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                beta=beta,
                threshold=threshold,
                spike_grad=spike_grad,
            )
        )
        for _ in range(1, num_blocks):
            blocks.append(
                _SpikingBasicBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    beta=beta,
                    threshold=threshold,
                    spike_grad=spike_grad,
                )
            )
        return blocks

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        spk_rec: List[torch.Tensor] = []
        mem_rec: List[torch.Tensor] = []

        out = self.stem_bn(self.stem_conv(x))
        out, mem = self.stem_lif(out)
        spk_rec.append(out)
        mem_rec.append(mem)

        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in stage:
                out, b_spk, b_mem = block(out)
                spk_rec.extend(b_spk)
                mem_rec.extend(b_mem)

        out = self.avgpool(out).flatten(1)
        out = self.fc(out)
        spk_out, mem_out = self.fc_lif(out)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
        return spk_rec, mem_rec

    def reset(self) -> None:
        self.stem_lif.reset_mem()
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in stage:
                block.reset()
        self.fc_lif.reset_mem()
