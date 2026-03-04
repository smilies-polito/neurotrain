"""VGG11 SNN for CIFAR-10 (single-step).

Assumptions (hardcoded for CIFAR-10):
- input is always (B, 3, 32, 32)
- num_classes is always 10
- VGG11 feature layout is fixed
- head is fixed: flatten after last pool (512*1*1) -> Linear(512->10) -> LIF

Conv -> BN -> LIF -> Pool (at the end of each VGG stage)
One-step forward; persistent membrane states; call reset() externally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from networks.base_snn import BaseSNN


class VGG11SNN_CIFAR10(BaseSNN):
    net_tags = frozenset({"convolutional", "vgg", "cifar10"})


    def __init__(
        self,
        beta: float = 0.95,
        threshold: float = 1.0,
        spike_grad=None,
    ) -> None:
        super().__init__()

        self.in_shape = (3, 32, 32)
        self._n_classes = 10

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        # --- VGG11 features: [64] M [128] M [256 256] M [512 512] M [512 512] M ---

        # 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad,
                             init_hidden=True, output=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 16x16 -> 8x8
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad,
                             init_hidden=True, output=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 8x8 -> 4x4
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad,
                             init_hidden=True, output=True)

        self.conv4 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad,
                             init_hidden=True, output=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 4x4 -> 2x2
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.lif5 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad,
                             init_hidden=True, output=True)

        self.conv6 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.lif6 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad,
                             init_hidden=True, output=True)
        self.pool4 = nn.MaxPool2d(2, 2)

        # 2x2 -> 1x1
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.lif7 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad,
                             init_hidden=True, output=True)

        self.conv8 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.lif8 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad,
                             init_hidden=True, output=True)
        self.pool5 = nn.MaxPool2d(2, 2)

        # --- CIFAR-10 head: after pool5 we already have (B, 512, 1, 1) ---
        self.fc = nn.Linear(512, 10, bias=False)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad,
                                 init_hidden=True, output=True)

        # Group all layers
        self.layers = nn.ModuleList([
            self.conv1, self.lif1, self.conv2, self.lif2,
            self.conv3, self.lif3, self.conv4, self.lif4,
            self.conv5, self.lif5, self.conv6, self.lif6,
            self.conv7, self.lif7, self.conv8, self.lif8,
            self.fc, self.lif_out
        ])

        print(f"[Net][VGG11SNN_CIFAR10] beta={beta} thr={threshold}\n")


    def forward(self, x: torch.Tensor):
        # CIFAR10 only: enforce (B, 3, 32, 32)
        if x.dim() != 4 or tuple(x.shape[1:]) != (3, 32, 32):
            raise ValueError(f"Expected input shape (B, 3, 32, 32), got {tuple(x.shape)}.")

        spk_rec, mem_rec = [], []

        # Block 1
        cur = self.bn1(self.conv1(x))
        spk, mem = self.lif1(cur); spk_rec.append(spk); mem_rec.append(mem)
        out = self.pool1(spk)

        # Block 2
        cur = self.bn2(self.conv2(out))
        spk, mem = self.lif2(cur); spk_rec.append(spk); mem_rec.append(mem)
        out = self.pool2(spk)

        # Block 3
        cur = self.bn3(self.conv3(out))
        spk, mem = self.lif3(cur); spk_rec.append(spk); mem_rec.append(mem)
        out = spk

        cur = self.bn4(self.conv4(out))
        spk, mem = self.lif4(cur); spk_rec.append(spk); mem_rec.append(mem)
        out = self.pool3(spk)

        # Block 4
        cur = self.bn5(self.conv5(out))
        spk, mem = self.lif5(cur); spk_rec.append(spk); mem_rec.append(mem)
        out = spk

        cur = self.bn6(self.conv6(out))
        spk, mem = self.lif6(cur); spk_rec.append(spk); mem_rec.append(mem)
        out = self.pool4(spk)

        # Block 5
        cur = self.bn7(self.conv7(out))
        spk, mem = self.lif7(cur); spk_rec.append(spk); mem_rec.append(mem)
        out = spk

        cur = self.bn8(self.conv8(out))
        spk, mem = self.lif8(cur); spk_rec.append(spk); mem_rec.append(mem)
        out = self.pool5(spk)

        # Head: (B, 512, 1, 1) -> (B, 512) -> (B, 10)
        out = out.flatten(1)
        spk, mem = self.lif_out(self.fc(out))
        spk_rec.append(spk); mem_rec.append(mem)

        return spk_rec, mem_rec

    def reset(self) -> None:
        # Reset all membrane states
        self.lif1.reset_mem(); self.lif2.reset_mem()
        self.lif3.reset_mem(); self.lif4.reset_mem()
        self.lif5.reset_mem(); self.lif6.reset_mem()
        self.lif7.reset_mem(); self.lif8.reset_mem()
        self.lif_out.reset_mem()

    @property
    def n_classes(self) -> int:
        return 10