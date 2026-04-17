from __future__ import annotations

from typing import Tuple, List

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from networks.base_snn import BaseSNN
is_recurrent = False

class ConvSNN(BaseSNN):
    """
    snnTorch tutorial-style small ConvSNN baseline:

      12C5 -> MP2 -> LIF
      32C5 -> MP2 -> LIF
      Flatten (-> 800) -> FC -> LIF(out)

    Single-step forward on (B,C,H,W); persistent membrane state; call reset() externally.
    Returns: (spk_rec, mem_rec) where each is a list of tensors per spiking layer.
    """
    net_tags = frozenset({"convolutional", "baseline", "snntorch_tutorial"})

    def __init__(
        self,
        in_shape: Tuple[int, int, int] = (3, 32, 32),
        num_classes: int = 10,
        beta: float = 0.95,
        threshold: float = 1.0,
        spike_grad=None,
    ) -> None:
        super().__init__()

        if len(in_shape) != 3:
            raise ValueError("in_shape must be (C, H, W).")
        self.in_shape = tuple(int(v) for v in in_shape)
        self._n_classes = int(num_classes)

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        # --- Feature extractor: 12C5-MP2, 32C5-MP2 ---
        self.conv1 = nn.Conv2d(self.in_shape[0], 12, kernel_size=5, stride=1, padding=0, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True, output=True)

        self.conv2 = nn.Conv2d(12, 32, kernel_size=5, stride=1, padding=0, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True, output=True)

        # Infer flatten size automatically (CIFAR-10 -> 800, FashionMNIST -> 512, etc.)
        with torch.no_grad():
            dummy = torch.zeros(
                1, *self.in_shape,
                device=self.conv1.weight.device,
                dtype=self.conv1.weight.dtype,
            )
            dummy = self.pool1(self.conv1(dummy))
            dummy = self.pool2(self.conv2(dummy))
            flat_features = int(dummy.flatten(1).shape[1])

        # --- Classifier: flat_features -> num_classes ---
        self.fc = nn.Linear(flat_features, self._n_classes, bias=False)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True, output=True)

        # Minimal registry list (synapses + neurons)
        self.layers = nn.ModuleList([self.conv1, self.lif1, self.conv2, self.lif2, self.fc, self.lif_out])

        # Print initialization summary
        print(f"\n{'='*60}")
        print(f"  ConvSNN")
        print(f"{'='*60}")
        print(f"  {'Input Shape':<25} {self.in_shape}")
        print(f"  {'Num Classes':<25} {self._n_classes}")
        print(f"  {'Conv Filters':<25} [12(k5), 32(k5)]")
        print(f"  {'Pools':<25} MaxPool(2,2)")
        print(f"  {'Flat Features':<25} {flat_features}")
        print(f"  {'Beta':<25} {beta}")
        print(f"  {'Threshold':<25} {threshold}")
        print(f"{'='*60}\n")

    def forward(self, x: torch.Tensor):
        if x.dim() != 4 or tuple(x.shape[1:]) != self.in_shape:
            raise ValueError(f"Expected input (B,{self.in_shape}), got {tuple(x.shape)}.")

        spk_rec = []
        mem_rec = []

        # Conv1 -> Pool -> LIF
        cur1 = self.pool1(self.conv1(x))
        spk1, mem1 = self.lif1(cur1)
        spk_rec.append(spk1)
        mem_rec.append(mem1)

        # Conv2 -> Pool -> LIF
        cur2 = self.pool2(self.conv2(spk1))
        spk2, mem2 = self.lif2(cur2)
        spk_rec.append(spk2)
        mem_rec.append(mem2)

        # FC -> LIF(out)
        flat = spk2.flatten(1)
        spk_out, mem_out = self.lif_out(self.fc(flat))
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

        return spk_rec, mem_rec

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset(self) -> None:
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif_out.reset_mem()