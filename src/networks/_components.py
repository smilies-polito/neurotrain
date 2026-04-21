"""
Shared conv / head / surrogate building blocks for benchmarking networks.

These used to be duplicated across six VGG-9 variants. They now live here and
are composed by the parameterized `VGG9` class in vgg9.py.

One weight-standardized conv flavor:

* `WSConv2d`  — weight-standardized Conv2d with a fixed scalar gain set at
                construction time. No learnable gain parameter, no bias.
                Inherits nn.Conv2d so trainer forward hooks (OTTT trace
                substitution) fire correctly on __call__.
                  TP-style presets use gain=1.8 (paper default).
                  OTTT-style presets use gain=1.0 (matches original init).

Two head flavors:

* `LeakyIntegrator` — pure leaky integrator with weight-standardized linear
                       projection. Used by TP-style VGG9 variants as a
                       non-firing readout.
* plain `nn.Linear` — OTTT variants use a plain Linear after AdaptiveAvgPool.

Surrogates:

* `ATanSurrogate`   — matches traces_propagation type "1":
                       grad = scale / (1 + (pi * (v - vth))^2).
                       snntorch's built-in atan uses a different formulation, so
                       we keep this custom one for faithful reproduction.
  For OTTT-style variants, use snntorch's surrogate.sigmoid(slope=4) directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Surrogate — traces_propagation type "1"
#   Θ(v) on the forward pass, grad = scale / (1 + (π·v)²) on backward.
# ---------------------------------------------------------------------------

class _ATanSurrogateFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, scale):
        ctx.save_for_backward(input_)
        ctx.scale = scale
        return (input_ >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad = ctx.scale / (1.0 + (math.pi * input_) ** 2)
        return grad * grad_output, None


class ATanSurrogate(nn.Module):
    """ArcTan-like surrogate (traces_propagation type "1")."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = float(scale)

    def forward(self, input_):
        return _ATanSurrogateFn.apply(input_, self.scale)


# ---------------------------------------------------------------------------
# Unified weight-standardized conv
# ---------------------------------------------------------------------------

class WSConv2d(nn.Conv2d):
    """
    Conv2d with Weight Standardization and a fixed scalar gain.

    w_std = gain * (w - mean) / sqrt(var * fan_in + eps)

    gain  is a plain float set at construction time — NOT a learnable
    nn.Parameter.  TP-style presets use 1.8; OTTT-style presets use 1.0
    (matching the original paper's initialization, with gain tunable via
    Optuna rather than backprop).

    No bias.  Inherits nn.Conv2d so that OTTT's forward hooks fire correctly
    on any subclass via __call__.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, dilation=1, groups=1,
                 gain: float = 1.8, eps: float = 1e-4):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False,
        )
        self.gain = float(gain)
        self.eps  = float(eps)

    def forward(self, x):
        w = self.weight
        fan_in = w.shape[1] * w.shape[2] * w.shape[3]
        mean = w.mean(dim=[1, 2, 3], keepdim=True)
        var  = w.var (dim=[1, 2, 3], keepdim=True, unbiased=True)
        w_std = self.gain * (w - mean) / ((var * fan_in + self.eps) ** 0.5)
        return F.conv2d(
            x, w_std, None,
            self.stride, self.padding, self.dilation, self.groups,
        )


# ---------------------------------------------------------------------------
# Simple scale layer — used by OTTT-style variants after each LIF
# ---------------------------------------------------------------------------

class Scale(nn.Module):
    """Multiply every activation by a fixed constant (OTTT post-LIF Scale(2.74))."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return x * self.scale


# ---------------------------------------------------------------------------
# Non-firing readout head (traces_propagation LI layer)
# ---------------------------------------------------------------------------

class LeakyIntegrator(nn.Module):
    """
    Non-firing readout: mem_t = leak * mem_{t-1} + WS_linear(x).

    Used by TP-style VGG9 variants as the output head. Returns the membrane
    potential, which trainers consume via `mem_rec[-1]`. `fc` is the
    underlying learnable nn.Linear — trainers can access it via
    `output_layer()` on the owning network.
    """

    def __init__(self, in_features: int, out_features: int,
                 leak: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.leak = float(leak)
        self.eps = float(eps)
        self.mem = torch.zeros(1)

    def forward(self, x):
        w = self.fc.weight
        fan_in = w.size(1)
        mean = w.mean(dim=1, keepdim=True)
        var = w.var(dim=1, keepdim=True, unbiased=False)
        w_std = 1.8 * (w - mean) / torch.sqrt(var * fan_in + self.eps)
        cur = F.linear(x, w_std, bias=None)
        self.mem = self.leak * self.mem + cur
        return self.mem

    def reset(self) -> None:
        self.mem = torch.zeros(1, device=self.fc.weight.device)
