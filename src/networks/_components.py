"""
Shared conv / head / surrogate building blocks for benchmarking networks.

These used to be duplicated across six VGG-9 variants. They now live here and
are composed by the parameterized `VGG9` class in vgg9.py.

Two weight-standardized conv flavors are exposed:

* `WSConv2d`        — traces-propagation / TP style. Fixed scale 1.8, no
                       learnable gain, no bias. Inherits from nn.Conv2d so that
                       trainer forward hooks (OTTT's trace substitution) fire
                       on __call__.
* `ScaledWSConv2d`  — OTTT style. Per-output-channel learnable gain (init 1.8),
                       optional bias. Also inherits from nn.Conv2d for hooks.

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
# Weight-standardized convs
# ---------------------------------------------------------------------------

class WSConv2d(nn.Conv2d):
    """
    Conv2d with fixed-scale Weight Standardization (traces_propagation recipe).

    w_std = 1.8 * (w - mean) / sqrt(var * fan_in + eps)

    No learnable gain, no bias by default. Inherits nn.Conv2d so that forward
    hooks registered on instances fire on __call__ — this is required for
    OTTT's per-synapse trace substitution to work on WS convolutions.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, dilation=1, groups=1, eps: float = 1e-5):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=False,
        )
        self.eps = float(eps)

    def forward(self, x):
        w = self.weight
        fan_in = w[0].numel()
        mean = w.mean(dim=[1, 2, 3], keepdim=True)
        var = w.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        w_std = 1.8 * (w - mean) / torch.sqrt(var * fan_in + self.eps)
        return F.conv2d(
            x, w_std, bias=None,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
        )


class ScaledWSConv2d(nn.Conv2d):
    """
    Conv2d with Weight Standardization and optional per-channel learnable gain
    (OTTT-SNN recipe).

    w_std = (w - mean) / sqrt(var * fan_in + eps)
    w_out = gain * w_std     (if gain is enabled)

    Inherits nn.Conv2d so OTTT's forward hooks fire correctly.
    """

    def __init__(self, *args, gain: bool = True, gain_init: float = 1.8,
                 eps: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        if gain:
            self.gain = nn.Parameter(
                torch.full((self.out_channels, 1, 1, 1), float(gain_init))
            )
        else:
            self.gain = None
        self.eps = float(eps)

    def forward(self, x):
        w = self.weight
        fan_in = w.shape[1] * w.shape[2] * w.shape[3]
        mean = w.mean(dim=[1, 2, 3], keepdim=True)
        var = w.var(dim=[1, 2, 3], keepdim=True)
        w_std = (w - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            w_std = w_std * self.gain
        return F.conv2d(
            x, w_std, self.bias,
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
