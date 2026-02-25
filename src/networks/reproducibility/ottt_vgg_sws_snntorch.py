"""OTTT CIFAR-10 VGG(sWS) implemented with snnTorch neurons."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate
from torch.nn.utils.parametrize import register_parametrization

from networks.base_snn import BaseSNN


class _Scale(nn.Module):
    """Multiply activations by a fixed scalar."""

    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.value


class _ScaledWeightStandardization2d(nn.Module):
    """
    sWS parametrization for Conv2d weights.

    The transformed weight is:
        (w - mean) / sqrt(var * fan_in + eps) * gain
    """

    def __init__(self, out_channels: int, eps: float = 1e-4, gain: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.gain = (
            nn.Parameter(torch.ones(int(out_channels), 1, 1, 1)) if gain else None
        )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        fan_in = int(math.prod(weight.shape[1:]))
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), keepdim=True)
        weight = (weight - mean) / torch.sqrt(var * fan_in + self.eps)
        if self.gain is not None:
            weight = weight * self.gain
        return weight


class OTTTVGGSWS_SNNtorch(BaseSNN):
    """
    Single-step VGG(sWS) for OTTT CIFAR-10 reproducibility.

    Architecture follows the official OTTT VGG(sWS) layout:
    64C3-128C3-AP2-256C3-256C3-AP2-512C3-512C3-AP2-512C3-512C3-GAP-FC
    """

    net_tags = frozenset({"convolutional", "vgg"})

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        num_classes: int = 10,
        tau: float = 2.0,
        threshold: float = 1.0,
        surrogate_slope: float = 4.0,
        feature_cfg: Sequence[Union[int, str]] | None = None,
        scale: float = 2.74,
        fc_hw: int = 1,
        weight_standardization: bool = True,
        ws_eps: float = 1e-4,
        ws_learnable_gain: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (C, H, W).")
        if float(tau) <= 1.0:
            raise ValueError("tau must be > 1.0.")
        if int(fc_hw) <= 0:
            raise ValueError("fc_hw must be a positive integer.")

        self.in_shape = tuple(int(v) for v in input_shape)
        self._flat_in_features = int(
            self.in_shape[0] * self.in_shape[1] * self.in_shape[2]
        )
        self._n_classes = int(num_classes)
        self.tau = float(tau)
        self.beta = 1.0 - 1.0 / self.tau
        self.threshold = float(threshold)
        self.surrogate_slope = float(surrogate_slope)
        self.scale = float(scale)
        self.fc_hw = int(fc_hw)
        self.weight_standardization = bool(weight_standardization)
        self.ws_eps = float(ws_eps)
        self.ws_learnable_gain = bool(ws_learnable_gain)
        self.has_residual_connections = False

        # Static image is repeated across timesteps by the experiment script.
        self.constant_input_per_timestep = True

        if feature_cfg is None:
            feature_cfg = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]
        tokens: List[Union[int, str]] = list(feature_cfg)
        if not tokens or not isinstance(tokens[0], int):
            raise ValueError("feature_cfg must start with an integer channel count.")
        if isinstance(tokens[-1], str):
            raise ValueError("feature_cfg cannot end with 'M'.")

        spike_grad = surrogate.sigmoid(slope=self.surrogate_slope)

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        self.scale_layers = nn.ModuleList()

        in_channels = int(self.in_shape[0])
        for token in tokens:
            if token == "M":
                if len(self.pool_layers) == 0:
                    raise ValueError("feature_cfg has pooling before first conv layer.")
                self.pool_layers[-1] = nn.AvgPool2d(kernel_size=2, stride=2)
                continue
            if isinstance(token, str):
                raise ValueError(f"Unknown feature_cfg token '{token}'. Use ints or 'M'.")

            out_channels = int(token)
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bool(bias),
            )
            if self.weight_standardization:
                register_parametrization(
                    conv,
                    "weight",
                    _ScaledWeightStandardization2d(
                        out_channels=out_channels,
                        eps=self.ws_eps,
                        gain=self.ws_learnable_gain,
                    ),
                )
            self.conv_layers.append(conv)
            self.pool_layers.append(nn.Identity())
            self.lif_layers.append(
                snn.Leaky(
                    beta=self.beta,
                    threshold=self.threshold,
                    spike_grad=spike_grad,
                    init_hidden=True,
                    output=True,
                    reset_mechanism="subtract",
                )
            )
            self.scale_layers.append(_Scale(self.scale))
            in_channels = out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))
        classifier_in = in_channels * self.fc_hw * self.fc_hw
        self.classifier = nn.Linear(classifier_in, self._n_classes, bias=bool(bias))

        self._initialize_weights()

    @staticmethod
    def _get_conv_weight_parameter(conv: nn.Conv2d) -> torch.Tensor:
        parametrizations = getattr(conv, "parametrizations", None)
        if parametrizations is not None and hasattr(parametrizations, "weight"):
            return parametrizations.weight.original
        return conv.weight

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                weight = self._get_conv_weight_parameter(module)
                nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def forward(self, x: torch.Tensor):
        if x.dim() == 2 and int(x.shape[1]) == self._flat_in_features:
            x = x.view(-1, *self.in_shape)
        if x.dim() != 4 or tuple(x.shape[1:]) != self.in_shape:
            raise ValueError(
                f"Expected one-step input shape (B, {self.in_shape}), got {tuple(x.shape)}."
            )

        out = x
        spk_rec = []
        mem_rec = []

        for conv, pool, lif, scale in zip(
            self.conv_layers, self.pool_layers, self.lif_layers, self.scale_layers
        ):
            cur = conv(out)
            spk, mem = lif(cur)
            spk = scale(spk)
            spk_rec.append(spk)
            mem_rec.append(mem)
            out = pool(spk)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)

        # Keep trainer contract: last entries represent output layer readout.
        spk_rec.append(logits)
        mem_rec.append(logits)
        return spk_rec, mem_rec

    def reset(self) -> None:
        for lif in self.lif_layers:
            lif.reset_mem()

