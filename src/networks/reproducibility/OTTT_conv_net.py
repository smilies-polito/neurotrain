"""OTTT CIFAR-10 VGG(sWS) style network for paper-oriented reproducibility."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base_snn import BaseSNN


class _ReplaceForGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_for_backward: torch.Tensor, x_for_forward: torch.Tensor):
        return x_for_forward

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, grad_output


class _SigmoidSpike(torch.autograd.Function):
    """Official OTTT surrogate: heaviside forward + sigmoid-derivative backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = float(alpha)
        return (x >= 0).to(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        if ctx.needs_input_grad[0]:
            (x,) = ctx.saved_tensors
            sgax = torch.sigmoid(x * ctx.alpha)
            grad_x = grad_output * (1.0 - sgax) * sgax * ctx.alpha
        return grad_x, None


class _Scale(nn.Module):
    """Multiply activations by a fixed scalar (2.74 in the OTTT reference VGG)."""

    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.value


class _ScaledWSConv2d(nn.Conv2d):
    """Scaled weight-standardized conv from official OTTT VGG implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        gain: bool = True,
        eps: float = 1e-4,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.eps = float(eps)
        self.gain = (
            nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        )

    def _get_weight(self) -> torch.Tensor:
        fan_in = int(math.prod(self.weight.shape[1:]))
        mean = self.weight.mean(dim=(1, 2, 3), keepdim=True)
        var = self.weight.var(dim=(1, 2, 3), keepdim=True)
        weight = (self.weight - mean) / torch.sqrt(var * fan_in + self.eps)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self._get_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _WrappedSNNOp(nn.Module):
    """
    Official OTTT grad-with-rate wrapper:
    forward value uses spike path; backward behaves as if rate path was used.
    """

    def __init__(self, op: nn.Module):
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        require_wrap = bool(kwargs.get("require_wrap", True))
        if not require_wrap:
            return self.op(x)

        batch_size = int(x.shape[0] // 2)
        spike = x[:batch_size]
        rate = x[batch_size:]

        with torch.no_grad():
            out = self.op(spike).detach()

        in_for_grad = _ReplaceForGrad.apply(spike, rate)
        out_for_grad = self.op(in_for_grad)
        return _ReplaceForGrad.apply(out_for_grad, out)


class _OnlineLIFNode(nn.Module):
    """Online LIF used by official OTTT (detached membrane + rate tracking)."""

    def __init__(self, tau: float = 2.0, threshold: float = 1.0, alpha: float = 4.0):
        super().__init__()
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.beta = 1.0 - 1.0 / self.tau
        self.v: torch.Tensor | None = None
        self.rate_tracking: torch.Tensor | None = None

    def _forward_init(self, x: torch.Tensor) -> None:
        self.v = torch.zeros_like(x)
        self.rate_tracking = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        init = bool(kwargs.get("init", False))
        output_type = str(kwargs.get("output_type", "spike"))

        if init or self.v is None or self.v.shape != x.shape:
            self._forward_init(x)

        self.v = self.v.detach() * self.beta + x
        spike = _SigmoidSpike.apply(self.v - self.threshold, self.alpha)
        self.v = self.v - spike.detach() * self.threshold

        with torch.no_grad():
            if self.rate_tracking is None:
                self.rate_tracking = spike.detach().clone()
            else:
                self.rate_tracking = self.rate_tracking * self.beta + spike.detach()

        if output_type == "spike_rate":
            return torch.cat((spike, self.rate_tracking), dim=0)
        return spike

    def reset_state(self) -> None:
        self.v = None
        self.rate_tracking = None


class _SequentialModule(nn.Sequential):
    """Sequential that forwards OTTT kwargs only to neuron/wrapped-op layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for module in self:
            if isinstance(module, (_OnlineLIFNode, _WrappedSNNOp)):
                x = module(x, **kwargs)
            else:
                x = module(x)
        return x


class OTTTConvNet(BaseSNN):
    """
    Official-style OTTT VGG(sWS) for CIFAR-10 (light classifier variant).

    This class mirrors the public OTTT-SNN implementation behavior:
    - Conv blocks with scaled WS
    - OnlineLIF neurons with per-layer rate tracking
    - wrapped synapses for grad-with-rate (except the first conv)
    """
    net_tags = frozenset({"convolutional"})

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        num_classes: int = 10,
        tau: float = 2.0,
        threshold: float = 1.0,
        feature_cfg: Sequence[Union[int, str]] | None = None,
        scale: float = 2.74,
        fc_hw: int = 1,
        weight_standardization: bool = True,
        ws_eps: float = 1e-4,
        bias: bool = True,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (C, H, W).")
        if float(tau) <= 1.0:
            raise ValueError("tau must be > 1.0 for a stable LIF update.")

        self.in_shape = tuple(int(v) for v in input_shape)
        self._flat_in_features = int(
            self.in_shape[0] * self.in_shape[1] * self.in_shape[2]
        )
        self._n_classes = int(num_classes)
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.scale = float(scale)
        self.fc_hw = int(fc_hw)
        self.weight_standardization = bool(weight_standardization)
        self.ws_eps = float(ws_eps)
        self.has_residual_connections = False

        # Official CIFAR OTTT: same static frame each timestep.
        self.constant_input_per_timestep = True
        # Signal to OTTTTrainer to use network-internal grad-with-rate path.
        self.uses_internal_ottt_grad = True

        if feature_cfg is None:
            feature_cfg = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]
        tokens: List[Union[int, str]] = list(feature_cfg)
        if not tokens or not isinstance(tokens[0], int):
            raise ValueError("feature_cfg must start with an integer channel count.")
        if isinstance(tokens[-1], str):
            raise ValueError("feature_cfg cannot end with 'M'.")

        Conv = _ScaledWSConv2d if self.weight_standardization else nn.Conv2d
        layers: List[nn.Module] = []
        in_channels = int(self.in_shape[0])
        first_conv = True

        for token in tokens:
            if token == "M":
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
                continue
            if isinstance(token, str):
                raise ValueError(f"Unknown feature_cfg token '{token}'. Use ints or 'M'.")

            out_channels = int(token)
            conv = Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bool(bias),
                eps=self.ws_eps,
            )

            if first_conv:
                conv_layer: nn.Module = conv
                first_conv = False
            else:
                conv_layer = _WrappedSNNOp(conv)

            layers.extend(
                [
                    conv_layer,
                    _OnlineLIFNode(tau=self.tau, threshold=self.threshold, alpha=4.0),
                    _Scale(self.scale),
                ]
            )
            in_channels = out_channels

        self.features = _SequentialModule(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))

        classifier_in = in_channels * self.fc_hw * self.fc_hw
        if self.training:
            self.classifier = _SequentialModule(
                _WrappedSNNOp(
                    nn.Linear(classifier_in, self._n_classes, bias=bool(bias))
                )
            )
        else:
            self.classifier = _SequentialModule(
                nn.Linear(classifier_in, self._n_classes, bias=bool(bias))
            )

        self._initialize_weights()
        self._step_idx = 0

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep classifier wrapping behavior aligned with official grad_with_rate logic.
        linear = None
        if isinstance(self.classifier[0], _WrappedSNNOp):
            linear = self.classifier[0].op
        elif isinstance(self.classifier[0], nn.Linear):
            linear = self.classifier[0]

        if linear is None:
            return self

        if mode and not isinstance(self.classifier[0], _WrappedSNNOp):
            self.classifier = _SequentialModule(_WrappedSNNOp(linear))
        elif (not mode) and isinstance(self.classifier[0], _WrappedSNNOp):
            self.classifier = _SequentialModule(linear)
        return self

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (_ScaledWSConv2d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if x.dim() == 2 and int(x.shape[1]) == self._flat_in_features:
            x = x.view(-1, *self.in_shape)
        if x.dim() != 4 or tuple(x.shape[1:]) != self.in_shape:
            raise ValueError(
                f"Expected one-step input shape (B, {self.in_shape}), got {tuple(x.shape)}."
            )

        init = self._step_idx == 0
        require_wrap = bool(self.training)

        if require_wrap:
            out = self.features(x, init=init, output_type="spike_rate", require_wrap=True)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            logits = self.classifier(
                out, init=init, output_type="spike_rate", require_wrap=True
            )
        else:
            out = self.features(x, init=init, require_wrap=False)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            logits = self.classifier(out, require_wrap=False)

        self._step_idx += 1
        spk_rec = [logits]
        mem_rec = [logits]
        return spk_rec, mem_rec

    def reset(self) -> None:
        self._step_idx = 0
        for module in self.modules():
            if isinstance(module, _OnlineLIFNode):
                module.reset_state()
