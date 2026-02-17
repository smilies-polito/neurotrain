"""
DRTP (Direct Random Target Projection) trainer for feed-forward SNNs.

Reference:
    Frenkel et al., "Learning Without Feedback: Fixed Random Learning Signals
    Allow for Feedforward Training of Deep Neural Networks", Frontiers, 2021.

Paper-aligned rule for hidden layers (Algorithm 1 / Fig. 1 conventions):
    delta_k = (y* B_k) * h'(u_k)
    dW_k = -eta * delta_k^T x_{k-1}

Output layer update uses local supervised error on selected readout
(`output_mode`: membrane or spike).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class DRTPTrainer(BaseTrainer):
    """Direct Random Target Projection trainer with fixed random feedback."""

    _VALID_DISTS = ("kaiming_uniform", "uniform", "normal")
    _VALID_OUTPUT_MODES = ("mem", "spike")
    _VALID_SURROGATES = ("logistic",)

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        loss_type: str = "mse",
        output_mode: str = "mem",
        surrogate_scale: float = 5.0,
        surrogate_type: str = "logistic",
        feedback_distribution: str = "kaiming_uniform",
        feedback_scale: float = 1.0,
        fixed_feedback: bool = True,
        freeze_conv: bool = False,
        quant: bool = False,
        use_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        update_last: bool = False,
        update_every: int = 1,
        **kwargs,
    ):
        super().__init__()

        if feedback_distribution not in self._VALID_DISTS:
            raise ValueError(
                f"feedback_distribution must be one of {self._VALID_DISTS}, got {feedback_distribution}"
            )
        if str(output_mode).lower() not in self._VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {self._VALID_OUTPUT_MODES}, got {output_mode}"
            )
        if str(surrogate_type).lower() not in self._VALID_SURROGATES:
            raise ValueError(
                f"surrogate_type must be one of {self._VALID_SURROGATES}, got {surrogate_type}"
            )
        if float(surrogate_scale) <= 0.0:
            raise ValueError("surrogate_scale must be positive")
        if int(update_every) <= 0:
            raise ValueError("update_every must be >= 1")

        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.loss_type = str(loss_type).lower()
        self.output_mode = str(output_mode).lower()
        self.surrogate_scale = float(surrogate_scale)
        self.surrogate_type = str(surrogate_type).lower()
        self.feedback_distribution = feedback_distribution
        self.feedback_scale = float(feedback_scale)
        self.fixed_feedback = bool(fixed_feedback)
        self.freeze_conv = bool(freeze_conv)
        self.quant = bool(quant)
        self.use_optimizer = bool(use_optimizer)
        self.update_last = bool(update_last)
        self.update_every = int(update_every)

        self.n_classes = int(getattr(network, "n_classes", 0))
        if self.n_classes <= 0:
            raise ValueError("DRTPTrainer requires network.n_classes > 0")

        if self.loss_type == "bce" and self.output_mode != "mem":
            warnings.warn(
                "Paper-reference DRTP uses loss_type='bce' with output_mode='mem'. "
                "Current configuration is non-reference.",
                stacklevel=2,
            )

        self.loss_fn = nn.MSELoss()
        self._output_error_fn = self._mse_error
        self._configure_loss(self.loss_type)

        self.trainable_layers, self.layer_types = self._resolve_trainable_layers()
        self.num_layers = len(self.trainable_layers)
        self.num_hidden = max(self.num_layers - 1, 0)
        if self.num_layers == 0:
            raise ValueError("DRTPTrainer requires at least one trainable layer.")

        self.layer_output_shapes = self._infer_layer_output_shapes()
        self.layer_thresholds = self._resolve_layer_thresholds()
        self.conv_pool_by_layer = self._resolve_conv_pools()

        self._external_optimizer = optimizer
        if self.use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(
                network.parameters(), lr=self.lr
            )
        else:
            self.optimizer = None

        # One fixed random feedback tensor per hidden layer.
        self.feedback = nn.ParameterList()
        if self.fixed_feedback:
            for shape in self.layer_output_shapes[:-1]:
                fb = torch.empty((self.n_classes, *shape))
                self._init_feedback_(fb)
                self.feedback.append(nn.Parameter(fb, requires_grad=False))

    def _init_feedback_(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.feedback_distribution == "kaiming_uniform":
            nn.init.kaiming_uniform_(tensor)
        elif self.feedback_distribution == "uniform":
            tensor.uniform_(-1.0, 1.0)
        elif self.feedback_distribution == "normal":
            tensor.normal_(mean=0.0, std=1.0)
        if self.feedback_scale != 1.0:
            tensor.mul_(self.feedback_scale)
        return tensor

    def _resolve_trainable_layers(self) -> tuple[List[nn.Module], List[str]]:
        # Preferred explicit contract.
        if hasattr(self.network, "trainable_layers"):
            layers = list(getattr(self.network, "trainable_layers"))
            types = list(getattr(self.network, "trainable_types", []))
            if len(types) != len(layers):
                types = [
                    "conv" if isinstance(layer, nn.Conv2d) else "linear"
                    for layer in layers
                ]
            return layers, types

        # FCSNN-style fallback.
        synapses = getattr(self.network, "synapses", None)
        if synapses is not None:
            layers = [
                layer for layer in synapses if isinstance(layer, (nn.Linear, nn.Conv2d))
            ]
            if layers:
                return layers, [
                    "conv" if isinstance(layer, nn.Conv2d) else "linear"
                    for layer in layers
                ]

        # Legacy alternating [Linear, Leaky, ...] fallback.
        layers = [
            layer
            for layer in getattr(self.network, "layers", [])
            if isinstance(layer, (nn.Linear, nn.Conv2d))
        ]
        return layers, [
            "conv" if isinstance(layer, nn.Conv2d) else "linear" for layer in layers
        ]

    def _infer_layer_output_shapes(self) -> List[tuple[int, ...]]:
        if hasattr(self.network, "layer_output_shapes"):
            shapes = list(self.network.layer_output_shapes())
        else:
            shapes = []
            for layer, layer_type in zip(self.trainable_layers, self.layer_types):
                if layer_type != "linear":
                    raise ValueError(
                        "Conv layer output shapes require network.layer_output_shapes()."
                    )
                shapes.append((int(layer.out_features),))

        if len(shapes) != self.num_layers:
            raise ValueError(
                "Mismatch between trainable layers and inferred output shapes."
            )
        return [tuple(int(v) for v in shape) for shape in shapes]

    @staticmethod
    def _to_float(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)

    def _resolve_layer_thresholds(self) -> List[float]:
        # FCSNN-style path.
        neurons = getattr(self.network, "neurons", None)
        if neurons is not None:
            thresholds = [
                self._to_float(getattr(lif, "threshold", 1.0)) for lif in neurons
            ]
            if len(thresholds) == self.num_layers:
                return thresholds

        # DRTPConvMNIST path.
        if hasattr(self.network, "conv_blocks") and hasattr(self.network, "fc_blocks"):
            thresholds: List[float] = []
            for _, lif, _ in getattr(self.network, "conv_blocks"):
                thresholds.append(self._to_float(getattr(lif, "threshold", 1.0)))
            for _, lif in getattr(self.network, "fc_blocks"):
                thresholds.append(self._to_float(getattr(lif, "threshold", 1.0)))
            if len(thresholds) == self.num_layers:
                return thresholds

        # Legacy alternating path.
        raw_layers = getattr(self.network, "layers", None)
        if raw_layers is not None:
            thresholds = [
                self._to_float(getattr(layer, "threshold", 1.0))
                for layer in raw_layers
                if hasattr(layer, "threshold")
            ]
            if len(thresholds) == self.num_layers:
                return thresholds

        return [1.0] * self.num_layers

    def _resolve_conv_pools(self) -> Dict[int, nn.Module]:
        """
        Map trainable conv-layer index -> pooling module applied before its LIF state,
        when such metadata is exposed by the network.
        """
        mapping: Dict[int, nn.Module] = {}

        conv_blocks = getattr(self.network, "conv_blocks", None)
        if conv_blocks is not None:
            for block in conv_blocks:
                if len(block) < 3:
                    continue
                conv_layer, _, pool_layer = block[0], block[1], block[2]
                for layer_idx, layer in enumerate(self.trainable_layers):
                    if layer is conv_layer:
                        mapping[layer_idx] = pool_layer

        conv_layers = getattr(self.network, "conv_layers", None)
        pool_layers = getattr(self.network, "pool_layers", None)
        if conv_layers is not None and pool_layers is not None:
            for conv_layer, pool_layer in zip(conv_layers, pool_layers):
                for layer_idx, layer in enumerate(self.trainable_layers):
                    if layer is conv_layer:
                        mapping[layer_idx] = pool_layer

        return mapping

    def _configure_loss(self, loss_type: str) -> None:
        if loss_type in ("mse", "mse_loss"):
            self.loss_fn = nn.MSELoss()
            self._output_error_fn = self._mse_error
        elif loss_type in ("bce", "binary_cross_entropy"):
            self.loss_fn = nn.BCEWithLogitsLoss()
            self._output_error_fn = self._bce_error
        else:
            raise ValueError('loss_type must be one of {"mse", "bce"}')

    @staticmethod
    def _mse_error(output: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        return output - target_onehot

    @staticmethod
    def _bce_error(output: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(output) - target_onehot

    def _project_targets(
        self, targets_onehot: torch.Tensor, feedback: torch.Tensor
    ) -> torch.Tensor:
        if feedback.dim() == 2:
            return targets_onehot @ feedback
        flat_fb = feedback.view(self.n_classes, -1)
        return (targets_onehot @ flat_fb).view(
            targets_onehot.size(0), *feedback.shape[1:]
        )

    def _sample_feedback(
        self, device: torch.device, dtype: torch.dtype
    ) -> List[torch.Tensor]:
        mats: List[torch.Tensor] = []
        for shape in self.layer_output_shapes[:-1]:
            fb = torch.empty((self.n_classes, *shape), device=device, dtype=dtype)
            self._init_feedback_(fb)
            mats.append(fb)
        return mats

    def _surrogate_derivative(
        self, membrane: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        # Logistic surrogate derivative h'(u-th) used in DRTP hidden updates.
        centered = membrane - threshold
        scaled = self.surrogate_scale * centered
        sig = torch.sigmoid(scaled)
        return self.surrogate_scale * sig * (1.0 - sig)

    @staticmethod
    def _conv_out_spatial_shape(
        x_pre: torch.Tensor, layer: nn.Conv2d
    ) -> tuple[int, int]:
        h_in, w_in = int(x_pre.shape[-2]), int(x_pre.shape[-1])
        k_h, k_w = (
            layer.kernel_size
            if isinstance(layer.kernel_size, tuple)
            else (layer.kernel_size, layer.kernel_size)
        )
        s_h, s_w = (
            layer.stride
            if isinstance(layer.stride, tuple)
            else (layer.stride, layer.stride)
        )
        p_h, p_w = (
            layer.padding
            if isinstance(layer.padding, tuple)
            else (layer.padding, layer.padding)
        )
        d_h, d_w = (
            layer.dilation
            if isinstance(layer.dilation, tuple)
            else (layer.dilation, layer.dilation)
        )
        h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
        return h_out, w_out

    def _align_conv_delta_for_weight(
        self,
        layer_idx: int,
        x_pre: torch.Tensor,
        delta_post: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a conv local signal into the conv-weight-gradient output shape.

        - If local signal already matches conv output map, use it directly.
        - If the network exposes max-pool metadata for this conv layer, unpool using
          current forward activations to match max-pool backward behavior.
        - Otherwise fall back to nearest upsampling (shape alignment only).
        """
        layer = self.trainable_layers[layer_idx]
        if not isinstance(layer, nn.Conv2d):
            raise TypeError("_align_conv_delta_for_weight called on non-conv layer.")
        if delta_post.dim() != 4:
            raise ValueError(
                f"Conv DRTP update expects 4D delta, got shape {tuple(delta_post.shape)}."
            )

        expected_hw = self._conv_out_spatial_shape(x_pre, layer)
        if tuple(delta_post.shape[-2:]) == expected_hw:
            return delta_post

        pool_layer = self.conv_pool_by_layer.get(layer_idx)
        if isinstance(pool_layer, nn.MaxPool2d):
            conv_out = layer(x_pre)
            pooled, indices = F.max_pool2d(
                conv_out,
                kernel_size=pool_layer.kernel_size,
                stride=pool_layer.stride,
                padding=pool_layer.padding,
                dilation=pool_layer.dilation,
                ceil_mode=pool_layer.ceil_mode,
                return_indices=True,
            )
            if pooled.shape != delta_post.shape:
                raise ValueError(
                    "DRTP conv projection shape mismatch after pool reconstruction: "
                    f"pooled={tuple(pooled.shape)} vs delta={tuple(delta_post.shape)}"
                )
            return F.max_unpool2d(
                delta_post,
                indices,
                kernel_size=pool_layer.kernel_size,
                stride=pool_layer.stride,
                padding=pool_layer.padding,
                output_size=conv_out.shape,
            )

        return F.interpolate(delta_post, size=expected_hw, mode="nearest")

    def _weight_grad(
        self,
        layer_idx: int,
        x_pre: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        layer = self.trainable_layers[layer_idx]

        if isinstance(layer, nn.Conv2d):
            if x_pre.dim() != 4:
                raise ValueError(
                    "Conv DRTP update expects 4D pre-synaptic input, got "
                    f"shape {tuple(x_pre.shape)}."
                )
            delta_for_weight = self._align_conv_delta_for_weight(
                layer_idx, x_pre, delta
            )
            grad = torch.nn.grad.conv2d_weight(
                x_pre,
                layer.weight.shape,
                delta_for_weight,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
            )
            return grad / float(x_pre.shape[0])

        if x_pre.dim() > 2:
            x_pre = x_pre.flatten(1)
        grad = delta.transpose(0, 1) @ x_pre
        return grad / float(x_pre.shape[0])

    def _apply_update(self, layer: nn.Module, grad_w: torch.Tensor) -> None:
        if self.freeze_conv and isinstance(layer, nn.Conv2d):
            return
        if self.use_optimizer and self.optimizer is not None:
            if layer.weight.grad is None:
                layer.weight.grad = grad_w.clone()
            else:
                layer.weight.grad += grad_w
        else:
            layer.weight.data -= self.lr * grad_w

    def _should_update(self, t: int, num_timesteps: int) -> bool:
        if self.update_last:
            return t == num_timesteps - 1
        return (t + 1) % self.update_every == 0

    @staticmethod
    def _read_layer_inputs(network: nn.Module, expected_layers: int):
        layer_inputs = getattr(network, "_last_layer_inputs", None)
        if isinstance(layer_inputs, list) and len(layer_inputs) == expected_layers:
            return layer_inputs
        return None

    @staticmethod
    def _read_layer_mems(network: nn.Module, expected_layers: int):
        layer_mems = getattr(network, "_last_layer_mems", None)
        if isinstance(layer_mems, list) and len(layer_mems) == expected_layers:
            return layer_mems
        return None

    @torch.no_grad()
    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Train one batch with online DRTP updates over timesteps."""
        if data.dim() < 3:
            raise ValueError(
                "DRTPTrainer expects data with shape [T, B, ...], got "
                f"{tuple(data.shape)}."
            )
        num_timesteps = int(data.shape[0])
        if num_timesteps <= 0:
            raise ValueError("DRTPTrainer received empty temporal sequence (T=0).")

        batch_size = int(data.shape[1])
        if target.dim() != 1 or int(target.shape[0]) != batch_size:
            raise ValueError(
                f"DRTPTrainer expects target shape [B], got {tuple(target.shape)}."
            )

        device = data.device
        target_onehot = torch.zeros(
            batch_size, self.n_classes, device=device, dtype=data.dtype
        )
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        if self.fixed_feedback:
            feedback = list(self.feedback)
        else:
            feedback = self._sample_feedback(device=device, dtype=data.dtype)

        spk_sum = torch.zeros(
            batch_size, self.n_classes, device=device, dtype=data.dtype
        )
        mem_sum = torch.zeros(
            batch_size, self.n_classes, device=device, dtype=data.dtype
        )
        output_last = None

        for t in range(num_timesteps):
            spk_rec, mem_rec = self.network(data[t])
            if len(spk_rec) != self.num_layers or len(mem_rec) != self.num_layers:
                raise ValueError(
                    "DRTPTrainer expects forward() to return per-layer spike/membrane "
                    "lists with length equal to trainable layers."
                )

            spk_sum += spk_rec[-1]
            mem_sum += mem_rec[-1]
            output_t = mem_rec[-1] if self.output_mode == "mem" else spk_rec[-1]
            output_last = output_t

            if not self._should_update(t, num_timesteps):
                continue

            layer_inputs = self._read_layer_inputs(self.network, self.num_layers)
            layer_mems = self._read_layer_mems(self.network, self.num_layers)

            # Hidden updates (paper rule): projected targets * local surrogate derivative.
            for layer_idx in range(self.num_hidden):
                x_pre = (
                    layer_inputs[layer_idx]
                    if layer_inputs is not None
                    else (data[t] if layer_idx == 0 else spk_rec[layer_idx - 1])
                )
                mem_k = (
                    layer_mems[layer_idx]
                    if layer_mems is not None
                    else mem_rec[layer_idx]
                )

                projected_targets = self._project_targets(
                    target_onehot, feedback[layer_idx]
                )
                h_prime = self._surrogate_derivative(
                    mem_k,
                    threshold=self.layer_thresholds[layer_idx],
                )
                delta_k = projected_targets * h_prime

                grad_w = self._weight_grad(layer_idx, x_pre, delta_k)
                self._apply_update(self.trainable_layers[layer_idx], grad_w)

            # Output update from selected readout.
            x_pre_out = (
                layer_inputs[-1]
                if layer_inputs is not None
                else (spk_rec[-2] if self.num_hidden > 0 else data[t])
            )
            output_error = self._output_error_fn(output_t, target_onehot)
            grad_out = self._weight_grad(self.num_layers - 1, x_pre_out, output_error)
            self._apply_update(self.trainable_layers[-1], grad_out)

            if self.use_optimizer and self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        if output_last is None:
            raise RuntimeError("DRTPTrainer internal error: output_last was never set.")

        if self.output_mode == "mem":
            readout_for_loss = output_last if self.update_last else mem_sum
            pred_readout = mem_sum
        else:
            readout_for_loss = output_last if self.update_last else spk_sum
            pred_readout = spk_sum

        loss = self.loss_fn(readout_for_loss, target_onehot)
        pred = pred_readout.argmax(dim=1, keepdim=True)
        return loss.detach(), pred

    def reset(self):
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def checkpoint_state(self) -> dict:
        if not self.fixed_feedback or len(self.feedback) == 0:
            return {}
        return {"feedback": self.feedback.state_dict()}

    def load_checkpoint_state(self, state: dict) -> None:
        feedback_state = state.get("feedback")
        if feedback_state:
            self.feedback.load_state_dict(feedback_state)

    def to(self, device):
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self
