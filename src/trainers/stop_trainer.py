"""
STOP trainer for feed-forward SNNs (STOP-WTL / STOP-W / STOP-WT / STOP-WL).

Implements the rule from:
  Gao et al. (2025), "STOP: spatiotemporal orthogonal propagation for
  weight-threshold-leakage synergistic training of deep SNNs".

Key properties implemented here:
  - No BPTT / no autograd-through-time.
  - Temporal gradient information flows only through forward traces
    (Eq. 20 / 26 / 30 in the paper).
  - Spatial error propagation is per-time-step only.
  - Supports Linear and Conv2d feed-forward stacks.
  - Supports trainable weights / thresholds / leakage in any combination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


@dataclass
class _LayerSpec:
    """Container for one trainable SNN layer in STOP notation."""

    synapse: nn.Module
    neuron: nn.Module
    layer_type: str  # "linear" or "conv"
    pool: Optional[nn.Module] = None


class STOPTrainer(BaseTrainer):
    """
    SpatioTemporal Orthogonal Propagation trainer.

    Notation mapping used in code:
      - U^l[t]          -> mem_rec[l]
      - s^l[t]          -> spk_rec[l]
      - delta^l[t]      -> deltas[l]
      - w_tilde^l[t]    -> w_trace[l]
      - theta_tilde^l[t]-> theta_trace[l]
      - alpha_tilde^l[t]-> alpha_trace[l]
      - DeltaW^l        -> d_w[l]
      - DeltaTheta^l    -> d_theta[l]
      - DeltaAlpha^l    -> d_alpha_scalar[l]
    """

    _VALID_SURROGATES = ("exp", "rational")
    _VALID_LOSSES = ("ce", "mse")

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        loss_type: str = "ce",
        surrogate: str = "exp",
        learn_weights: bool = True,
        learn_thresholds: bool = True,
        learn_leakage: bool = True,
        lr_weight: Optional[float] = None,
        lr_threshold: Optional[float] = None,
        lr_leakage: Optional[float] = None,
        threshold_min: float = 1e-3,
        momentum: float = 0.0,
        cosine_schedule: bool = False,
        cosine_t_max: int = 0,
        static_input_timesteps: int = 1,
        quant: bool = False,
        use_optimizer: bool = False,
        optimizer=None,
        **kwargs,
    ):
        super().__init__()

        if use_optimizer or optimizer is not None:
            raise ValueError(
                "STOPTrainer uses manual STOP updates. Set training.optimizer to null."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if threshold_min <= 0.0:
            raise ValueError("threshold_min must be > 0.")

        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.loss_type = str(loss_type).lower()
        self.surrogate = str(surrogate).lower()
        self.learn_weights = bool(learn_weights)
        self.learn_thresholds = bool(learn_thresholds)
        self.learn_leakage = bool(learn_leakage)
        self.lr_weight = float(self.lr if lr_weight is None else lr_weight)
        self.lr_threshold = float(self.lr if lr_threshold is None else lr_threshold)
        self.lr_leakage = float(self.lr if lr_leakage is None else lr_leakage)
        self.threshold_min = float(threshold_min)
        self.momentum = float(momentum)
        self.cosine_schedule = bool(cosine_schedule)
        self.cosine_t_max = int(cosine_t_max)
        self.static_input_timesteps = int(static_input_timesteps)
        self.quant = bool(quant)

        if self.loss_type not in self._VALID_LOSSES:
            raise ValueError(f"loss_type must be one of {self._VALID_LOSSES}")
        if self.surrogate not in self._VALID_SURROGATES:
            raise ValueError(f"surrogate must be one of {self._VALID_SURROGATES}")
        if (
            self.cosine_schedule
            and self.cosine_t_max <= 0
        ):
            raise ValueError("cosine_t_max must be > 0 when cosine_schedule=True.")
        if (
            not self.learn_weights
            and not self.learn_thresholds
            and not self.learn_leakage
        ):
            raise ValueError("At least one of learn_weights/thresholds/leakage must be True.")

        self.layer_specs = self._resolve_layer_specs()
        self.num_layers = len(self.layer_specs)
        self.n_classes = int(getattr(network, "n_classes"))

        # Runtime buffers populated by forward hooks each timestep.
        self._hook_inputs: List[torch.Tensor] = []
        self._hook_spikes: List[torch.Tensor] = []
        self._hook_mems: List[torch.Tensor] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_state_hooks()

        # Optional optimizer-like extras inside the trainer.
        self._momentum_buffers: Dict[str, torch.Tensor] = {}
        self._global_step = 0

    def _resolve_layer_specs(self) -> List[_LayerSpec]:
        """Resolve trainable synapse + neuron pairs in forward order."""
        specs: List[_LayerSpec] = []

        # ConvFCNetwork path: explicit conv/fc blocks and known pool location.
        if hasattr(self.network, "conv_blocks") and hasattr(self.network, "fc_blocks"):
            for block in getattr(self.network, "conv_blocks"):
                if len(block) != 3:
                    raise ValueError("Expected conv block format [Conv2d, LIF, Pool].")
                conv, lif, pool = block
                if not isinstance(conv, nn.Conv2d):
                    raise TypeError("Conv block must start with nn.Conv2d.")
                if not hasattr(lif, "beta") or not hasattr(lif, "threshold"):
                    raise TypeError("STOP expects neuron modules with beta and threshold.")
                specs.append(_LayerSpec(conv, lif, "conv", pool))

            for block in getattr(self.network, "fc_blocks"):
                if len(block) != 2:
                    raise ValueError("Expected FC block format [Linear, LIF].")
                fc, lif = block
                if not isinstance(fc, nn.Linear):
                    raise TypeError("FC block must start with nn.Linear.")
                if not hasattr(lif, "beta") or not hasattr(lif, "threshold"):
                    raise TypeError("STOP expects neuron modules with beta and threshold.")
                specs.append(_LayerSpec(fc, lif, "linear", None))

            if not specs:
                raise ValueError("No trainable layers found for STOP.")
            return specs

        # FC-like path: alternating [Linear, Leaky, Linear, Leaky, ...]
        if hasattr(self.network, "layers"):
            modules = list(getattr(self.network, "layers"))
            idx = 0
            while idx < len(modules):
                syn = modules[idx]
                if not isinstance(syn, (nn.Linear, nn.Conv2d)):
                    idx += 1
                    continue
                if idx + 1 >= len(modules):
                    raise ValueError("Synapse layer must be followed by a spiking neuron.")
                neuron = modules[idx + 1]
                if not hasattr(neuron, "beta") or not hasattr(neuron, "threshold"):
                    raise TypeError("STOP expects neuron modules with beta and threshold.")
                layer_type = "conv" if isinstance(syn, nn.Conv2d) else "linear"
                specs.append(_LayerSpec(syn, neuron, layer_type, None))
                idx += 2

            if not specs:
                raise ValueError("No trainable layers found for STOP.")
            return specs

        raise TypeError(
            "Unsupported network structure for STOPTrainer. Expected conv_blocks/fc_blocks "
            "or alternating layers with Linear/Conv2d + LIF modules."
        )

    def _register_state_hooks(self) -> None:
        """Capture per-layer presynaptic input, spikes, and membrane each forward step."""

        def _synapse_hook(module, inputs, output):
            if not inputs:
                raise RuntimeError("Synapse hook did not receive inputs.")
            self._hook_inputs.append(inputs[0].detach())

        def _neuron_hook(module, inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(
                    "Neuron output must be (spike, membrane) for STOPTrainer."
                )
            spk, mem = output[0], output[1]
            self._hook_spikes.append(spk.detach())
            self._hook_mems.append(mem.detach())

        for spec in self.layer_specs:
            self._hook_handles.append(spec.synapse.register_forward_hook(_synapse_hook))
            self._hook_handles.append(spec.neuron.register_forward_hook(_neuron_hook))

    def _clear_step_hooks(self) -> None:
        self._hook_inputs.clear()
        self._hook_spikes.clear()
        self._hook_mems.clear()

    @staticmethod
    def _as_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=device, dtype=dtype)
        return torch.tensor(float(value), device=device, dtype=dtype)

    def _theta_tensor(
        self, neuron: nn.Module, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self._as_tensor(getattr(neuron, "threshold"), device, dtype)

    def _alpha_tensor(
        self, neuron: nn.Module, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self._as_tensor(getattr(neuron, "beta"), device, dtype)

    def _expand_theta(
        self, theta: torch.Tensor, out_spikes: torch.Tensor, layer_type: str
    ) -> torch.Tensor:
        """Broadcast threshold tensor to the current layer output shape."""
        if theta.numel() == 1:
            return theta.view(1, *([1] * (out_spikes.dim() - 1)))

        if layer_type == "linear":
            if theta.shape == (out_spikes.shape[1],):
                return theta.view(1, -1)
            if theta.shape == out_spikes.shape[1:]:
                return theta.view(1, *theta.shape)

        if layer_type == "conv":
            if theta.dim() == 1 and theta.shape[0] == out_spikes.shape[1]:
                return theta.view(1, -1, 1, 1)
            if theta.shape == out_spikes.shape[1:]:
                return theta.view(1, *theta.shape)
            if theta.dim() == 4 and theta.shape[0] == 1:
                return theta

        # Generic fallback: try direct broadcast from [1, *theta.shape].
        candidate = theta.view((1,) + tuple(theta.shape))
        try:
            _ = candidate + out_spikes
        except RuntimeError as exc:
            raise ValueError(
                f"Cannot broadcast threshold shape {tuple(theta.shape)} "
                f"to output shape {tuple(out_spikes.shape)}."
            ) from exc
        return candidate

    def _reduce_threshold_update(
        self, raw_update: torch.Tensor, theta_param, layer_type: str
    ) -> torch.Tensor:
        """
        Reduce DeltaTheta_j map to parameter shape.

        - Conv per-channel theta: mean over batch + spatial dims.
        - FC per-neuron theta: mean over batch.
        - Scalar theta: mean over all dimensions.
        """
        if not isinstance(theta_param, torch.Tensor):
            return raw_update.mean()

        target = theta_param.detach()
        if target.numel() == 1:
            return raw_update.mean().reshape_as(target)

        if layer_type == "linear":
            if target.shape == (raw_update.shape[1],):
                return raw_update.mean(dim=0)
            if target.shape == raw_update.shape[1:]:
                return raw_update.mean(dim=0).reshape_as(target)

        if layer_type == "conv":
            if target.dim() == 1 and target.shape[0] == raw_update.shape[1]:
                return raw_update.mean(dim=(0, 2, 3))
            if target.shape == raw_update.shape[1:]:
                return raw_update.mean(dim=0)
            if target.dim() == 4 and target.shape[0] == 1:
                reduced = raw_update.mean(dim=0, keepdim=True)
                # For broadcasted singleton dims in parameter, average across that dim.
                for dim in range(1, reduced.dim()):
                    if target.shape[dim] == 1 and reduced.shape[dim] != 1:
                        reduced = reduced.mean(dim=dim, keepdim=True)
                return reduced.reshape_as(target)

        mean_no_batch = raw_update.mean(dim=0)
        if mean_no_batch.numel() == target.numel():
            return mean_no_batch.reshape_as(target)

        raise ValueError(
            f"Cannot reduce threshold update {tuple(raw_update.shape)} "
            f"to theta shape {tuple(target.shape)}."
        )

    def _set_neuron_attr(self, neuron: nn.Module, attr_name: str, value: torch.Tensor) -> None:
        """Write back threshold/beta regardless of tensor-vs-scalar storage."""
        current = getattr(neuron, attr_name)
        if isinstance(current, (nn.Parameter, torch.Tensor)):
            target = current.data if isinstance(current, nn.Parameter) else current
            value = value.to(device=target.device, dtype=target.dtype)
            if value.numel() == 1 and target.numel() > 1:
                value = value.expand_as(target)
            elif value.numel() == target.numel():
                value = value.reshape_as(target)
            elif tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Cannot set {attr_name} with shape {tuple(value.shape)} "
                    f"to tensor shape {tuple(target.shape)}."
                )
        if isinstance(current, nn.Parameter):
            current.data.copy_(value)
            return
        if isinstance(current, torch.Tensor):
            current.copy_(value)
            return
        if value.numel() != 1:
            raise ValueError(f"Cannot set non-scalar {attr_name} on scalar-backed neuron.")
        setattr(neuron, attr_name, float(value.item()))

    def _surrogate_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Eq. (3): surrogate derivative phi_SG(U - theta) used for spatial propagation.
        """
        if self.surrogate == "exp":
            return torch.exp(-x.abs())
        # "rational" option from request: 1 / (1 + pi^2 * x^2)
        return 1.0 / (1.0 + (math.pi * x).pow(2))

    def _loss_and_output_error(
        self,
        out_spikes: torch.Tensor,
        target: torch.Tensor,
        target_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Instantaneous E[t] and dE/ds^L[t].
        """
        if self.loss_type == "ce":
            loss_t = F.cross_entropy(out_spikes, target, reduction="mean")
            grad_out = torch.softmax(out_spikes, dim=1) - target_onehot
            return loss_t, grad_out

        # MSE
        diff = out_spikes - target_onehot
        loss_t = 0.5 * diff.pow(2).sum(dim=1).mean()
        return loss_t, diff

    def _compute_pool_cache(self, spk_rec: List[torch.Tensor]) -> List[Optional[dict]]:
        """
        Cache MaxPool indices to backpropagate spatial error from pooled input
        back to pre-pool spike maps when needed.
        """
        cache: List[Optional[dict]] = [None] * self.num_layers
        for idx, spec in enumerate(self.layer_specs):
            pool = spec.pool
            if pool is None:
                continue
            if not isinstance(pool, nn.MaxPool2d):
                raise TypeError(
                    "STOPTrainer currently supports MaxPool2d in conv blocks."
                )
            pooled, indices = F.max_pool2d(
                spk_rec[idx],
                kernel_size=pool.kernel_size,
                stride=pool.stride,
                padding=pool.padding,
                dilation=pool.dilation,
                ceil_mode=pool.ceil_mode,
                return_indices=True,
            )
            cache[idx] = {
                "indices": indices,
                "input_shape": tuple(spk_rec[idx].shape),
                "pooled_shape": tuple(pooled.shape),
                "kernel_size": pool.kernel_size,
                "stride": pool.stride,
                "padding": pool.padding,
            }
        return cache

    def _propagate_from_next(
        self,
        delta_next: torch.Tensor,
        next_layer_idx: int,
        current_layer_idx: int,
        current_spk: torch.Tensor,
        pool_cache: List[Optional[dict]],
    ) -> torch.Tensor:
        """
        Eq. (16): backprop term from layer l+1 to l (same timestep only).
        """
        next_spec = self.layer_specs[next_layer_idx]
        current_spec = self.layer_specs[current_layer_idx]

        if next_spec.layer_type == "linear":
            # Linear: sum_k delta_{k}^{l+1} * w_{k,j}^{l+1}
            err = torch.matmul(delta_next, next_spec.synapse.weight)
        else:
            # Conv: transpose convolution with next-layer kernels.
            err = F.conv_transpose2d(
                delta_next,
                next_spec.synapse.weight,
                stride=next_spec.synapse.stride,
                padding=next_spec.synapse.padding,
                dilation=next_spec.synapse.dilation,
                groups=next_spec.synapse.groups,
            )

        # If the current layer feeds a pooled representation to the next layer,
        # unpool the error back to current spike resolution.
        if current_spec.layer_type == "conv":
            cache = pool_cache[current_layer_idx]
            if cache is not None:
                if err.dim() == 2:
                    err = err.view(cache["pooled_shape"])
                if tuple(err.shape) != tuple(cache["pooled_shape"]):
                    # Shape alignment fallback for odd border cases.
                    if err.dim() == 4:
                        err = F.interpolate(
                            err,
                            size=cache["pooled_shape"][2:],
                            mode="nearest",
                        )
                    if tuple(err.shape) != tuple(cache["pooled_shape"]):
                        raise RuntimeError(
                            "STOP spatial backprop shape mismatch before unpool: "
                            f"got {tuple(err.shape)}, expected {cache['pooled_shape']}."
                        )
                err = F.max_unpool2d(
                    err,
                    cache["indices"],
                    kernel_size=cache["kernel_size"],
                    stride=cache["stride"],
                    padding=cache["padding"],
                    output_size=cache["input_shape"],
                )
            elif err.dim() == 2:
                err = err.view_as(current_spk)
        else:
            if err.dim() > 2:
                err = err.flatten(1)

        if tuple(err.shape) != tuple(current_spk.shape):
            raise RuntimeError(
                "STOP spatial backprop shape mismatch at current layer: "
                f"got {tuple(err.shape)}, expected {tuple(current_spk.shape)}."
            )
        return err

    def _prepare_temporal_input(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Accept either [T, B, ...] or [B, ...] and return [T, B, ...].
        """
        if data.dim() >= 3 and data.shape[1] == target.shape[0]:
            return data
        if data.shape[0] == target.shape[0]:
            # Static input: repeat across configured timesteps.
            return data.unsqueeze(0).repeat(self.static_input_timesteps, *([1] * data.dim()))
        raise ValueError(
            "Expected data shaped [T, B, ...] or [B, ...] aligned with target."
        )

    def _lr_scale(self, base_lr: float) -> float:
        if not self.cosine_schedule:
            return base_lr
        step = min(self._global_step, self.cosine_t_max)
        scale = 0.5 * (1.0 + math.cos(math.pi * step / float(self.cosine_t_max)))
        return base_lr * scale

    def _with_momentum(self, key: str, grad_like: torch.Tensor) -> torch.Tensor:
        if self.momentum <= 0.0:
            return grad_like
        buf = self._momentum_buffers.get(key)
        if buf is None or tuple(buf.shape) != tuple(grad_like.shape):
            buf = torch.zeros_like(grad_like)
            self._momentum_buffers[key] = buf
        buf.mul_(self.momentum).add_(grad_like)
        return buf

    def _sanity_check(self) -> None:
        """Lightweight runtime checks requested in the task."""
        for idx, spec in enumerate(self.layer_specs):
            if torch.isnan(spec.synapse.weight).any():
                raise RuntimeError(f"NaN detected in layer-{idx} weights.")

            theta = getattr(spec.neuron, "threshold")
            alpha = getattr(spec.neuron, "beta")
            if isinstance(theta, torch.Tensor):
                if torch.isnan(theta).any():
                    raise RuntimeError(f"NaN detected in layer-{idx} threshold.")
                if torch.min(theta).item() < self.threshold_min - 1e-8:
                    raise RuntimeError(f"Threshold clamp violated in layer-{idx}.")
            elif float(theta) < self.threshold_min - 1e-8:
                raise RuntimeError(f"Threshold clamp violated in layer-{idx}.")

            if isinstance(alpha, torch.Tensor):
                if torch.isnan(alpha).any():
                    raise RuntimeError(f"NaN detected in layer-{idx} leakage.")
                if torch.min(alpha).item() < -1e-8 or torch.max(alpha).item() > 1.0 + 1e-8:
                    raise RuntimeError(f"Leakage clamp violated in layer-{idx}.")
            else:
                aval = float(alpha)
                if aval < -1e-8 or aval > 1.0 + 1e-8:
                    raise RuntimeError(f"Leakage clamp violated in layer-{idx}.")

    @torch.no_grad()
    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        STOP Algorithm 1 over one batch sequence.

        Args:
            data: [T, B, ...] or [B, ...]
            target: [B]
        """
        data = self._prepare_temporal_input(data, target)
        num_timesteps, batch_size = data.shape[0], data.shape[1]
        device = data.device
        dtype = data.dtype

        target_onehot = F.one_hot(target, num_classes=self.n_classes).to(dtype=dtype)

        # Algorithm 1 line 1: reset model state and traces.
        self.network.reset()

        w_trace: List[torch.Tensor] = []
        theta_trace: List[torch.Tensor] = []
        alpha_trace: List[torch.Tensor] = []
        prev_spk: List[torch.Tensor] = []
        prev_mem: List[torch.Tensor] = []
        trace_initialized = False

        d_w = [torch.zeros_like(spec.synapse.weight) for spec in self.layer_specs]
        d_theta: List[torch.Tensor] = []
        d_alpha_scalar = [
            torch.zeros((), device=device, dtype=dtype) for _ in self.layer_specs
        ]
        for spec in self.layer_specs:
            theta_param = getattr(spec.neuron, "threshold")
            if isinstance(theta_param, torch.Tensor):
                d_theta.append(torch.zeros_like(theta_param))
            else:
                d_theta.append(torch.zeros((), device=device, dtype=dtype))

        spk_sum = torch.zeros(batch_size, self.n_classes, device=device, dtype=dtype)
        total_loss = torch.zeros((), device=device, dtype=dtype)

        for t in range(num_timesteps):
            x_t = data[t]

            # Forward one step and capture per-layer tensors externally via hooks.
            self._clear_step_hooks()
            out = self.network(x_t)

            # Prefer explicit forward return when available, else use hooks.
            spk_rec = None
            mem_rec = None
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                if isinstance(out[0], (tuple, list)) and isinstance(out[1], (tuple, list)):
                    spk_rec = list(out[0])
                    mem_rec = list(out[1])
            if spk_rec is None or mem_rec is None:
                spk_rec = list(self._hook_spikes)
                mem_rec = list(self._hook_mems)

            if (
                len(spk_rec) != self.num_layers
                or len(mem_rec) != self.num_layers
                or len(self._hook_inputs) != self.num_layers
            ):
                raise RuntimeError(
                    "STOPTrainer failed to capture per-layer states. "
                    f"spk={len(spk_rec)}, mem={len(mem_rec)}, inputs={len(self._hook_inputs)}, "
                    f"expected={self.num_layers}."
                )

            layer_inputs = list(self._hook_inputs)
            pool_cache = self._compute_pool_cache(spk_rec)

            if not trace_initialized:
                w_trace = [torch.zeros_like(inp) for inp in layer_inputs]
                theta_trace = [torch.zeros_like(spk) for spk in spk_rec]
                alpha_trace = [torch.zeros_like(spk) for spk in spk_rec]
                prev_spk = [torch.zeros_like(spk) for spk in spk_rec]
                prev_mem = [torch.zeros_like(mem) for mem in mem_rec]
                trace_initialized = True

            # Eq. (20), (26), (30): forward-only temporal trace recurrences.
            theta_expanded_per_layer: List[torch.Tensor] = []
            for l_idx, spec in enumerate(self.layer_specs):
                alpha_l = self._alpha_tensor(spec.neuron, device=device, dtype=dtype).mean()
                theta_l = self._theta_tensor(spec.neuron, device=device, dtype=dtype)
                theta_broadcast = self._expand_theta(theta_l, spk_rec[l_idx], spec.layer_type)
                theta_expanded_per_layer.append(theta_broadcast)

                w_trace[l_idx] = alpha_l * w_trace[l_idx] + layer_inputs[l_idx]
                theta_trace[l_idx] = alpha_l * (theta_trace[l_idx] - prev_spk[l_idx])
                alpha_trace[l_idx] = alpha_l * alpha_trace[l_idx] + (
                    prev_mem[l_idx] - theta_broadcast * prev_spk[l_idx]
                )

            # Instantaneous output loss E[t].
            spk_out = spk_rec[-1]
            loss_t, grad_out_spk = self._loss_and_output_error(
                spk_out, target, target_onehot
            )
            total_loss = total_loss + loss_t
            spk_sum = spk_sum + spk_out

            # Eq. (11): delta^L[t] = dE/ds^L[t] * phi(U^L[t] - theta^L).
            deltas: List[torch.Tensor] = [torch.zeros_like(spk) for spk in spk_rec]
            deltas[-1] = grad_out_spk * self._surrogate_grad(
                mem_rec[-1] - theta_expanded_per_layer[-1]
            )

            # Eq. (16): hidden spatial deltas at current t only (no temporal backprop).
            for l_idx in range(self.num_layers - 2, -1, -1):
                backprop_term = self._propagate_from_next(
                    delta_next=deltas[l_idx + 1],
                    next_layer_idx=l_idx + 1,
                    current_layer_idx=l_idx,
                    current_spk=spk_rec[l_idx],
                    pool_cache=pool_cache,
                )
                deltas[l_idx] = backprop_term * self._surrogate_grad(
                    mem_rec[l_idx] - theta_expanded_per_layer[l_idx]
                )

            # Eq. (17), (24), (27): accumulate parameter deltas across timesteps.
            for l_idx, spec in enumerate(self.layer_specs):
                delta_l = deltas[l_idx]

                if self.learn_weights and spec.synapse.weight.requires_grad:
                    if spec.layer_type == "linear":
                        grad_w = torch.matmul(delta_l.transpose(0, 1), w_trace[l_idx])
                    else:
                        grad_w = torch.nn.grad.conv2d_weight(
                            w_trace[l_idx],
                            spec.synapse.weight.shape,
                            delta_l,
                            stride=spec.synapse.stride,
                            padding=spec.synapse.padding,
                            dilation=spec.synapse.dilation,
                            groups=spec.synapse.groups,
                        )
                    d_w[l_idx] = d_w[l_idx] + grad_w / float(batch_size)

                if self.learn_thresholds:
                    raw_dtheta = delta_l * (theta_trace[l_idx] - 1.0)
                    theta_param = getattr(spec.neuron, "threshold")
                    reduced = self._reduce_threshold_update(
                        raw_dtheta, theta_param, spec.layer_type
                    )
                    d_theta[l_idx] = d_theta[l_idx] + reduced

                if self.learn_leakage:
                    raw_dalpha = delta_l * alpha_trace[l_idx]
                    # Paper Algorithm 1 lines 30-33: average over all neurons.
                    d_alpha_scalar[l_idx] = d_alpha_scalar[l_idx] + raw_dalpha.mean()

            # Keep previous states for trace recurrences at t+1.
            prev_spk = [spk.detach() for spk in spk_rec]
            prev_mem = [mem.detach() for mem in mem_rec]

        # Apply updates once after full sequence (Algorithm 1 lines 34-42).
        lr_w = self._lr_scale(self.lr_weight)
        lr_theta = self._lr_scale(self.lr_threshold)
        lr_alpha = self._lr_scale(self.lr_leakage)

        for l_idx, spec in enumerate(self.layer_specs):
            if self.learn_weights and spec.synapse.weight.requires_grad:
                upd_w = self._with_momentum(f"W_{l_idx}", d_w[l_idx])
                spec.synapse.weight.data.add_(-lr_w * upd_w)

            if self.learn_thresholds:
                theta_cur = self._theta_tensor(spec.neuron, device=device, dtype=dtype)
                upd_theta = self._with_momentum(f"THETA_{l_idx}", d_theta[l_idx])
                theta_new = torch.clamp(theta_cur - lr_theta * upd_theta, min=self.threshold_min)
                self._set_neuron_attr(spec.neuron, "threshold", theta_new)

            if self.learn_leakage:
                alpha_cur = self._alpha_tensor(spec.neuron, device=device, dtype=dtype)
                upd_alpha = self._with_momentum(f"ALPHA_{l_idx}", d_alpha_scalar[l_idx])
                alpha_new = torch.clamp(alpha_cur - lr_alpha * upd_alpha, min=0.0, max=1.0)
                self._set_neuron_attr(spec.neuron, "beta", alpha_new)

        self._global_step += 1
        self._sanity_check()

        pred = spk_sum.argmax(dim=1, keepdim=True)
        return (total_loss / float(num_timesteps)).detach(), pred

    @torch.no_grad()
    def train_step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Public one-batch helper."""
        data, target = batch
        if data.dim() >= 3 and data.shape[0] == target.shape[0]:
            data = data.transpose(0, 1)
        return self.train_sample(data, target)

    @torch.no_grad()
    def train_epoch(
        self,
        dataloader,
        device: Optional[torch.device | str] = None,
        print_every: Optional[int] = None,
    ) -> dict:
        """Public epoch helper compatible with the framework metrics style."""
        if device is not None:
            self.to(device)
        self.network.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in dataloader:
            if device is not None:
                non_blocking = (
                    device == "cuda"
                    if isinstance(device, str)
                    else getattr(device, "type", None) == "cuda"
                )
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
            temporal = (
                data.transpose(0, 1)
                if data.dim() >= 3 and data.shape[0] == target.shape[0]
                else data
            )
            loss, pred = self.train_sample(temporal, target)
            batch_size = target.shape[0]
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_correct += int(pred.eq(target.view_as(pred)).sum().item())

            if print_every and total_samples % print_every == 0:
                print(
                    f"[STOP Train] samples={total_samples} "
                    f"loss={total_loss / total_samples:.4f} "
                    f"acc={total_correct / total_samples:.4f}"
                )

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        device: Optional[torch.device | str] = None,
        print_every: Optional[int] = None,
    ) -> dict:
        """Public evaluation helper (no parameter updates)."""
        if device is not None:
            self.to(device)
        self.network.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in dataloader:
            if device is not None:
                non_blocking = (
                    device == "cuda"
                    if isinstance(device, str)
                    else getattr(device, "type", None) == "cuda"
                )
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
            temporal = (
                data.transpose(0, 1)
                if data.dim() >= 3 and data.shape[0] == target.shape[0]
                else data
            )

            temporal = self._prepare_temporal_input(temporal, target)
            self.network.reset()

            spk_sum = torch.zeros(
                target.shape[0], self.n_classes, device=temporal.device, dtype=temporal.dtype
            )
            loss_sum = torch.zeros((), device=temporal.device, dtype=temporal.dtype)
            onehot = F.one_hot(target, num_classes=self.n_classes).to(dtype=temporal.dtype)

            for t in range(temporal.shape[0]):
                self._clear_step_hooks()
                out = self.network(temporal[t])
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    spk_rec = list(out[0]) if isinstance(out[0], (tuple, list)) else list(self._hook_spikes)
                else:
                    spk_rec = list(self._hook_spikes)
                spk_out = spk_rec[-1]
                spk_sum = spk_sum + spk_out
                loss_t, _ = self._loss_and_output_error(spk_out, target, onehot)
                loss_sum = loss_sum + loss_t

            pred = spk_sum.argmax(dim=1, keepdim=True)
            batch_size = target.shape[0]
            total_samples += batch_size
            total_loss += float((loss_sum / float(temporal.shape[0])).item()) * batch_size
            total_correct += int(pred.eq(target.view_as(pred)).sum().item())

            if print_every and total_samples % print_every == 0:
                print(
                    f"[STOP Eval] samples={total_samples} "
                    f"loss={total_loss / total_samples:.4f} "
                    f"acc={total_correct / total_samples:.4f}"
                )

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }

    def reset(self) -> None:
        """Reset model states before each batch."""
        self.network.reset()

    def checkpoint_state(self) -> dict:
        """Save trainer-specific state (momentum + scheduler step)."""
        if not self._momentum_buffers:
            return {"global_step": self._global_step}
        return {
            "global_step": self._global_step,
            "momentum_buffers": {k: v.detach().clone() for k, v in self._momentum_buffers.items()},
        }

    def load_checkpoint_state(self, state: dict) -> None:
        """Restore trainer-specific state."""
        self._global_step = int(state.get("global_step", 0))
        mb = state.get("momentum_buffers", {})
        self._momentum_buffers = {
            k: v.detach().clone() for k, v in mb.items() if isinstance(v, torch.Tensor)
        }

    def __del__(self):
        # Best-effort hook cleanup to avoid dangling references in long sessions.
        for handle in getattr(self, "_hook_handles", []):
            try:
                handle.remove()
            except Exception:
                pass
