"""
STOP trainer for feed-forward SNNs (STOP-WTL / STOP-W / STOP-WT / STOP-WL).

Implements Algorithm 1 from:
  Gao et al. (2025), "STOP: spatiotemporal orthogonal propagation for
  weight-threshold-leakage synergistic training of deep SNNs".

This implementation keeps STOP parameter updates fully manual:
  - No optimizer.step()
  - No BPTT
  - Optional autograd is used only per-timestep to extract spatial deltas
    dE[t]/dU^l[t], then detached immediately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import surrogate as snn_surrogate

from trainers.base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# STOP-specific surrogate gradients (Eq. 3 of Gao et al., 2025)
# ---------------------------------------------------------------------------

def _exp_surrogate(input_, grad_input, spikes):
    """Eq. (3): phi(x) = exp(-|x|)."""
    return grad_input * torch.exp(-torch.abs(input_))


def _rational_surrogate(input_, grad_input, spikes):
    """Rational fallback used in some ablations."""
    return grad_input / (1.0 + (torch.pi * input_).pow(2))


def get_stop_spike_grad(name: str = "exp"):
    """Return a snnTorch-compatible surrogate callable."""
    key = str(name).lower()
    if key == "exp":
        return snn_surrogate.custom_surrogate(_exp_surrogate)
    if key == "rational":
        return snn_surrogate.custom_surrogate(_rational_surrogate)
    raise ValueError(f"Unknown STOP surrogate '{name}'.")


@dataclass
class _LayerSpec:
    """One trainable STOP layer (synapse + spiking neuron)."""

    synapse: nn.Module
    neuron: nn.Module
    layer_type: str  # "linear" or "conv"
    pool: Optional[nn.Module] = None
    mem_post_reset: bool = False
    update_neuron_params: bool = True


class STOPTrainer(BaseTrainer):
    """
    SpatioTemporal Orthogonal Propagation trainer.

    Notation mapping:
      - U^l[t]            -> mem_rec[l]
      - s^l[t]            -> spk_rec[l]
      - delta^l[t]        -> deltas[l]
      - w_tilde^l[t]      -> w_trace[l]          (Eq. 20)
      - theta_tilde^l[t]  -> theta_trace[l]      (Eq. 26)
      - alpha_tilde^l[t]  -> alpha_trace[l]      (Eq. 30)
      - DeltaW^l          -> d_w[l]              (Eq. 17)
      - DeltaTheta^l      -> d_theta[l]          (Eq. 24)
      - DeltaAlpha^l      -> d_alpha_scalar[l]   (Eq. 27)
    """

    _VALID_SURROGATES = ("exp", "rational")
    _VALID_LOSSES = ("ce", "mse")
    _VALID_SPATIAL_MODES = ("autograd", "manual_chain")

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
        use_autograd_spatial: bool = False,
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
        if self.cosine_schedule and self.cosine_t_max <= 0:
            raise ValueError("cosine_t_max must be > 0 when cosine_schedule=True.")
        if (
            not self.learn_weights
            and not self.learn_thresholds
            and not self.learn_leakage
        ):
            raise ValueError(
                "At least one of learn_weights/thresholds/leakage must be True."
            )

        self.layer_specs = self._resolve_layer_specs()
        self.num_layers = len(self.layer_specs)
        self.n_classes = int(getattr(network, "n_classes"))

        has_residual = bool(getattr(self.network, "has_residual_connections", False))
        has_pool = any(spec.pool is not None for spec in self.layer_specs)
        self.spatial_delta_mode = (
            "autograd"
            if (use_autograd_spatial or has_residual or has_pool)
            else "manual_chain"
        )
        if self.spatial_delta_mode not in self._VALID_SPATIAL_MODES:
            raise ValueError(
                f"spatial_delta_mode must be one of {self._VALID_SPATIAL_MODES}."
            )

        # Hooks store either detached or raw tensors based on this runtime mode.
        self._spatial_mode = self.spatial_delta_mode
        self._hook_inputs: Dict[nn.Module, torch.Tensor] = {}
        self._hook_spikes: Dict[nn.Module, torch.Tensor] = {}
        self._hook_mems: Dict[nn.Module, torch.Tensor] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_state_hooks()

        # In autograd spatial mode, dE/dU must use STOP surrogate Eq. (3).
        if self.spatial_delta_mode == "autograd":
            stop_spike_grad = (
                get_stop_spike_grad("exp")
                if self.surrogate == "exp"
                else snn_surrogate.custom_surrogate(
                    lambda input_, grad_input, spikes: grad_input
                    / (1.0 + (torch.pi * input_).pow(2))
                )
            )
            for spec in self.layer_specs:
                if hasattr(spec.neuron, "spike_grad"):
                    spec.neuron.spike_grad = stop_spike_grad

        self._momentum_buffers: Dict[str, torch.Tensor] = {}
        self._global_step = 0

    def _resolve_layer_specs(self) -> List[_LayerSpec]:
        """
        Resolve trainable synapse+neuron pairs in STOP update order.

        Preferred path: network.stop_layer_specs (for custom topologies e.g. ResNet).
        Fallbacks: ConvFCNetwork blocks or alternating [synapse, neuron] lists.
        """
        specs: List[_LayerSpec] = []

        if hasattr(self.network, "stop_layer_specs"):
            for raw in list(getattr(self.network, "stop_layer_specs")):
                if isinstance(raw, dict):
                    syn = raw["synapse"]
                    neu = raw["neuron"]
                    layer_type = raw.get("layer_type")
                    pool = raw.get("pool", None)
                    update_neuron_params = bool(raw.get("update_neuron_params", True))
                elif isinstance(raw, (tuple, list)):
                    if len(raw) < 3:
                        raise ValueError(
                            "stop_layer_specs tuple format must be "
                            "(synapse, neuron, layer_type[, pool])."
                        )
                    syn, neu, layer_type = raw[0], raw[1], raw[2]
                    pool = raw[3] if len(raw) > 3 else None
                    update_neuron_params = True
                else:
                    raise TypeError(
                        "Each stop_layer_specs entry must be dict/tuple/list."
                    )
                if layer_type not in ("linear", "conv"):
                    raise ValueError(
                        f"Invalid layer_type '{layer_type}' in stop_layer_specs."
                    )
                specs.append(
                    _LayerSpec(
                        synapse=syn,
                        neuron=neu,
                        layer_type=layer_type,
                        pool=pool,
                        mem_post_reset=bool(getattr(neu, "reset_delay", True) is False),
                        update_neuron_params=update_neuron_params,
                    )
                )

        # ConvFCNetwork explicit block path.
        elif hasattr(self.network, "conv_blocks") and hasattr(self.network, "fc_blocks"):
            for block in getattr(self.network, "conv_blocks"):
                if len(block) != 3:
                    raise ValueError("Expected conv block format [Conv2d, LIF, Pool].")
                conv, lif, pool = block
                specs.append(
                    _LayerSpec(
                        synapse=conv,
                        neuron=lif,
                        layer_type="conv",
                        pool=pool,
                        mem_post_reset=bool(getattr(lif, "reset_delay", True) is False),
                        update_neuron_params=True,
                    )
                )
            for block in getattr(self.network, "fc_blocks"):
                if len(block) != 2:
                    raise ValueError("Expected FC block format [Linear, LIF].")
                fc, lif = block
                specs.append(
                    _LayerSpec(
                        synapse=fc,
                        neuron=lif,
                        layer_type="linear",
                        pool=None,
                        mem_post_reset=bool(getattr(lif, "reset_delay", True) is False),
                        update_neuron_params=True,
                    )
                )

        # FC-like alternating path.
        elif hasattr(self.network, "layers"):
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
                layer_type = "conv" if isinstance(syn, nn.Conv2d) else "linear"
                specs.append(
                    _LayerSpec(
                        synapse=syn,
                        neuron=neuron,
                        layer_type=layer_type,
                        pool=None,
                        mem_post_reset=bool(
                            getattr(neuron, "reset_delay", True) is False
                        ),
                        update_neuron_params=True,
                    )
                )
                idx += 2

        else:
            raise TypeError(
                "Unsupported network structure for STOPTrainer. Expected stop_layer_specs, "
                "conv_blocks/fc_blocks, or alternating layers."
            )

        if not specs:
            raise ValueError("No trainable layers found for STOP.")

        for spec in specs:
            if not hasattr(spec.neuron, "beta") or not hasattr(spec.neuron, "threshold"):
                raise TypeError(
                    "STOP expects neuron modules exposing both 'beta' and 'threshold'."
                )
        return specs

    def _register_state_hooks(self) -> None:
        """Capture per-layer synapse inputs and neuron outputs each timestep."""

        def _synapse_hook(module, inputs, output):
            if not inputs:
                raise RuntimeError("Synapse hook did not receive inputs.")
            x = inputs[0]
            self._hook_inputs[module] = x if self._spatial_mode == "autograd" else x.detach()

        def _neuron_hook(module, inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError("Neuron output must be (spike, membrane) for STOP.")
            spk, mem = output[0], output[1]
            if self._spatial_mode == "autograd":
                if mem.requires_grad:
                    mem.retain_grad()
                self._hook_spikes[module] = spk
                self._hook_mems[module] = mem
            else:
                self._hook_spikes[module] = spk.detach()
                self._hook_mems[module] = mem.detach()

        seen_synapses = set()
        seen_neurons = set()
        for spec in self.layer_specs:
            syn_id = id(spec.synapse)
            neu_id = id(spec.neuron)
            if syn_id not in seen_synapses:
                self._hook_handles.append(spec.synapse.register_forward_hook(_synapse_hook))
                seen_synapses.add(syn_id)
            if neu_id not in seen_neurons:
                self._hook_handles.append(spec.neuron.register_forward_hook(_neuron_hook))
                seen_neurons.add(neu_id)

    def _clear_step_cache(self) -> None:
        self._hook_inputs.clear()
        self._hook_spikes.clear()
        self._hook_mems.clear()

    def _prepare_temporal_input(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Accept [T, B, ...] or [B, ...], return [T, B, ...]."""
        if data.dim() >= 3 and data.shape[1] == target.shape[0]:
            return data
        if data.shape[0] == target.shape[0]:
            return data.unsqueeze(0).repeat(
                self.static_input_timesteps, *([1] * data.dim())
            )
        raise ValueError("Expected data shaped [T, B, ...] or [B, ...] aligned with target.")

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

    @staticmethod
    def _expand_theta(
        theta: torch.Tensor, out_spikes: torch.Tensor, layer_type: str
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

        candidate = theta.view((1,) + tuple(theta.shape))
        _ = candidate + out_spikes
        return candidate

    @staticmethod
    def _reduce_threshold_update(
        raw_update: torch.Tensor, theta_param, layer_type: str
    ) -> torch.Tensor:
        """Reduce DeltaTheta map to parameter shape."""
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
                for dim in range(1, reduced.dim()):
                    if target.shape[dim] == 1 and reduced.shape[dim] != 1:
                        reduced = reduced.mean(dim=dim, keepdim=True)
                return reduced.reshape_as(target)

        mean_no_batch = raw_update.mean(dim=0)
        if mean_no_batch.numel() == target.numel():
            return mean_no_batch.reshape_as(target)
        raise ValueError(
            f"Cannot reduce threshold update {tuple(raw_update.shape)} to "
            f"theta shape {tuple(target.shape)}."
        )

    @staticmethod
    def _set_neuron_attr(neuron: nn.Module, attr_name: str, value: torch.Tensor) -> None:
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
                    f"to shape {tuple(target.shape)}."
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
        """Eq. (3) surrogate derivative for manual spatial propagation."""
        if self.surrogate == "exp":
            return torch.exp(-x.abs())
        return 1.0 / (1.0 + (math.pi * x).pow(2))

    def _loss_and_output_error(
        self,
        out_spikes: torch.Tensor,
        target: torch.Tensor,
        target_onehot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Instantaneous E[t] and dE/ds^L[t] for manual delta path."""
        if self.loss_type == "ce":
            loss_t = F.cross_entropy(out_spikes, target, reduction="mean")
            grad_out = torch.softmax(out_spikes, dim=1) - target_onehot
            return loss_t, grad_out

        diff = out_spikes - target_onehot
        loss_t = 0.5 * diff.pow(2).sum(dim=1).mean()
        return loss_t, diff

    @staticmethod
    def _alpha_trace_update(
        prev_trace: torch.Tensor,
        alpha_l: torch.Tensor,
        prev_mem: torch.Tensor,
        prev_spk: torch.Tensor,
        theta_broadcast: torch.Tensor,
        mem_is_post_reset: bool,
    ) -> torch.Tensor:
        """
        Eq. (30) update term with membrane semantics guard.

        STOP uses:
          alpha_tilde[t] = alpha * alpha_tilde[t-1] + (U[t-1] - theta * s[t-1])

        For snnTorch neurons:
          - reset_delay=True  -> returned mem is pre-reset wrt current spike, use (mem - theta*s)
          - reset_delay=False -> returned mem is already post-reset, use mem directly
        """
        membrane_term = prev_mem if mem_is_post_reset else (prev_mem - theta_broadcast * prev_spk)
        return alpha_l * prev_trace + membrane_term

    def _detach_hidden_state(self) -> None:
        """
        Detach recurrent state buffers to prevent temporal graph chaining.

        This enforces STOP's no-BPTT requirement when autograd spatial deltas are used.
        """
        if hasattr(self.network, "detach_hidden") and callable(self.network.detach_hidden):
            self.network.detach_hidden()
            return

        classes = {spec.neuron.__class__ for spec in self.layer_specs}
        for cls in classes:
            detach_hidden = getattr(cls, "detach_hidden", None)
            if callable(detach_hidden):
                try:
                    detach_hidden()
                except Exception:
                    pass

        for spec in self.layer_specs:
            for attr in ("mem", "syn"):
                state = getattr(spec.neuron, attr, None)
                if isinstance(state, torch.Tensor):
                    state.detach_()

    def _compute_spatial_deltas_autograd(
        self, loss_t: torch.Tensor, mem_rec: Sequence[torch.Tensor], batch_size: int
    ) -> List[torch.Tensor]:
        """Option 1: per-timestep autograd deltas delta^l[t] = dE[t]/dU^l[t]."""
        self.network.zero_grad(set_to_none=True)
        loss_t.backward()
        deltas: List[torch.Tensor] = []
        for idx, mem in enumerate(mem_rec):
            if mem.grad is None:
                raise RuntimeError(
                    f"STOP autograd spatial mode: missing membrane grad at layer {idx}. "
                    "Ensure hooks keep raw mem tensors and call retain_grad()."
                )
            # loss_t uses reduction="mean", so autograd gives gradients scaled by 1/B.
            # STOP Eq. (11)/(16)/(17)/(24)/(27) accumulators expect per-sample sums;
            # rescale by batch size to match the manual STOP path.
            deltas.append(mem.grad.detach() * float(batch_size))
        self.network.zero_grad(set_to_none=True)
        return deltas

    def _compute_spatial_deltas_manual(
        self,
        spk_rec: Sequence[torch.Tensor],
        mem_rec: Sequence[torch.Tensor],
        theta_broadcast: Sequence[torch.Tensor],
        grad_out_spk: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Manual chain-only STOP Eq. (11) and Eq. (16)."""
        deltas = [torch.zeros_like(spk) for spk in spk_rec]
        deltas[-1] = grad_out_spk * self._surrogate_grad(mem_rec[-1] - theta_broadcast[-1])

        for l_idx in range(self.num_layers - 2, -1, -1):
            next_spec = self.layer_specs[l_idx + 1]
            current_spk = spk_rec[l_idx]
            delta_next = deltas[l_idx + 1]

            if next_spec.layer_type == "linear":
                err = torch.matmul(delta_next.flatten(1), next_spec.synapse.weight)
            else:
                err = F.conv_transpose2d(
                    delta_next,
                    next_spec.synapse.weight,
                    stride=next_spec.synapse.stride,
                    padding=next_spec.synapse.padding,
                    dilation=next_spec.synapse.dilation,
                    groups=next_spec.synapse.groups,
                )

            if current_spk.dim() > 2 and err.dim() == 2:
                expected = int(torch.tensor(current_spk.shape[1:]).prod().item())
                if err.shape[1] != expected:
                    raise RuntimeError(
                        "STOP manual_chain shape mismatch. "
                        f"Layer {l_idx}: cannot reshape {tuple(err.shape)} "
                        f"to {tuple(current_spk.shape)}."
                    )
                err = err.view_as(current_spk)
            elif current_spk.dim() == 2 and err.dim() > 2:
                err = err.flatten(1)

            if tuple(err.shape) != tuple(current_spk.shape):
                raise RuntimeError(
                    "STOP manual_chain shape mismatch. "
                    f"Layer {l_idx}: got {tuple(err.shape)}, expected {tuple(current_spk.shape)}."
                )

            deltas[l_idx] = err * self._surrogate_grad(
                mem_rec[l_idx] - theta_broadcast[l_idx]
            )

        return deltas

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

    def train_sample(self, data: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        STOP Algorithm 1 over one batch sequence.

        Flow:
          init traces -> timestep loop (forward, Eq.20/26/30 traces, Eq.11/16 deltas,
          Eq.17/24/27 accumulators) -> apply updates.
        """
        data = self._prepare_temporal_input(data, target)
        num_timesteps, batch_size = data.shape[0], data.shape[1]
        device = data.device
        dtype = data.dtype

        target_onehot = F.one_hot(target, num_classes=self.n_classes).to(dtype=dtype)

        self.network.reset()

        trace_initialized = False
        w_trace: List[torch.Tensor] = []
        theta_trace: List[torch.Tensor] = []
        alpha_trace: List[torch.Tensor] = []
        prev_spk: List[torch.Tensor] = []
        prev_mem: List[torch.Tensor] = []

        d_w = [torch.zeros_like(spec.synapse.weight) for spec in self.layer_specs]
        d_theta: List[torch.Tensor] = []
        d_alpha_scalar = [
            torch.zeros((), device=device, dtype=dtype) for _ in self.layer_specs
        ]
        for spec in self.layer_specs:
            theta_param = getattr(spec.neuron, "threshold")
            d_theta.append(
                torch.zeros_like(theta_param)
                if isinstance(theta_param, torch.Tensor)
                else torch.zeros((), device=device, dtype=dtype)
            )

        spk_sum = torch.zeros(batch_size, self.n_classes, device=device, dtype=dtype)
        total_loss = torch.zeros((), device=device, dtype=dtype)

        for t in range(num_timesteps):
            x_t = data[t]

            # No BPTT: detach hidden states before each timestep.
            self._detach_hidden_state()
            self._spatial_mode = self.spatial_delta_mode
            self._clear_step_cache()

            if self.spatial_delta_mode == "autograd":
                with torch.enable_grad():
                    out = self.network(x_t)
            else:
                with torch.no_grad():
                    out = self.network(x_t)

            if isinstance(out, (tuple, list)) and len(out) >= 2:
                out_spk = out[0] if isinstance(out[0], (tuple, list)) else None
                out_mem = out[1] if isinstance(out[1], (tuple, list)) else None
            else:
                out_spk, out_mem = None, None

            layer_inputs: List[torch.Tensor] = []
            spk_rec: List[torch.Tensor] = []
            mem_rec: List[torch.Tensor] = []
            for l_idx, spec in enumerate(self.layer_specs):
                inp = self._hook_inputs.get(spec.synapse)
                spk = self._hook_spikes.get(spec.neuron)
                mem = self._hook_mems.get(spec.neuron)

                if inp is None or spk is None or mem is None:
                    if out_spk is None or out_mem is None:
                        raise RuntimeError(
                            "STOPTrainer failed to capture per-layer states from hooks."
                        )
                    if l_idx >= len(out_spk) or l_idx >= len(out_mem):
                        raise RuntimeError(
                            "STOPTrainer fallback from network forward output failed."
                        )
                    spk = out_spk[l_idx]
                    mem = out_mem[l_idx]
                    inp = x_t if l_idx == 0 else out_spk[l_idx - 1]
                layer_inputs.append(inp)
                spk_rec.append(spk)
                mem_rec.append(mem)

            if not trace_initialized:
                w_trace = [torch.zeros_like(inp) for inp in layer_inputs]
                theta_trace = [torch.zeros_like(spk) for spk in spk_rec]
                alpha_trace = [torch.zeros_like(spk) for spk in spk_rec]
                prev_spk = [torch.zeros_like(spk) for spk in spk_rec]
                prev_mem = [torch.zeros_like(mem) for mem in mem_rec]
                trace_initialized = True

            theta_expanded_per_layer: List[torch.Tensor] = []
            for l_idx, spec in enumerate(self.layer_specs):
                alpha_l = self._alpha_tensor(spec.neuron, device, dtype).mean()
                theta_l = self._theta_tensor(spec.neuron, device, dtype)
                theta_broadcast = self._expand_theta(theta_l, spk_rec[l_idx], spec.layer_type)
                theta_expanded_per_layer.append(theta_broadcast)

                # Eq. (20): w_tilde^l[t] = alpha * w_tilde^l[t-1] + x^l[t]
                w_trace[l_idx] = alpha_l * w_trace[l_idx] + layer_inputs[l_idx].detach()

                # Eq. (26): theta_tilde^l[t] = alpha * (theta_tilde^l[t-1] - s^l[t-1])
                theta_trace[l_idx] = alpha_l * (theta_trace[l_idx] - prev_spk[l_idx])

                # Eq. (30): alpha_tilde^l[t] recurrence with mem semantics guard.
                alpha_trace[l_idx] = self._alpha_trace_update(
                    prev_trace=alpha_trace[l_idx],
                    alpha_l=alpha_l,
                    prev_mem=prev_mem[l_idx],
                    prev_spk=prev_spk[l_idx],
                    theta_broadcast=theta_broadcast,
                    mem_is_post_reset=spec.mem_post_reset,
                )

            spk_out = spk_rec[-1]
            if self.spatial_delta_mode == "autograd":
                # Loss must be built under enable_grad to keep dE/dU graph even if
                # global grad mode was disabled elsewhere in the process.
                with torch.enable_grad():
                    loss_t, grad_out_spk = self._loss_and_output_error(
                        spk_out, target, target_onehot
                    )
            else:
                loss_t, grad_out_spk = self._loss_and_output_error(
                    spk_out, target, target_onehot
                )
            total_loss = total_loss + loss_t.detach()
            spk_sum = spk_sum + spk_out.detach()

            if self.spatial_delta_mode == "autograd":
                deltas = self._compute_spatial_deltas_autograd(
                    loss_t, mem_rec, batch_size=batch_size
                )
            else:
                deltas = self._compute_spatial_deltas_manual(
                    spk_rec=spk_rec,
                    mem_rec=mem_rec,
                    theta_broadcast=theta_expanded_per_layer,
                    grad_out_spk=grad_out_spk,
                )

            for l_idx, spec in enumerate(self.layer_specs):
                delta_l = deltas[l_idx].detach()

                # Eq. (17): DeltaW^l += delta^l[t] * w_tilde^{l-1}[t]
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

                # Eq. (24): DeltaTheta^l += delta^l[t] * (theta_tilde^l[t] - 1)
                if self.learn_thresholds and spec.update_neuron_params:
                    raw_dtheta = delta_l * (theta_trace[l_idx] - 1.0)
                    theta_param = getattr(spec.neuron, "threshold")
                    reduced = self._reduce_threshold_update(
                        raw_dtheta, theta_param, spec.layer_type
                    )
                    d_theta[l_idx] = d_theta[l_idx] + reduced

                # Eq. (27): DeltaAlpha^l += mean(delta^l[t] * alpha_tilde^l[t])
                if self.learn_leakage and spec.update_neuron_params:
                    raw_dalpha = delta_l * alpha_trace[l_idx]
                    d_alpha_scalar[l_idx] = d_alpha_scalar[l_idx] + raw_dalpha.mean()

            prev_spk = [spk.detach() for spk in spk_rec]
            prev_mem = [mem.detach() for mem in mem_rec]

        lr_w = self._lr_scale(self.lr_weight)
        lr_theta = self._lr_scale(self.lr_threshold)
        lr_alpha = self._lr_scale(self.lr_leakage)

        with torch.no_grad():
            for l_idx, spec in enumerate(self.layer_specs):
                if self.learn_weights and spec.synapse.weight.requires_grad:
                    upd_w = self._with_momentum(f"W_{l_idx}", d_w[l_idx])
                    spec.synapse.weight.data.add_(-lr_w * upd_w)

                if self.learn_thresholds and spec.update_neuron_params:
                    theta_cur = self._theta_tensor(spec.neuron, device=device, dtype=dtype)
                    upd_theta = self._with_momentum(f"THETA_{l_idx}", d_theta[l_idx])
                    theta_new = torch.clamp(
                        theta_cur - lr_theta * upd_theta, min=self.threshold_min
                    )
                    self._set_neuron_attr(spec.neuron, "threshold", theta_new)

                if self.learn_leakage and spec.update_neuron_params:
                    alpha_cur = self._alpha_tensor(spec.neuron, device=device, dtype=dtype)
                    upd_alpha = self._with_momentum(f"ALPHA_{l_idx}", d_alpha_scalar[l_idx])
                    alpha_new = torch.clamp(alpha_cur - lr_alpha * upd_alpha, min=0.0, max=1.0)
                    self._set_neuron_attr(spec.neuron, "beta", alpha_new)

        self._global_step += 1
        self._sanity_check()

        pred = spk_sum.argmax(dim=1, keepdim=True)
        return (total_loss / float(num_timesteps)).detach(), pred

    def reset(self) -> None:
        self.network.reset()

    def checkpoint_state(self) -> dict:
        if not self._momentum_buffers:
            return {"global_step": self._global_step}
        return {
            "global_step": self._global_step,
            "momentum_buffers": {
                k: v.detach().clone() for k, v in self._momentum_buffers.items()
            },
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self._global_step = int(state.get("global_step", 0))
        mb = state.get("momentum_buffers", {})
        self._momentum_buffers = {
            k: v.detach().clone() for k, v in mb.items() if isinstance(v, torch.Tensor)
        }

    def __del__(self):
        for handle in getattr(self, "_hook_handles", []):
            try:
                handle.remove()
            except Exception:
                pass
