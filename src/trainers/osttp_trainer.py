"""
OSTTP (Online Spatio-Temporal Learning with Target Projection) trainer.

This trainer implements a forward-only, no-BPTT learning rule where each
spiking layer is updated with:

    Delta theta_l ~ sum_t (L_t^l * e_{t,theta_l}^l)

- Output learning signal: analytic dE_t/dy_t^K
- Hidden learning signal: fixed random target projection B_l y*_t
- Eligibility traces: OSTL recursions over time (epsilon/e)

The network is not modified. Layer states and presynaptic inputs are captured
through module introspection + forward hooks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


@dataclass
class _LayerSpec:
    """Metadata for one trainable spiking layer in OSTTP."""

    synapse: nn.Linear
    neuron: nn.Module
    n_in: int
    n_post: int
    reset_mechanism: str
    recurrent_weight: Optional[torch.nn.Parameter]
    threshold_param: Optional[torch.nn.Parameter]


class OSTTPTrainer(BaseTrainer):
    """
    Online Spatio-Temporal Learning with Target Projection (OSTTP).

    Supported layer pattern:
        Linear -> snn.Leaky / snn.RLeaky

    Notes:
    - Convolutional synapses are currently not implemented because the exact
      per-neuron eligibility tensor for OSTL recursions is expensive and not
      represented in this framework yet.
    - RLeaky with `reset_delay=False` is rejected to keep the implementation
      exact with the provided OSTTP equations (which use y_{t-1} reset terms).
    """

    _VALID_PSEUDO = ("tanh", "fast_sigmoid")
    _VALID_OUTPUT_LOSS = ("ce", "bce", "mse")

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        pseudo_derivative: str = "tanh",
        output_loss: str = "ce",
        feedback_scale: float = 1.0,
        feedback_seed: int = 42,
        target_dim: Optional[int] = None,
        grad_clip: float = 0.0,
        use_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()

        pseudo_derivative = str(pseudo_derivative).lower()
        if pseudo_derivative not in self._VALID_PSEUDO:
            raise ValueError(
                f"pseudo_derivative must be one of {self._VALID_PSEUDO}, got {pseudo_derivative}"
            )

        output_loss = str(output_loss).lower()
        if output_loss not in self._VALID_OUTPUT_LOSS:
            raise ValueError(
                f"output_loss must be one of {self._VALID_OUTPUT_LOSS}, got {output_loss}"
            )

        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.pseudo_derivative = pseudo_derivative
        self.output_loss = output_loss
        self.feedback_scale = float(feedback_scale)
        self.feedback_seed = int(feedback_seed)
        self.grad_clip = float(grad_clip)
        self.use_optimizer = bool(use_optimizer)
        self.debug = bool(debug)

        self.layer_specs = self._resolve_osttp_layers()
        if not self.layer_specs:
            raise ValueError(
                "OSTTPTrainer requires at least one Linear -> (Leaky/RLeaky) layer pair."
            )

        self.num_layers = len(self.layer_specs)
        self.output_size = self.layer_specs[-1].n_post

        default_target_dim = int(getattr(self.network, "n_classes", self.output_size))
        self.target_dim: Optional[int] = int(target_dim) if target_dim is not None else default_target_dim

        # Fixed hidden feedback matrices B_l are created once and never trained.
        self._feedback_names: List[str] = []
        self._feedback_ready = False

        self._external_optimizer = optimizer
        if self.use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(self.network.parameters(), lr=self.lr)
        else:
            self.optimizer = None

        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._hook_inputs: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self._hook_spikes: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self._hook_mems: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self._register_layer_hooks()

    def _resolve_osttp_layers(self) -> List[_LayerSpec]:
        """
        Discover Linear->spiking pairs without modifying the network.

        Current framework conventions are covered by:
        - network.layers = [Linear, Leaky/RLeaky, ...]
        - fallback RecurrentSRNN-like (fc_in + lif)
        """
        pairs: List[Tuple[nn.Module, nn.Module]] = []

        if hasattr(self.network, "layers"):
            modules = list(getattr(self.network, "layers"))
            i = 0
            while i < len(modules) - 1:
                syn = modules[i]
                neuron = modules[i + 1]
                if self._is_supported_synapse(syn) and self._is_supported_neuron(neuron):
                    pairs.append((syn, neuron))
                    i += 2
                else:
                    i += 1

        elif hasattr(self.network, "fc_in") and hasattr(self.network, "lif"):
            syn = getattr(self.network, "fc_in")
            neuron = getattr(self.network, "lif")
            if self._is_supported_synapse(syn) and self._is_supported_neuron(neuron):
                pairs.append((syn, neuron))

        specs: List[_LayerSpec] = []
        for synapse, neuron in pairs:
            if isinstance(synapse, nn.Conv2d):
                raise NotImplementedError(
                    "OSTTPTrainer currently supports nn.Linear synapses only. "
                    "Conv2d OSTL eligibility requires explicit unfolded patch traces "
                    "for each postsynaptic unit and is not implemented here."
                )
            if not isinstance(synapse, nn.Linear):
                continue

            if synapse.bias is not None and synapse.bias.requires_grad:
                raise NotImplementedError(
                    "OSTTP equations here update W/H/threshold only; "
                    "trainable synaptic bias is not implemented."
                )

            if getattr(neuron, "reset_delay", True) is False:
                raise NotImplementedError(
                    "OSTTPTrainer expects reset_delay=True to match equations based on y_{t-1}."
                )

            reset_mechanism = str(getattr(neuron, "reset_mechanism", "subtract"))
            if reset_mechanism not in ("zero", "subtract", "none"):
                raise ValueError(
                    f"Unsupported reset_mechanism '{reset_mechanism}' for OSTTP."
                )

            rec_weight = None
            recurrent = getattr(neuron, "recurrent", None)
            if recurrent is not None:
                if not hasattr(recurrent, "weight"):
                    raise NotImplementedError(
                        "RLeaky one-to-one recurrent mode is not supported; "
                        "use all_to_all=True so recurrent.weight exists."
                    )
                rec_weight = recurrent.weight
                if rec_weight.dim() != 2 or rec_weight.shape[0] != rec_weight.shape[1]:
                    raise NotImplementedError(
                        "OSTTP recurrent support requires square recurrent weight matrices."
                    )
                if int(rec_weight.shape[0]) != int(synapse.out_features):
                    raise NotImplementedError(
                        "Recurrent matrix size must match synapse out_features for OSTTP."
                    )

            threshold_param = None
            threshold = getattr(neuron, "threshold", None)
            if isinstance(threshold, torch.nn.Parameter) and threshold.requires_grad:
                threshold_param = threshold

            specs.append(
                _LayerSpec(
                    synapse=synapse,
                    neuron=neuron,
                    n_in=int(synapse.in_features),
                    n_post=int(synapse.out_features),
                    reset_mechanism=reset_mechanism,
                    recurrent_weight=rec_weight,
                    threshold_param=threshold_param,
                )
            )

        return specs

    @staticmethod
    def _is_supported_synapse(module: nn.Module) -> bool:
        return isinstance(module, (nn.Linear, nn.Conv2d))

    @staticmethod
    def _is_supported_neuron(module: nn.Module) -> bool:
        return isinstance(module, (snn.Leaky, snn.RLeaky))

    def _register_layer_hooks(self) -> None:
        for idx, spec in enumerate(self.layer_specs):
            self._hook_handles.append(
                spec.synapse.register_forward_hook(self._make_synapse_hook(idx))
            )
            self._hook_handles.append(
                spec.neuron.register_forward_hook(self._make_neuron_hook(idx))
            )

    def _make_synapse_hook(self, idx: int):
        def _hook(_module, inputs, _output):
            if inputs and torch.is_tensor(inputs[0]):
                self._hook_inputs[idx] = inputs[0].detach()

        return _hook

    def _make_neuron_hook(self, idx: int):
        def _hook(_module, _inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                return
            spk_t, mem_t = output[0], output[1]
            self._hook_spikes[idx] = spk_t.detach()
            self._hook_mems[idx] = mem_t.detach()

        return _hook

    def _clear_step_cache(self) -> None:
        for idx in range(self.num_layers):
            self._hook_inputs[idx] = None
            self._hook_spikes[idx] = None
            self._hook_mems[idx] = None

    def _get_step_tensors(
        self,
        x_t: torch.Tensor,
        forward_out,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        hooks_complete = (
            all(val is not None for val in self._hook_inputs)
            and all(val is not None for val in self._hook_spikes)
            and all(val is not None for val in self._hook_mems)
        )
        if hooks_complete:
            return (
                [val for val in self._hook_inputs if val is not None],
                [val for val in self._hook_spikes if val is not None],
                [val for val in self._hook_mems if val is not None],
            )

        if (
            isinstance(forward_out, (tuple, list))
            and len(forward_out) >= 2
            and isinstance(forward_out[0], (tuple, list))
            and isinstance(forward_out[1], (tuple, list))
        ):
            spk_rec = [t.detach() for t in forward_out[0]]
            mem_rec = [t.detach() for t in forward_out[1]]
            if len(spk_rec) == self.num_layers and len(mem_rec) == self.num_layers:
                layer_inputs = [x_t.detach()]
                layer_inputs.extend(spk_rec[:-1])
                return layer_inputs, spk_rec, mem_rec

        raise RuntimeError(
            "OSTTPTrainer could not capture per-layer (input, spike, membrane) tensors. "
            "Ensure the network uses Linear->Leaky/RLeaky layers reachable by hooks."
        )

    def _prepare_temporal_targets(
        self,
        target: torch.Tensor,
        num_timesteps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build y*_t with shape [T, B, C].

        Supported target formats:
        - [B] class indices
        - [B, C] per-sample vectors (broadcast across time)
        - [T, B, C] per-timestep vectors
        - [B, T, C] per-timestep vectors
        """
        if target.dim() == 1:
            if self.target_dim is None:
                self.target_dim = self.output_size
            y_star = torch.zeros(target.size(0), self.target_dim, device=device, dtype=dtype)
            y_star.scatter_(1, target.view(-1, 1), 1.0)
            return y_star.unsqueeze(0).expand(num_timesteps, -1, -1)

        if target.dim() == 2:
            y_star = target.to(device=device, dtype=dtype)
            self.target_dim = int(y_star.size(-1))
            return y_star.unsqueeze(0).expand(num_timesteps, -1, -1)

        if target.dim() == 3:
            if target.size(0) == num_timesteps:
                y_star = target
            elif target.size(1) == num_timesteps:
                y_star = target.transpose(0, 1)
            else:
                raise ValueError(
                    "3D targets must have shape [T, B, C] or [B, T, C]."
                )
            y_star = y_star.to(device=device, dtype=dtype)
            self.target_dim = int(y_star.size(-1))
            return y_star

        raise ValueError(
            "Unsupported target shape for OSTTP. Expected [B], [B,C], [T,B,C], or [B,T,C]."
        )

    def _ensure_feedback_matrices(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.target_dim is None:
            raise RuntimeError("target_dim is undefined; cannot build OSTTP feedback matrices.")

        if self.output_size != int(self.target_dim):
            raise NotImplementedError(
                "OSTTP output layer size must match target dimension in this implementation. "
                f"Got output_size={self.output_size}, target_dim={self.target_dim}."
            )

        if self._feedback_ready:
            return

        std = 1.0 / math.sqrt(max(float(self.target_dim), 1.0))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.feedback_seed)

        self._feedback_names = []
        for idx, spec in enumerate(self.layer_specs[:-1]):
            fb = torch.empty(self.target_dim, spec.n_post, dtype=dtype)
            fb.normal_(mean=0.0, std=std, generator=generator)
            if self.feedback_scale != 1.0:
                fb.mul_(self.feedback_scale)

            name = f"_feedback_{idx}"
            self.register_buffer(name, fb.to(device=device), persistent=True)
            self._feedback_names.append(name)

        self._feedback_ready = True

    def _feedback_for_layer(self, layer_idx: int) -> torch.Tensor:
        return getattr(self, self._feedback_names[layer_idx])

    def _expand_neuron_param(
        self,
        value,
        batch_size: int,
        n_post: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device=device, dtype=dtype)
        else:
            tensor = torch.tensor(float(value), device=device, dtype=dtype)

        if tensor.numel() == 1:
            return tensor.view(1, 1).expand(batch_size, n_post)

        flat = tensor.flatten()
        if flat.numel() == n_post:
            return flat.view(1, n_post).expand(batch_size, n_post)

        raise ValueError(
            f"Neuron parameter with {tensor.numel()} elements cannot map to n_post={n_post}."
        )

    def _pseudo_derivative_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self.pseudo_derivative == "tanh":
            # psi(x) = 1 - tanh(x)^2
            return 1.0 - torch.tanh(x).pow(2)

        # psi(x) = 1 / (100*|x| + 1)^2
        return 1.0 / (100.0 * x.abs() + 1.0).pow(2)

    def _output_learning_signal(
        self,
        y_out: torch.Tensor,
        y_star: torch.Tensor,
    ) -> torch.Tensor:
        if self.output_loss == "ce":
            return torch.softmax(y_out, dim=1) - y_star
        if self.output_loss == "bce":
            return torch.sigmoid(y_out) - y_star
        return y_out - y_star

    def _output_loss_value(self, y_out: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
        if self.output_loss == "ce":
            return -(y_star * F.log_softmax(y_out, dim=1)).sum(dim=1).mean()
        if self.output_loss == "bce":
            return F.binary_cross_entropy_with_logits(y_out, y_star)
        return F.mse_loss(y_out, y_star)

    def _apply_jacobian_to_trace(
        self,
        trace: torch.Tensor,
        diag_term: torch.Tensor,
        recurrent_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute (ds_t/dy_{t-1}) @ trace.

        - Feedforward case: Jacobian is diagonal -> row-wise scaling.
        - Recurrent case: full matrix H + diag(diag_term).
        """
        if recurrent_weight is None:
            return diag_term.unsqueeze(-1) * trace

        batch_size = trace.size(0)
        j_y = recurrent_weight.detach().unsqueeze(0).expand(batch_size, -1, -1)
        j_y = j_y + torch.diag_embed(diag_term)
        return torch.bmm(j_y, trace)

    def _apply_jacobian_to_vector(
        self,
        vec: torch.Tensor,
        diag_term: torch.Tensor,
        recurrent_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if recurrent_weight is None:
            return diag_term * vec

        batch_size = vec.size(0)
        j_y = recurrent_weight.detach().unsqueeze(0).expand(batch_size, -1, -1)
        j_y = j_y + torch.diag_embed(diag_term)
        return torch.bmm(j_y, vec.unsqueeze(-1)).squeeze(-1)

    def _reduce_threshold_grad(
        self,
        grad_per_neuron: torch.Tensor,
        threshold_param: torch.nn.Parameter,
    ) -> torch.Tensor:
        param_numel = threshold_param.numel()
        if param_numel == 1:
            return grad_per_neuron.sum().view_as(threshold_param)
        if param_numel == grad_per_neuron.numel():
            return grad_per_neuron.view_as(threshold_param)
        raise ValueError(
            "Trainable threshold shape is incompatible with OSTTP per-neuron threshold updates."
        )

    def _accumulate_or_apply(self, param: torch.nn.Parameter, grad: torch.Tensor) -> None:
        if self.grad_clip > 0.0:
            grad = grad.clamp(-self.grad_clip, self.grad_clip)

        if self.use_optimizer and self.optimizer is not None:
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad += grad
        else:
            param.data -= self.lr * grad

    @torch.no_grad()
    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Perform one OSTTP update on a temporal batch.

        Args:
            data: [T, B, F]
            target: [B] or [B,C] or [T,B,C] or [B,T,C]

        Returns:
            loss: scalar (time-averaged output loss)
            pred: [B, 1] argmax on output spike counts
        """
        num_timesteps, batch_size, _ = data.shape
        device = data.device
        dtype = data.dtype

        y_star = self._prepare_temporal_targets(
            target=target,
            num_timesteps=num_timesteps,
            device=device,
            dtype=dtype,
        )
        self._ensure_feedback_matrices(device=device, dtype=dtype)

        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # Eligibility states per layer/parameter.
        layer_state: List[Dict[str, torch.Tensor]] = []
        prev_spk: List[torch.Tensor] = []
        prev_mem: List[torch.Tensor] = []

        for spec in self.layer_specs:
            state: Dict[str, torch.Tensor] = {
                "eps_w": torch.zeros(batch_size, spec.n_post, spec.n_in, device=device, dtype=dtype),
                "e_w": torch.zeros(batch_size, spec.n_post, spec.n_in, device=device, dtype=dtype),
                "delta_w": torch.zeros_like(spec.synapse.weight, device=device, dtype=dtype),
            }
            if spec.recurrent_weight is not None and spec.recurrent_weight.requires_grad:
                state["eps_h"] = torch.zeros(
                    batch_size, spec.n_post, spec.n_post, device=device, dtype=dtype
                )
                state["e_h"] = torch.zeros(
                    batch_size, spec.n_post, spec.n_post, device=device, dtype=dtype
                )
                state["delta_h"] = torch.zeros_like(
                    spec.recurrent_weight, device=device, dtype=dtype
                )
            if spec.threshold_param is not None:
                state["eps_b"] = torch.zeros(batch_size, spec.n_post, device=device, dtype=dtype)
                state["e_b"] = torch.zeros(batch_size, spec.n_post, device=device, dtype=dtype)
                state["delta_b"] = torch.zeros(spec.n_post, device=device, dtype=dtype)

            layer_state.append(state)
            prev_spk.append(torch.zeros(batch_size, spec.n_post, device=device, dtype=dtype))
            prev_mem.append(torch.zeros(batch_size, spec.n_post, device=device, dtype=dtype))

        spk_sum = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)
        total_loss = torch.zeros((), device=device, dtype=dtype)

        for t in range(num_timesteps):
            self._clear_step_cache()
            forward_out = self.network(data[t])
            x_rec, y_rec, s_rec = self._get_step_tensors(data[t], forward_out)

            spk_sum += y_rec[-1]

            y_star_t = y_star[t]
            l_out = self._output_learning_signal(y_rec[-1], y_star_t)
            total_loss += self._output_loss_value(y_rec[-1], y_star_t)

            for layer_idx, spec in enumerate(self.layer_specs):
                state = layer_state[layer_idx]

                x_t = x_rec[layer_idx]
                s_t = s_rec[layer_idx]
                y_prev = prev_spk[layer_idx]
                s_prev = prev_mem[layer_idx]

                beta_t = self._expand_neuron_param(
                    getattr(spec.neuron, "beta", 1.0),
                    batch_size=batch_size,
                    n_post=spec.n_post,
                    device=device,
                    dtype=dtype,
                )
                thresh_t = self._expand_neuron_param(
                    getattr(spec.neuron, "threshold", 1.0),
                    batch_size=batch_size,
                    n_post=spec.n_post,
                    device=device,
                    dtype=dtype,
                )

                psi_t = self._pseudo_derivative_fn(s_t - thresh_t)

                # Jacobians from OSTTP/OSTL equations:
                # J_s = ds_t/ds_{t-1}
                # J_y = ds_t/dy_{t-1} = H + diag(diag_term)
                if spec.reset_mechanism == "zero":
                    j_s = beta_t * (1.0 - y_prev)
                    diag_term = -(beta_t * s_prev)
                    ds_db = torch.zeros_like(y_prev)
                elif spec.reset_mechanism == "subtract":
                    j_s = beta_t
                    diag_term = -thresh_t
                    ds_db = -y_prev
                else:  # "none"
                    j_s = beta_t
                    diag_term = torch.zeros_like(beta_t)
                    ds_db = torch.zeros_like(y_prev)

                rec_weight = spec.recurrent_weight

                # Eq. (1): epsilon_t = J_s * epsilon_{t-1} + ds/dtheta + J_y @ e_{t-1}
                ds_d_w = x_t.unsqueeze(1).expand(-1, spec.n_post, -1)
                state["eps_w"] = (
                    j_s.unsqueeze(-1) * state["eps_w"]
                    + ds_d_w
                    + self._apply_jacobian_to_trace(state["e_w"], diag_term, rec_weight)
                )

                # Eq. (2): e_t = psi_t * epsilon_t (+ direct term when needed)
                state["e_w"] = psi_t.unsqueeze(-1) * state["eps_w"]

                if "eps_h" in state and "e_h" in state:
                    ds_d_h = y_prev.unsqueeze(1).expand(-1, spec.n_post, -1)
                    state["eps_h"] = (
                        j_s.unsqueeze(-1) * state["eps_h"]
                        + ds_d_h
                        + self._apply_jacobian_to_trace(state["e_h"], diag_term, rec_weight)
                    )
                    state["e_h"] = psi_t.unsqueeze(-1) * state["eps_h"]

                if "eps_b" in state and "e_b" in state:
                    state["eps_b"] = (
                        j_s * state["eps_b"]
                        + ds_db
                        + self._apply_jacobian_to_vector(state["e_b"], diag_term, rec_weight)
                    )
                    dy_db = -psi_t
                    state["e_b"] = psi_t * state["eps_b"] + dy_db

                if layer_idx == self.num_layers - 1:
                    l_t = l_out
                else:
                    l_t = torch.matmul(y_star_t, self._feedback_for_layer(layer_idx))

                # Delta theta accumulation over time and batch.
                state["delta_w"] += (l_t.unsqueeze(-1) * state["e_w"]).sum(dim=0)
                if "delta_h" in state:
                    state["delta_h"] += (l_t.unsqueeze(-1) * state["e_h"]).sum(dim=0)
                if "delta_b" in state:
                    state["delta_b"] += (l_t * state["e_b"]).sum(dim=0)

            for idx in range(self.num_layers):
                prev_spk[idx] = y_rec[idx]
                prev_mem[idx] = s_rec[idx]

        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        denom = float(max(batch_size, 1))
        for spec, state in zip(self.layer_specs, layer_state):
            grad_w = state["delta_w"] / denom
            self._accumulate_or_apply(spec.synapse.weight, grad_w)

            if "delta_h" in state and spec.recurrent_weight is not None:
                grad_h = state["delta_h"] / denom
                self._accumulate_or_apply(spec.recurrent_weight, grad_h)

            if "delta_b" in state and spec.threshold_param is not None:
                grad_b_neuron = state["delta_b"] / denom
                grad_b = self._reduce_threshold_grad(grad_b_neuron, spec.threshold_param)
                self._accumulate_or_apply(spec.threshold_param, grad_b)

        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        if self.debug:
            for idx, state in enumerate(layer_state):
                print(
                    f"[OSTTP] layer={idx} |dW|={state['delta_w'].norm().item():.4e}",
                    flush=True,
                )
                if "delta_h" in state:
                    print(
                        f"[OSTTP] layer={idx} |dH|={state['delta_h'].norm().item():.4e}",
                        flush=True,
                    )

        loss = total_loss / float(max(num_timesteps, 1))
        pred = spk_sum.argmax(dim=1, keepdim=True)
        return loss.detach(), pred

    def reset(self) -> None:
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def checkpoint_state(self) -> dict:
        if not self._feedback_ready:
            return {"target_dim": self.target_dim, "feedback": []}

        mats = [getattr(self, name).detach().cpu() for name in self._feedback_names]
        return {
            "target_dim": self.target_dim,
            "feedback": mats,
        }

    def load_checkpoint_state(self, state: dict) -> None:
        target_dim = state.get("target_dim", self.target_dim)
        if target_dim is not None:
            self.target_dim = int(target_dim)

        feedback = state.get("feedback", None)
        if feedback is None:
            return

        device = self.layer_specs[0].synapse.weight.device
        dtype = self.layer_specs[0].synapse.weight.dtype
        self._ensure_feedback_matrices(device=device, dtype=dtype)

        if len(feedback) != len(self._feedback_names):
            raise ValueError(
                "Checkpoint feedback matrix count does not match trainer hidden layers."
            )

        for name, matrix in zip(self._feedback_names, feedback):
            getattr(self, name).copy_(matrix.to(device=device, dtype=dtype))

    def to(self, device):
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self
