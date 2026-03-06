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
    recurrent_bias: Optional[torch.nn.Parameter]
    threshold_param: Optional[torch.nn.Parameter]


class OSTTPTrainer(BaseTrainer):
    """
    Online Spatio-Temporal Learning with Target Projection (OSTTP).

    Supported patterns:
    - Spiking layers: nn.Linear -> snn.Leaky / snn.RLeaky
    - Output readout:
      - "spk": last spiking layer spikes
      - "mem": last spiking layer membrane
      - "logits": separate non-spiking nn.Linear readout
      - "probs": separate non-spiking nn.Linear (+ optional Sigmoid) readout

    Notes:
    - Convolutional synapses are currently not implemented because exact
      per-neuron eligibility tensors are not represented in this framework.
    - RLeaky with `reset_delay=False` is rejected to stay exact with equations
      based on y_{t-1}.
    """

    _VALID_PSEUDO = ("tanh", "fast_sigmoid")
    _VALID_OUTPUT_LOSS = ("ce", "mse", "bce", "bce_logits", "bce_probs")
    _VALID_OUTPUT_READOUT = ("spk", "mem", "logits", "probs")

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        pseudo_derivative: str = "tanh",
        output_loss: str = "ce",
        output_readout: str = "mem",
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

        output_readout = str(output_readout).lower()
        if output_readout not in self._VALID_OUTPUT_READOUT:
            raise ValueError(
                f"output_readout must be one of {self._VALID_OUTPUT_READOUT}, got {output_readout}"
            )

        # Backward-compatible alias.
        if output_loss == "bce":
            if output_readout == "probs":
                output_loss = "bce_probs"
            else:
                output_loss = "bce_logits"

        if output_loss == "bce_logits" and output_readout != "logits":
            raise ValueError(
                "output_loss='bce_logits' expects output_readout='logits'."
            )
        if output_loss == "bce_probs" and output_readout != "probs":
            raise ValueError(
                "output_loss='bce_probs' expects output_readout='probs'."
            )
        if output_loss == "ce" and output_readout == "probs":
            raise ValueError(
                "output_loss='ce' expects logits/membrane readout, not post-sigmoid probabilities."
            )

        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.pseudo_derivative = pseudo_derivative
        self.output_loss = output_loss
        self.output_readout = output_readout
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
        self._output_is_spiking = self.output_readout in ("spk", "mem")

        self._leaf_modules = self._list_leaf_modules()
        self.output_synapse: Optional[nn.Linear] = None
        self.output_activation: Optional[nn.Module] = None

        if self._output_is_spiking:
            self.output_size = self.layer_specs[-1].n_post
        else:
            self.output_synapse, self.output_activation = self._resolve_output_readout_modules()
            self.output_size = int(self.output_synapse.out_features)

        hidden_count = self.num_layers - 1 if self._output_is_spiking else self.num_layers
        if hidden_count < 0:
            hidden_count = 0
        self.hidden_layer_count = hidden_count

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
        self._hook_currents: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self._hook_spikes: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self._hook_mems: List[Optional[torch.Tensor]] = [None] * self.num_layers

        self._hook_out_input: Optional[torch.Tensor] = None
        self._hook_out_logits: Optional[torch.Tensor] = None
        self._hook_out_probs: Optional[torch.Tensor] = None

        self._register_layer_hooks()

    def _list_leaf_modules(self) -> List[Tuple[str, nn.Module]]:
        leaves: List[Tuple[str, nn.Module]] = []
        for name, module in self.network.named_modules():
            if not name:
                continue
            if any(True for _ in module.children()):
                continue
            leaves.append((name, module))
        return leaves

    def _resolve_osttp_layers(self) -> List[_LayerSpec]:
        """
        Discover Linear->spiking pairs in container definition order.

        Pairing rule: adjacent modules within the same container.
        This supports ModuleList/Sequential and common patterns like fc1/lif1,
        fc2/lif2 without modifying the network class.
        """

        pairs: List[Tuple[nn.Module, nn.Module]] = []
        seen_pairs = set()

        def _try_add_pair(syn: nn.Module, neuron: nn.Module) -> None:
            if not self._is_supported_synapse(syn):
                return
            if not self._is_supported_neuron(neuron):
                return
            key = (id(syn), id(neuron))
            if key in seen_pairs:
                return
            seen_pairs.add(key)
            pairs.append((syn, neuron))

        # Explicit `network.layers` fallback for custom frameworks where layers
        # may live in a plain list/tuple.
        if hasattr(self.network, "layers"):
            modules = getattr(self.network, "layers")
            if isinstance(modules, (list, tuple, nn.ModuleList, nn.Sequential)):
                seq = list(modules)
                for i in range(len(seq) - 1):
                    _try_add_pair(seq[i], seq[i + 1])

        # Generic scan over all containers (same-container adjacency).
        for _container_name, container in self.network.named_modules():
            children = list(container.children())
            for i in range(len(children) - 1):
                _try_add_pair(children[i], children[i + 1])

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
                    "OSTTP equations here update W/H/threshold only for spiking layers; "
                    "trainable synaptic bias on spiking layers is not implemented."
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
            rec_bias = None
            recurrent = getattr(neuron, "recurrent", None)
            if recurrent is not None:
                if not hasattr(recurrent, "weight"):
                    raise NotImplementedError(
                        "RLeaky one-to-one recurrent mode is not supported; "
                        "use all_to_all=True so recurrent.weight exists."
                    )
                rec_weight = recurrent.weight
                rec_bias = recurrent.bias if hasattr(recurrent, "bias") else None
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
                    recurrent_bias=rec_bias,
                    threshold_param=threshold_param,
                )
            )

        return specs

    def _resolve_output_readout_modules(self) -> Tuple[nn.Linear, Optional[nn.Module]]:
        used_synapse_ids = {id(spec.synapse) for spec in self.layer_specs}
        candidates: List[Tuple[int, nn.Linear]] = []

        for idx, (name, module) in enumerate(self._leaf_modules):
            if not isinstance(module, nn.Linear):
                continue
            if id(module) in used_synapse_ids:
                continue
            if ".recurrent" in name:
                continue
            candidates.append((idx, module))

        if not candidates:
            raise ValueError(
                "output_readout='logits'/'probs' requires a non-spiking nn.Linear "
                "readout after the spiking stack."
            )

        out_idx, out_linear = candidates[-1]

        out_activation: Optional[nn.Module] = None
        if self.output_readout == "probs":
            if out_idx + 1 < len(self._leaf_modules):
                next_mod = self._leaf_modules[out_idx + 1][1]
                if isinstance(next_mod, nn.Sigmoid):
                    out_activation = next_mod

        return out_linear, out_activation

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

        if self.output_synapse is not None:
            self._hook_handles.append(
                self.output_synapse.register_forward_hook(self._make_output_synapse_hook())
            )
            if self.output_readout == "probs" and self.output_activation is not None:
                self._hook_handles.append(
                    self.output_activation.register_forward_hook(self._make_output_probs_hook())
                )

    def _make_synapse_hook(self, idx: int):
        def _hook(_module, inputs, output):
            if inputs and torch.is_tensor(inputs[0]):
                self._hook_inputs[idx] = inputs[0].detach()
            if torch.is_tensor(output):
                self._hook_currents[idx] = output.detach()

        return _hook

    def _make_neuron_hook(self, idx: int):
        def _hook(_module, _inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                return
            spk_t, mem_t = output[0], output[1]
            self._hook_spikes[idx] = spk_t.detach()
            self._hook_mems[idx] = mem_t.detach()

        return _hook

    def _make_output_synapse_hook(self):
        def _hook(_module, inputs, output):
            if inputs and torch.is_tensor(inputs[0]):
                self._hook_out_input = inputs[0].detach()
            if torch.is_tensor(output):
                self._hook_out_logits = output.detach()

        return _hook

    def _make_output_probs_hook(self):
        def _hook(_module, _inputs, output):
            if torch.is_tensor(output):
                self._hook_out_probs = output.detach()

        return _hook

    def _clear_step_cache(self) -> None:
        for idx in range(self.num_layers):
            self._hook_inputs[idx] = None
            self._hook_currents[idx] = None
            self._hook_spikes[idx] = None
            self._hook_mems[idx] = None

        self._hook_out_input = None
        self._hook_out_logits = None
        self._hook_out_probs = None

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
                "OSTTP output size must match target dimension in this implementation. "
                f"Got output_size={self.output_size}, target_dim={self.target_dim}."
            )

        if self._feedback_ready:
            return

        std = 1.0 / math.sqrt(max(float(self.target_dim), 1.0))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.feedback_seed)

        self._feedback_names = []
        for idx, spec in enumerate(self.layer_specs[: self.hidden_layer_count]):
            fb = torch.empty(self.target_dim, spec.n_post, dtype=dtype)
            fb.normal_(mean=0.0, std=std, generator=generator)
            if self.feedback_scale != 1.0:
                fb.mul_(self.feedback_scale)

            name = f"_feedback_{idx}"
            self.register_buffer(name, fb.to(device=device), persistent=True)
            self._feedback_names.append(name)

        self._feedback_ready = True

    def _feedback_for_hidden(self, hidden_idx: int) -> torch.Tensor:
        return getattr(self, self._feedback_names[hidden_idx])

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
        if self.output_loss == "bce_logits":
            return torch.sigmoid(y_out) - y_star
        if self.output_loss == "bce_probs":
            probs = y_out.clamp(1e-6, 1.0 - 1e-6)
            return (probs - y_star) / (probs * (1.0 - probs))
        return y_out - y_star

    def _output_loss_value(self, y_out: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
        if self.output_loss == "ce":
            return -(y_star * F.log_softmax(y_out, dim=1)).sum(dim=1).mean()
        if self.output_loss == "bce_logits":
            return F.binary_cross_entropy_with_logits(y_out, y_star)
        if self.output_loss == "bce_probs":
            probs = y_out.clamp(1e-6, 1.0 - 1e-6)
            return F.binary_cross_entropy(probs, y_star)
        return F.mse_loss(y_out, y_star)

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
            data: [T, B, ...]
            target: [B] or [B,C] or [T,B,C] or [B,T,C]

        Returns:
            loss: scalar (time-averaged output loss)
            pred: [B, 1] argmax over time-summed readout

        Paper-to-code mapping used below:
        - s_t^l: membrane/state tensor (s_rec)
        - y_t^l: layer output tensor (spike or analog readout)
        - b: threshold
        - H: recurrent weights
        - beta: decay
        - L_t^K = dE_t/dy_t^K (output learning signal)
        - L_t^l = B_l y*_t for hidden layers (Eq. 11)
        - Delta theta_l ~ sum_t L_t^l e_t^{l,theta} (Eq. 6)
        - epsilon/e recursions follow Eq. (8)-(9):
          epsilon_t = (ds_t/ds_{t-1}) epsilon_{t-1} + ds_t/dtheta +
                      (ds_t/dy_{t-1}) (dy_{t-1}/dtheta)
          e_t = (dy_t/ds_t) epsilon_t + dy_t/dtheta
        """

        if not hasattr(self.network, "reset"):
            raise ValueError("OSTTPTrainer requires network.reset() to clear temporal state.")

        if data.dim() < 3:
            raise ValueError(
                f"OSTTPTrainer expects input shape [T, B, ...], got {tuple(data.shape)}."
            )

        num_timesteps = int(data.shape[0])
        batch_size = int(data.shape[1])
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

        # Eligibility states per spiking layer and parameter.
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

        output_state: Optional[Dict[str, torch.Tensor]] = None
        if self.output_synapse is not None:
            output_state = {
                "delta_w": torch.zeros_like(self.output_synapse.weight, device=device, dtype=dtype)
            }
            if self.output_synapse.bias is not None and self.output_synapse.bias.requires_grad:
                output_state["delta_b"] = torch.zeros_like(
                    self.output_synapse.bias, device=device, dtype=dtype
                )

        readout_sum = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)
        total_loss = torch.zeros((), device=device, dtype=dtype)

        for t in range(num_timesteps):
            self._clear_step_cache()
            forward_out = self.network(data[t])

            # (a) Forward step capture: x_t^l, y_t^l, s_t^l for spiking layers.
            hooks_complete = (
                all(val is not None for val in self._hook_inputs)
                and all(val is not None for val in self._hook_spikes)
                and all(val is not None for val in self._hook_mems)
            )

            if hooks_complete:
                x_rec = [val for val in self._hook_inputs if val is not None]
                y_rec = [val for val in self._hook_spikes if val is not None]
                s_rec = [val for val in self._hook_mems if val is not None]
                i_rec = [val for val in self._hook_currents if val is not None]
                if len(i_rec) != self.num_layers:
                    i_rec = [
                        F.linear(x_t, spec.synapse.weight.detach(), spec.synapse.bias)
                        for x_t, spec in zip(x_rec, self.layer_specs)
                    ]
            elif (
                isinstance(forward_out, (tuple, list))
                and len(forward_out) >= 2
                and isinstance(forward_out[0], (tuple, list))
                and isinstance(forward_out[1], (tuple, list))
            ):
                y_rec = [tens.detach() for tens in forward_out[0]]
                s_rec = [tens.detach() for tens in forward_out[1]]
                if len(y_rec) != self.num_layers or len(s_rec) != self.num_layers:
                    raise RuntimeError(
                        "OSTTPTrainer fallback capture expected one spike/mem pair per discovered layer."
                    )
                x_rec = [data[t].detach()]
                x_rec.extend(y_rec[:-1])
                i_rec = [
                    F.linear(x_t, spec.synapse.weight.detach(), spec.synapse.bias)
                    for x_t, spec in zip(x_rec, self.layer_specs)
                ]
            else:
                raise RuntimeError(
                    "OSTTPTrainer could not capture per-layer (input, spike, membrane) tensors. "
                    "Ensure the network uses Linear->Leaky/RLeaky layers reachable by hooks."
                )

            # (b) Output learning signal L_t^K and hidden projections L_t^l (Eq. 11).
            if self.output_readout == "spk":
                y_out_t = y_rec[-1]
            elif self.output_readout == "mem":
                y_out_t = s_rec[-1]
            elif self.output_readout == "logits":
                if self._hook_out_logits is not None:
                    y_out_t = self._hook_out_logits
                elif torch.is_tensor(forward_out):
                    y_out_t = forward_out.detach()
                else:
                    raise RuntimeError(
                        "output_readout='logits' requires a captured linear readout tensor."
                    )
            else:  # probs
                if self._hook_out_probs is not None:
                    y_out_t = self._hook_out_probs
                elif torch.is_tensor(forward_out):
                    y_out_t = forward_out.detach()
                elif self._hook_out_logits is not None:
                    y_out_t = torch.sigmoid(self._hook_out_logits)
                else:
                    raise RuntimeError(
                        "output_readout='probs' requires captured probabilities "
                        "or a tensor output from network.forward."
                    )

            y_star_t = y_star[t]
            l_out = self._output_learning_signal(y_out_t, y_star_t)
            total_loss += self._output_loss_value(y_out_t, y_star_t)
            readout_sum += y_out_t

            hidden_fb_idx = 0

            for layer_idx, spec in enumerate(self.layer_specs):
                state = layer_state[layer_idx]

                x_t = x_rec[layer_idx]
                s_t = s_rec[layer_idx]
                y_prev = prev_spk[layer_idx]
                s_prev = prev_mem[layer_idx]
                cur_t = i_rec[layer_idx]

                # Expand beta/threshold into [B, n_post] tensors.
                beta_raw = getattr(spec.neuron, "beta", 1.0)
                if isinstance(beta_raw, torch.Tensor):
                    beta_t = beta_raw.detach().to(device=device, dtype=dtype)
                else:
                    beta_t = torch.tensor(float(beta_raw), device=device, dtype=dtype)
                if beta_t.numel() == 1:
                    beta_t = beta_t.view(1, 1).expand(batch_size, spec.n_post)
                elif beta_t.numel() == spec.n_post:
                    beta_t = beta_t.flatten().view(1, spec.n_post).expand(batch_size, spec.n_post)
                else:
                    raise ValueError(
                        f"beta with {beta_t.numel()} values cannot map to n_post={spec.n_post}."
                    )

                thresh_raw = getattr(spec.neuron, "threshold", 1.0)
                if isinstance(thresh_raw, torch.Tensor):
                    thresh_t = thresh_raw.detach().to(device=device, dtype=dtype)
                else:
                    thresh_t = torch.tensor(float(thresh_raw), device=device, dtype=dtype)
                if thresh_t.numel() == 1:
                    thresh_t = thresh_t.view(1, 1).expand(batch_size, spec.n_post)
                elif thresh_t.numel() == spec.n_post:
                    thresh_t = thresh_t.flatten().view(1, spec.n_post).expand(batch_size, spec.n_post)
                else:
                    raise ValueError(
                        f"threshold with {thresh_t.numel()} values cannot map to n_post={spec.n_post}."
                    )

                psi_t = self._pseudo_derivative_fn(s_t - thresh_t)

                # (c) Jacobians ds_t/ds_{t-1}, ds_t/dy_{t-1} for snnTorch reset dynamics.
                # snn.Leaky(reset='zero', reset_delay=True):
                #   s_t = beta*(1 - y_{t-1})*s_{t-1} + I_t
                # snn.RLeaky(reset='zero', reset_delay=True):
                #   s_t = (1 - y_{t-1})*(beta*s_{t-1} + I_t + H*y_{t-1})
                rec_weight = spec.recurrent_weight.detach() if spec.recurrent_weight is not None else None
                rec_bias = spec.recurrent_bias.detach() if spec.recurrent_bias is not None else None
                rec_row_scale = None

                if spec.reset_mechanism == "subtract":
                    j_s = beta_t
                    diag_term = -thresh_t
                    ds_db = -y_prev
                elif spec.reset_mechanism == "none":
                    j_s = beta_t
                    diag_term = torch.zeros_like(beta_t)
                    ds_db = torch.zeros_like(y_prev)
                else:  # "zero"
                    j_s = beta_t * (1.0 - y_prev)
                    ds_db = torch.zeros_like(y_prev)

                    if isinstance(spec.neuron, snn.RLeaky):
                        rec_term = torch.zeros_like(s_t)
                        if rec_weight is not None:
                            rec_term = F.linear(y_prev, rec_weight, rec_bias)
                        base_pre_reset = beta_t * s_prev + cur_t + rec_term
                        diag_term = -base_pre_reset
                        rec_row_scale = 1.0 - y_prev
                    else:
                        if rec_weight is not None:
                            raise NotImplementedError(
                                "Unexpected recurrent weights on Leaky layer for reset='zero'."
                            )
                        diag_term = -(beta_t * s_prev)

                if rec_weight is None:
                    jy_e_w = diag_term.unsqueeze(-1) * state["e_w"]
                    jy_e_h = None
                    jy_e_b = diag_term * state["e_b"] if "e_b" in state else None
                else:
                    rec_mat = rec_weight.unsqueeze(0).expand(batch_size, -1, -1)
                    if rec_row_scale is not None:
                        rec_mat = rec_row_scale.unsqueeze(-1) * rec_mat
                    j_y = rec_mat + torch.diag_embed(diag_term)
                    jy_e_w = torch.bmm(j_y, state["e_w"])
                    jy_e_h = torch.bmm(j_y, state["e_h"]) if "e_h" in state else None
                    jy_e_b = (
                        torch.bmm(j_y, state["e_b"].unsqueeze(-1)).squeeze(-1)
                        if "e_b" in state
                        else None
                    )

                # (d) Eligibility recursions epsilon/e for W/H/b (Eq. 8-9).
                ds_d_w = x_t.unsqueeze(1).expand(-1, spec.n_post, -1)
                state["eps_w"] = j_s.unsqueeze(-1) * state["eps_w"] + ds_d_w + jy_e_w

                is_spiking_output_layer = self._output_is_spiking and layer_idx == (self.num_layers - 1)
                if is_spiking_output_layer and self.output_readout == "mem":
                    dy_ds = torch.ones_like(psi_t)
                    dy_db = torch.zeros_like(psi_t)
                else:
                    dy_ds = psi_t
                    dy_db = -psi_t

                state["e_w"] = dy_ds.unsqueeze(-1) * state["eps_w"]

                if "eps_h" in state and "e_h" in state:
                    ds_d_h = y_prev.unsqueeze(1).expand(-1, spec.n_post, -1)
                    if jy_e_h is None:
                        raise RuntimeError("Recurrent eligibility requested without recurrent Jacobian.")
                    state["eps_h"] = j_s.unsqueeze(-1) * state["eps_h"] + ds_d_h + jy_e_h
                    state["e_h"] = dy_ds.unsqueeze(-1) * state["eps_h"]

                if "eps_b" in state and "e_b" in state:
                    if jy_e_b is None:
                        jy_e_b = diag_term * state["e_b"]
                    state["eps_b"] = j_s * state["eps_b"] + ds_db + jy_e_b
                    state["e_b"] = dy_ds * state["eps_b"] + dy_db

                # Hidden layers use fixed projection; output layer uses dE/dy_t^K.
                if is_spiking_output_layer:
                    l_t = l_out
                else:
                    l_t = torch.matmul(y_star_t, self._feedback_for_hidden(hidden_fb_idx))
                    hidden_fb_idx += 1

                # (e) Time accumulation of Delta theta_l from Eq. (6).
                state["delta_w"] += (l_t.unsqueeze(-1) * state["e_w"]).sum(dim=0)
                if "delta_h" in state:
                    state["delta_h"] += (l_t.unsqueeze(-1) * state["e_h"]).sum(dim=0)
                if "delta_b" in state:
                    state["delta_b"] += (l_t * state["e_b"]).sum(dim=0)

            # Non-spiking output readout branch (logits/probs): update output linear.
            if output_state is not None and self.output_synapse is not None:
                if self._hook_out_input is None:
                    raise RuntimeError(
                        "Could not capture output readout input. Ensure output linear module "
                        "is used in network.forward."
                    )
                out_in_t = self._hook_out_input

                if self.output_readout == "logits":
                    dy_ds_out = torch.ones_like(l_out)
                else:  # probs
                    dy_ds_out = y_out_t * (1.0 - y_out_t)

                e_out_w = dy_ds_out.unsqueeze(-1) * out_in_t.unsqueeze(1)
                output_state["delta_w"] += (l_out.unsqueeze(-1) * e_out_w).sum(dim=0)

                if "delta_b" in output_state and self.output_synapse.bias is not None:
                    e_out_b = dy_ds_out
                    output_state["delta_b"] += (l_out * e_out_b).sum(dim=0)

            for idx in range(self.num_layers):
                prev_spk[idx] = y_rec[idx]
                prev_mem[idx] = s_rec[idx]

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

        if output_state is not None and self.output_synapse is not None:
            grad_out_w = output_state["delta_w"] / denom
            self._accumulate_or_apply(self.output_synapse.weight, grad_out_w)

            if "delta_b" in output_state and self.output_synapse.bias is not None:
                grad_out_b = output_state["delta_b"] / denom
                self._accumulate_or_apply(self.output_synapse.bias, grad_out_b)

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
            if output_state is not None:
                print(
                    f"[OSTTP] output_readout |dW|={output_state['delta_w'].norm().item():.4e}",
                    flush=True,
                )

        loss = total_loss / float(max(num_timesteps, 1))
        pred = readout_sum.argmax(dim=1, keepdim=True)
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
