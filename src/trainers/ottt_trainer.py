"""
OTTT (Online Training Through Time) trainer.
Key idea used here:
- spatial credit: exact per-timestep backprop (current-time computational graph)
- temporal credit: presynaptic trace substitution in synapse gradients
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class _ReplaceForGrad(torch.autograd.Function):
    """
    Forward uses the second argument; backward routes gradients to both.

    This mirrors the OTTT reference trick:
    - keep forward values unchanged
    - make gradients behave as if traces were used
    """

    @staticmethod
    def forward(ctx, x_for_backward: torch.Tensor, x_for_forward: torch.Tensor):
        return x_for_forward

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, grad_output


class OTTTTrainer(BaseTrainer):
    """Online Training Through Time trainer with trace-based synapse gradients."""

    def __init__(
        self,
        network: nn.Module,                     # SNN to be trained with OTTT
        lr: float,                              # Learning rate for optimizer updates
        batch_size: int,                        # Batch size for training (used for loss scaling; not a dataloader batch size)
        trace_decay: Optional[float] = None,    # Decay factor for synaptic traces (default: 0.5)
        online_updates: bool = False,
        loss_lambda: Optional[float] = None,    # CE/MSE interpolation used by official CIFAR OTTT recipe
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        super().__init__()
        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self._use_internal_ottt_grad = bool(
            getattr(self.network, "uses_internal_ottt_grad", False)
        )
        # Match OTTT default: decay = 1 - 1/tau for LIF traces.
        if trace_decay is None:
            tau = float(getattr(self.network, "tau", 2.0))
            trace_decay = 1.0 - 1.0 / tau
        self.trace_decay = float(trace_decay)
        self.online_updates = bool(online_updates)
        if loss_lambda is None:
            loss_lambda = (
                0.05 if getattr(self.network, "constant_input_per_timestep", False) else 0.0
            )
        self.loss_lambda = float(loss_lambda)
        self._external_optimizer = optimizer
        # Official OTTT training defaults to SGD with momentum when no optimizer is supplied.
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=0.9)
        )

        # OTTT traces are defined on synaptic inputs.
        # For networks that implement official grad-with-rate internally, hooks are not needed.
        self.synapse_layers = []
        self.trace_synapse_layers = []
        if not self._use_internal_ottt_grad:
            unsupported_synapse_types = set()
            allowed_parametrized_types = {"ParametrizedLinear", "ParametrizedConv2d"}
            for module in self.network.modules():
                if not isinstance(module, (nn.Linear, nn.Conv2d)):
                    continue
                if type(module) in (nn.Linear, nn.Conv2d) or type(module).__name__ in allowed_parametrized_types:
                    self.synapse_layers.append(module)
                else:
                    unsupported_synapse_types.add(type(module).__name__)
            if unsupported_synapse_types:
                raise TypeError(
                    "OTTTTrainer supports nn.Linear/nn.Conv2d and torch parametrized variants; "
                    f"found unsupported subclasses: {sorted(unsupported_synapse_types)}"
                )
            if not self.synapse_layers:
                raise TypeError(
                    "OTTTTrainer requires at least one nn.Linear or nn.Conv2d in the network."
                )
            # Match official OTTT grad-with-rate behavior by excluding the first synapse.
            self.trace_synapse_layers = self.synapse_layers[1:]

        self._trace_by_module: Dict[nn.Module, torch.Tensor] = {}

    @staticmethod
    def _module_forward_with_input(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Functional forward for supported synapse modules."""
        if isinstance(module, nn.Linear):
            return F.linear(x, module.weight, module.bias)
        if isinstance(module, nn.Conv2d):
            return F.conv2d(
                x,
                module.weight,
                module.bias,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
        raise TypeError(f"Unsupported synapse module type: {type(module).__name__}")

    def _make_trace_hook(self, module: nn.Module):
        """Per-synapse hook implementing OTTT trace substitution (Paper Eq. 4/5/7)."""

        def hook(_module: nn.Module, inputs, output):
            if not inputs:
                return output
            pre = inputs[0]
            if not isinstance(pre, torch.Tensor):
                return output

            pre_detached = pre.detach()
            prev_trace = self._trace_by_module.get(module)
            if prev_trace is None or prev_trace.shape != pre_detached.shape:
                prev_trace = torch.zeros_like(pre_detached)
            trace = prev_trace * self.trace_decay + pre_detached
            self._trace_by_module[module] = trace

            # Forward uses real presynaptic activity.
            # Backward acts as if presynaptic trace was used.
            pre_for_grad = _ReplaceForGrad.apply(pre, trace)
            out_for_grad = self._module_forward_with_input(module, pre_for_grad)
            return _ReplaceForGrad.apply(out_for_grad, output.detach())

        return hook

    def _register_trace_hooks(self):
        handles = []
        for layer in self.trace_synapse_layers:
            handles.append(layer.register_forward_hook(self._make_trace_hook(layer)))
        return handles

    def _detach_neuron_state(self) -> None:
        """
        Block temporal graph links between timesteps (OTTT does not use BPTT).
        """
        for module in self.network.modules():
            for attr in ("mem", "spk", "syn"):
                value = getattr(module, attr, None)
                if isinstance(value, torch.Tensor):
                    setattr(module, attr, value.detach())

    def _zero_all_grads(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train one sequence batch.

        Args:
            data: [T, B, ...]
            target: [B]
        """
        num_timesteps = int(data.shape[0])
        device = data.device

        self.network.reset()
        self._trace_by_module = {}
        self._zero_all_grads()

        hooks = [] if self._use_internal_ottt_grad else self._register_trace_hooks()
        total_loss = torch.tensor(0.0, device=device)
        spk_sum = None
        use_constant_input = bool(
            getattr(self.network, "constant_input_per_timestep", False)
        )
        x_const = data.mean(dim=0) if use_constant_input else None

        try:
            with torch.enable_grad():
                # Loop on TIMESTEPS
                for t in range(num_timesteps):
                    if not self._use_internal_ottt_grad:
                        self._detach_neuron_state()
                    if self.online_updates:
                        self._zero_all_grads()

                    # FORWARD PASS of the network at current timestep
                    x_t = x_const if x_const is not None else data[t]
                    spk_rec, mem_rec = self.network(x_t)
                    spk_out = spk_rec[-1]
                    # Static-image OTTT path uses CE on last-layer membrane/logits readout.
                    logits = mem_rec[-1]

                    if spk_sum is None:
                        spk_sum = spk_out.detach()
                    else:
                        spk_sum = spk_sum + spk_out.detach()

                    # Keep paper/repo semantics: each timestep contributes 1/T of sequence loss.
                    if self.loss_lambda > 0.0:
                        n_classes = int(getattr(self.network, "n_classes", logits.shape[1]))
                        target_one_hot = F.one_hot(target, n_classes).float()
                        loss_t = (
                            (1.0 - self.loss_lambda) * F.cross_entropy(logits, target)
                            + self.loss_lambda * F.mse_loss(logits, target_one_hot)
                        ) / num_timesteps
                    else:
                        loss_t = F.cross_entropy(logits, target) / num_timesteps
                    total_loss = total_loss + loss_t.detach()
                    loss_t.backward()

                    if self.online_updates:
                        self.optimizer.step()

                if not self.online_updates:
                    self.optimizer.step()
        finally:
            for handle in hooks:
                handle.remove()
            self._zero_all_grads()

        pred = spk_sum.argmax(dim=1, keepdim=True)
        return total_loss, pred

    def reset(self):
        self.network.reset()
        self._zero_all_grads()

    def to(self, device):
        super().to(device)
        if self._external_optimizer is None:
            self.optimizer = torch.optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=0.9
            )
        return self
