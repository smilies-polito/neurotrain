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
        lr: float,                              # Learning rate for synapse updates (ignored if use_optimizer=True and optimizer is provided)   
        batch_size: int,                        # Batch size for training (used for loss scaling; not a dataloader batch size)
        trace_decay: Optional[float] = None,    # Decay factor for synaptic traces
        online_updates: bool = True,
        use_optimizer: bool = False,
        optimizer=None,
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.trace_decay = (
            float(trace_decay) if trace_decay is not None else self._infer_trace_decay()
        )
        self.online_updates = bool(online_updates)
        self.use_optimizer = bool(use_optimizer)
        self._external_optimizer = optimizer
        self.optimizer = optimizer

        # OTTT traces are defined on synaptic inputs; we support linear/conv synapses.
        self.synapse_layers = [
            module
            for module in self.network.modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))
        ]
        if not self.synapse_layers:
            raise TypeError(
                "OTTTTrainer requires at least one nn.Linear or nn.Conv2d in the network."
            )

        self._trace_by_module: Dict[nn.Module, torch.Tensor] = {}

        if self.use_optimizer and self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def _infer_trace_decay(self) -> float:
        """Use neuron leak beta when available; fall back to 0.9."""
        for module in self.network.modules():
            beta = getattr(module, "beta", None)
            if beta is None:
                continue
            try:
                return float(beta)
            except (TypeError, ValueError):
                if isinstance(beta, torch.Tensor):
                    return float(beta.detach().item())
        return 0.9

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
        for layer in self.synapse_layers:
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
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            return
        for param in self.network.parameters():
            param.grad = None

    def _manual_step(self) -> None:
        with torch.no_grad():
            for module in self.synapse_layers:
                if module.weight.requires_grad and module.weight.grad is not None:
                    module.weight -= self.lr * module.weight.grad
                if (
                    module.bias is not None
                    and module.bias.requires_grad
                    and module.bias.grad is not None
                ):
                    module.bias -= self.lr * module.bias.grad

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

        hooks = self._register_trace_hooks()
        total_loss = torch.tensor(0.0, device=device)
        spk_sum = None

        try:
            with torch.enable_grad():
                # Loop on TIMESTEPS
                for t in range(num_timesteps):
                    self._detach_neuron_state()
                    if self.online_updates:
                        self._zero_all_grads()

                    # FORWARD PASS of the network at current timestep
                    spk_rec, mem_rec = self.network(data[t])
                    spk_out = spk_rec[-1]
                    logits = mem_rec[-1]

                    if spk_sum is None:
                        spk_sum = spk_out.detach()
                    else:
                        spk_sum = spk_sum + spk_out.detach()

                    # Paper Eq. (2): sequence loss is average of per-step losses.
                    loss_t = F.cross_entropy(logits, target) / num_timesteps
                    total_loss = total_loss + loss_t.detach()
                    loss_t.backward()

                    if self.online_updates:
                        if self.use_optimizer and self.optimizer is not None:
                            self.optimizer.step()
                        else:
                            self._manual_step()

                if not self.online_updates:
                    if self.use_optimizer and self.optimizer is not None:
                        self.optimizer.step()
                    else:
                        self._manual_step()
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
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self
