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
        network: nn.Module,                              # SNN to be trained with OTTT
        lr: float,                                       # Learning rate for optimizer updates
        batch_size: int,                                 # Batch size for training (used for loss scaling; not a dataloader batch size)
        online_updates: bool = True,
<<<<<<< Updated upstream
        constant_input_per_timestep: bool = False,       # True when the dataset feeds the same frame every timestep (direct/analog coding)
        loss_lambda: Optional[float] = None,             # CE/MSE interpolation used by official CIFAR OTTT recipe
        grad_clip: Optional[float] = None,               # Element-wise gradient clip before optimizer step
        sanitize_grads: Optional[bool] = None,           # Replace NaN/Inf grads with finite values before step
=======
<<<<<<< Updated upstream
        loss_lambda: Optional[float] = None,    # CE/MSE interpolation used by official CIFAR OTTT recipe
        grad_clip: Optional[float] = None,      # Element-wise gradient clip before optimizer step
        sanitize_grads: Optional[bool] = None,  # Replace NaN/Inf grads with finite values before step
=======
        loss_lambda: float = 0.0,                        # CE/MSE interpolation weight (0 = pure CE)
        grad_clip: float = 0.0,                          # Element-wise gradient clip before optimizer step (0 = disabled)
        sanitize_grads: bool = False,                    # Replace NaN/Inf grads with finite values before step
>>>>>>> Stashed changes
>>>>>>> Stashed changes
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        super().__init__()
        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        # trace_decay = beta, since beta = 1 - 1/tau by definition.
        # Falls back to 0.5 (tau=2.0) if the network doesn't expose beta.
        self.trace_decay = float(getattr(self.network, "beta", 0.5))
        self.online_updates = bool(online_updates)
<<<<<<< Updated upstream
        self.constant_input_per_timestep = bool(constant_input_per_timestep)
=======
<<<<<<< Updated upstream
>>>>>>> Stashed changes
        if loss_lambda is None:
            loss_lambda = 0.05 if self.constant_input_per_timestep else 0.0
        self.loss_lambda = float(loss_lambda)
        if grad_clip is None:
            grad_clip = 0.2 if self.constant_input_per_timestep else 0.0
        self.grad_clip = float(grad_clip)
        if self.grad_clip < 0.0:
            raise ValueError("OTTTTrainer requires grad_clip >= 0.")
        if sanitize_grads is None:
<<<<<<< Updated upstream
            sanitize_grads = self.constant_input_per_timestep
=======
            sanitize_grads = static_input_recipe
=======
        self.loss_lambda = float(loss_lambda)
        self.grad_clip = float(grad_clip)
        if self.grad_clip < 0.0:
            raise ValueError("OTTTTrainer requires grad_clip >= 0.")
>>>>>>> Stashed changes
>>>>>>> Stashed changes
        self.sanitize_grads = bool(sanitize_grads)
        self._external_optimizer = optimizer
        # Official OTTT training defaults to SGD with momentum when no optimizer is supplied.
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=0.9)
        )

        # OTTT traces are defined on synaptic inputs.
        # Any nn.Linear / nn.Conv2d (including subclasses that override forward, such as
        # ScaledWSConv2d) is accepted — forward hooks fire on __call__ for any subclass.
        # Note: modules that hold an nn.Linear/Conv2d as an *attribute* and bypass it via
        # F.linear/F.conv2d directly (e.g. a WSConv2d wrapper pattern) will have hooks
        # registered on the inner module that never fire — those synapses receive no trace
        # substitution silently. Use subclassing instead of wrapping for full compatibility.
        self.synapse_layers = []
        self.trace_synapse_layers = []
        for module in self.network.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.synapse_layers.append(module)
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
        """
        This function is the hearth of the OTTT learning rule in the trainer, here is where:
        - Trace computation happens
        - Perform a gradient substitution so that during backward we habe multiplication of the gradient with the trace instead of the spikes
        Per-synapse hook implementing OTTT trace substitution (Paper Eq. 4/5/7)
        """

        # Define the hook function
        def hook(_module: nn.Module, inputs, output):
            # Safe-guard
            if not inputs:
                return output
            pre = inputs[0]
            if not isinstance(pre, torch.Tensor):
                return output

            # Easy to understand calculation of the trace in the hook
            pre_detached = pre.detach()                                         # Important to detach the trace from the graph
            prev_trace = self._trace_by_module.get(module)                      # Get previous trace or create a new one
            if prev_trace is None or prev_trace.shape != pre_detached.shape:
                prev_trace = torch.zeros_like(pre_detached)
            trace = prev_trace * self.trace_decay + pre_detached                # Trace calculation
            self._trace_by_module[module] = trace                               # Trace storage

            # Build a proxy input whose *forward value* is the presynaptic trace,
            # while still allowing the gradient to be routed to the original presynaptic tensor (spatial credit within the timestep).
            pre_for_grad = _ReplaceForGrad.apply(pre, trace)
            
            # Shadow forward used for gradient computation: behaves like module(trace) for autograd.
            out_for_grad = self._module_forward_with_input(module, pre_for_grad)
            
            # Return the real forward output (detached so it carries no gradients),
            # but route gradients through out_for_grad so synapse weight grads use the trace.
            return _ReplaceForGrad.apply(out_for_grad, output.detach())

        # Return the hook function
        return hook

    def _register_trace_hooks(self):
        handles = []
        for layer in self.trace_synapse_layers:
            # HOOKS FOR FORWARD PASS: 
            handles.append(layer.register_forward_hook(self._make_trace_hook(layer)))
        return handles

    def _detach_neuron_state(self) -> None:
        """
        Block temporal graph links between timesteps (OTTT does not use BPTT).

        Covers two storage patterns:
        - vars(module): regular instance attributes (externally managed states, e.g. mem1).
        - module._buffers: snntorch registers lif.mem as a buffer, not a plain attribute,
          so vars() misses it.  We detach buffers separately.
        Parameters live in module._parameters and must not be detached.
        """
        for module in self.network.modules():
            for name, value in vars(module).items():
                if isinstance(value, torch.Tensor):
                    setattr(module, name, value.detach())
            # snntorch stores lif.mem (and reset) as registered buffers
            for name, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[name] = buf.detach()

    def _zero_all_grads(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _stabilize_grads(self) -> None:
        if not self.sanitize_grads and self.grad_clip <= 0.0:
            return
        limit = self.grad_clip if self.grad_clip > 0.0 else 0.0
        for param in self.network.parameters():
            grad = param.grad
            if grad is None:
                continue
            if self.sanitize_grads:
                grad.nan_to_num_(nan=0.0, posinf=limit, neginf=-limit)
            if self.grad_clip > 0.0:
                grad.clamp_(-self.grad_clip, self.grad_clip)

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train one sequence batch.

        Args:
            data: [T, B, ...]
            target: [B]
        """
        
        # INITIALIZATIONS
        num_timesteps = int(data.shape[0])
        device = data.device

        self.network.reset()
        self._trace_by_module = {}
        self._zero_all_grads()

        hooks = self._register_trace_hooks()
        total_loss = torch.tensor(0.0, device=device)
        spk_sum = None
<<<<<<< Updated upstream
        x_const = data.mean(dim=0) if self.constant_input_per_timestep else None
=======
<<<<<<< Updated upstream
        use_constant_input = bool(
            getattr(self.network, "constant_input_per_timestep", False)
        )
        x_const = data.mean(dim=0) if use_constant_input else None
=======
>>>>>>> Stashed changes
>>>>>>> Stashed changes

        try:
            with torch.enable_grad():
                # Loop on TIMESTEPS
                for t in range(num_timesteps):
                    self._detach_neuron_state()
                    if self.online_updates:
                        self._zero_all_grads()

                    # FORWARD PASS of the network at current timestep
                    x_t = data[t]
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
                        self._stabilize_grads()
                        self.optimizer.step()

                if not self.online_updates:
                    self._stabilize_grads()
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
