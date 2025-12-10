"""
OTTT (Online Training Through Time) trainer.

Implements a forward-in-time three-factor learning rule with eligibility
traces, without modifying the network's forward pass. Gradients are computed
per time-step using presynaptic traces and surrogate derivatives of the
postsynaptic membrane potential.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class OTTTTrainer(BaseTrainer):
    """
    Trainer implementing the Online Training Through Time rule with eligibility traces.

    Args:
        network: FCNetwork to train (forward pass left untouched)
        lr: Learning rate for manual updates
        batch_size: Training batch size
        trace_decay: Eligibility trace decay (lambda)
        surrogate_slope: Slope for sigmoid surrogate derivative
        online_updates: If True apply updates every timestep (online),
            otherwise accumulate over the sequence
        quant: Kept for interface compatibility (unused)
        use_optimizer: If True, populate .grad and call optimizer.step()
        optimizer: Optional optimizer instance
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        trace_decay: float = 0.9,
        surrogate_slope: float = 10.0,
        online_updates: bool = False,
        quant: bool = False,
        use_optimizer: bool = False,
        optimizer=None,
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.trace_decay = trace_decay
        self.surrogate_slope = surrogate_slope
        self.online_updates = online_updates
        self.quant = quant
        self.use_optimizer = use_optimizer
        self._external_optimizer = optimizer
        self.optimizer = optimizer
        self.threshold = 1.0

        # Linear layers for weight access (network forward left untouched)
        self.linear_layers: List[nn.Linear] = [
            layer
            for layer in getattr(self.network, "layers", [])
            if isinstance(layer, nn.Linear)
        ]
        self.num_layers = len(self.linear_layers)

        if self.use_optimizer and self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def surrogate_derivative(self, membrane: torch.Tensor) -> torch.Tensor:
        """Sigmoid surrogate derivative around the firing threshold."""
        x = (membrane - self.threshold) * self.surrogate_slope
        sig = torch.sigmoid(x)
        return self.surrogate_slope * sig * (1 - sig)

    def _apply_update(self, layer: nn.Linear, grad_w: torch.Tensor) -> None:
        """Apply or accumulate weight updates."""
        if self.use_optimizer and self.optimizer is not None:
            if layer.weight.grad is None:
                layer.weight.grad = grad_w.clone()
            else:
                layer.weight.grad += grad_w
        else:
            layer.weight.data -= self.lr * grad_w

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on a single batch using OTTT.

        Args:
            data: [num_timesteps, batch, in_features]
            target: [batch]

        Returns:
            loss: scalar tensor (no gradients attached)
            pred: [batch, 1] predictions from summed spikes
        """
        num_timesteps, batch_size, _ = data.shape
        device = data.device
        num_classes = self.network.n_classes

        # One-hot targets for instantaneous losses
        target_one_hot = F.one_hot(target, num_classes=num_classes).float()

        # Reset network state and traces
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        traces = [
            torch.zeros(batch_size, layer.in_features, device=device)
            for layer in self.linear_layers
        ]

        # Accumulate gradients across time if not updating online
        accum_grads = (
            [torch.zeros_like(layer.weight.data) for layer in self.linear_layers]
            if not self.online_updates
            else None
        )

        total_loss = torch.zeros(1, device=device)
        spk_sum = None

        with torch.no_grad():
            for t in range(num_timesteps):
                spks, mems = self.network(data[t])
                spk_sum = spks[-1] if spk_sum is None else spk_sum + spks[-1]

                # Update eligibility traces with current presynaptic spikes
                pre_acts = [data[t]] + spks[:-1]
                for l in range(self.num_layers):
                    traces[l].mul_(self.trace_decay).add_(pre_acts[l])

                # Instantaneous loss on output membrane (scaled by T)
                logits = mems[-1]
                loss_t = (
                    F.cross_entropy(logits, target, reduction="mean") / num_timesteps
                )
                total_loss += loss_t

                # Output error signal and membrane gradient
                probs = torch.softmax(logits, dim=1)
                delta = probs - target_one_hot

                g_u = [torch.zeros_like(m) for m in mems]
                g_u[-1] = delta * self.surrogate_derivative(mems[-1])

                # Backpropagate error across layers (same timestep)
                for l in reversed(range(self.num_layers - 1)):
                    delta_prev = torch.matmul(
                        g_u[l + 1], self.linear_layers[l + 1].weight
                    )
                    g_u[l] = delta_prev * self.surrogate_derivative(mems[l])

                # Weight updates via eligibility traces
                for l, layer in enumerate(self.linear_layers):
                    grad_w = (
                        torch.matmul(g_u[l].transpose(0, 1), traces[l]) / batch_size
                    )
                    if self.online_updates:
                        self._apply_update(layer, grad_w)
                    else:
                        accum_grads[l] += grad_w

        # Apply accumulated updates after sequence
        if not self.online_updates and accum_grads is not None:
            for layer, grad_w in zip(self.linear_layers, accum_grads):
                self._apply_update(layer, grad_w)

        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.step()

        # Predictions from spike counts
        pred = spk_sum.argmax(dim=1, keepdim=True)
        return total_loss.detach(), pred

    def reset(self):
        """Reset all LIF states and zero gradients if needed."""
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def to(self, device):
        """
        Move trainer and network to device, recreating optimizer if owned by this trainer.
        """
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self
