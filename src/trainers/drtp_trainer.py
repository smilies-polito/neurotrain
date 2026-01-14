"""
DRTP (Direct Random Target Projection) trainer for FCNetwork.

Uses fixed random feedback matrices to project targets directly to hidden layers,
bypassing backpropagation through the network. Output layer is trained with a
local MSE loss on spike counts, while hidden layers receive target projections.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from trainers.base_trainer import BaseTrainer
from networks.fc_network import FCNetwork


class DRTPTrainer(BaseTrainer):
    """
    Direct Random Target Projection trainer for FCNetwork.

    Args:
        network: FCNetwork to train
        lr: Learning rate
        batch_size: Training batch size
        feedback_distribution: Distribution for random feedback matrices
        feedback_scale: Multiplicative scale for feedback matrices
        fixed_feedback: If True, keep fixed feedback matrices for the run
        quant: Quantization flag (unused; kept for interface compatibility)
        use_optimizer: If True, populate .grad and call optimizer.step()
        optimizer: Optional optimizer instance
        update_last: If True, update only at last timestep
        update_every: Update every N timesteps (default: 1)
    """

    _VALID_DISTS = ("kaiming_uniform", "uniform", "normal")

    def __init__(
        self,
        network: FCNetwork,
        lr: float,
        batch_size: int,
        feedback_distribution: str = "kaiming_uniform",
        feedback_scale: float = 1.0,
        fixed_feedback: bool = True,
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

        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.feedback_distribution = feedback_distribution
        self.feedback_scale = float(feedback_scale)
        self.fixed_feedback = bool(fixed_feedback)
        self.quant = quant
        self.use_optimizer = use_optimizer
        self.update_last = update_last
        self.update_every = update_every

        self.n_classes = int(getattr(network, "n_classes", 0))
        self.loss_value = 2.0 / max(self.n_classes, 1)
        self.loss_fn = nn.MSELoss()

        # Linear layers for weight access (FCNetwork: [Linear, LIF, ...])
        self.linear_layers: List[nn.Linear] = [
            layer for layer in getattr(self.network, "layers", []) if isinstance(layer, nn.Linear)
        ]
        self.num_layers = len(self.linear_layers)
        self.num_hidden = max(self.num_layers - 1, 0)
        self.hidden_sizes = [layer.out_features for layer in self.linear_layers[:-1]]

        if self.num_layers == 0:
            raise ValueError("DRTPTrainer requires a network with at least one Linear layer.")

        # Setup optimizer if requested
        self._external_optimizer = optimizer
        if self.use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(network.parameters(), lr=self.lr)
        else:
            self.optimizer = None

        # Fixed random feedback matrices (one per hidden layer)
        self.feedback = nn.ParameterList()
        if self.fixed_feedback:
            for hidden_size in self.hidden_sizes:
                fb = torch.empty(self.n_classes, hidden_size)
                self._init_feedback_(fb)
                self.feedback.append(nn.Parameter(fb, requires_grad=False))

    def _init_feedback_(self, tensor: torch.Tensor) -> torch.Tensor:
        """Initialize feedback weights in-place."""
        if self.feedback_distribution == "kaiming_uniform":
            nn.init.kaiming_uniform_(tensor)
        elif self.feedback_distribution == "uniform":
            tensor.uniform_(-1.0, 1.0)
        elif self.feedback_distribution == "normal":
            tensor.normal_(mean=0.0, std=1.0)
        if self.feedback_scale != 1.0:
            tensor.mul_(self.feedback_scale)
        return tensor

    def _sample_feedback(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        """Sample fresh feedback matrices for this batch."""
        mats = []
        for hidden_size in self.hidden_sizes:
            fb = torch.empty((self.n_classes, hidden_size), device=device, dtype=dtype)
            self._init_feedback_(fb)
            mats.append(fb)
        return mats

    def _accumulate_grad(self, layer: nn.Linear, grad_w: torch.Tensor) -> None:
        """Accumulate gradients into layer.weight.grad for optimizer usage."""
        if layer.weight.grad is None:
            layer.weight.grad = grad_w.clone()
        else:
            layer.weight.grad += grad_w

    def _apply_update(self, layer: nn.Linear, grad_w: torch.Tensor) -> None:
        """Apply manual or optimizer-backed update."""
        if self.use_optimizer and self.optimizer is not None:
            self._accumulate_grad(layer, grad_w)
        else:
            layer.weight.data -= grad_w

    @torch.no_grad()
    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on a single batch using DRTP.

        Args:
            data: [timesteps, batch, in_features]
            target: [batch]

        Returns:
            loss: scalar tensor
            pred: [batch, 1] predictions from summed spikes
        """
        num_timesteps, batch_size, _ = data.shape
        device = data.device

        tgt = torch.zeros(batch_size, self.n_classes, device=device)
        tgt.scatter_(1, target.unsqueeze(1), 1.0)

        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        feedback = self.feedback if self.fixed_feedback else self._sample_feedback(
            device=device, dtype=data.dtype
        )

        spk_sum = None

        for t in range(num_timesteps):
            spks, _ = self.network(data[t])
            spk_sum = spks[-1] if spk_sum is None else spk_sum + spks[-1]

            if self.update_last and t < num_timesteps - 1:
                continue
            if self.update_every > 1 and (t + 1) % self.update_every != 0:
                continue

            # Hidden layer updates with target projection
            for layer_idx in range(self.num_hidden):
                x_pre = data[t] if layer_idx == 0 else spks[layer_idx - 1]
                x_post = spks[layer_idx]
                proj = torch.matmul(tgt, feedback[layer_idx])
                if not self.use_optimizer:
                    proj = proj * (self.lr / batch_size)
                grad_w = torch.matmul((proj * x_post).transpose(0, 1), x_pre)
                self._apply_update(self.linear_layers[layer_idx], grad_w)

            # Output layer update (local MSE on spikes)
            error = spks[-1] - tgt
            x_pre_out = spks[-2] if self.num_hidden > 0 else data[t]
            if self.use_optimizer:
                loss_grad = error * self.loss_value
            else:
                loss_grad = error * self.loss_value * (self.lr / batch_size)
            grad_out = torch.matmul(loss_grad.transpose(0, 1), x_pre_out)
            self._apply_update(self.linear_layers[-1], grad_out)
            if self.use_optimizer and self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        loss = self.loss_fn(spk_sum, tgt)
        pred = spk_sum.argmax(dim=1, keepdim=True)
        return loss.detach(), pred

    def reset(self):
        """Reset network state and optimizer gradients."""
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
