"""
BELL (Backprop Event-based Local Learning) Trainer.

Per-layer local classifiers, MSE to one-hot. No detach. Accumulate loss over
all timesteps; single backward at end; one optimizer.step() per layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from networks.local_classifier_network import LocalClassifierNetwork


class BELLTrainer(BaseTrainer):
    """
    BELL trainer: accumulate loss over T; single backward; step per layer.

    Block uses mode='bell' (no detach). Trainer accumulates local MSE over
    all timesteps, backward once, optimizer.step() per layer at end.
    """

    def __init__(
        self,
        network: LocalClassifierNetwork,
        lr: float,
        batch_size: int,
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size

        self.optimizers = [
            torch.optim.Adam(block.parameters(), lr=lr, weight_decay=0.0)
            for block in network.blocks
        ]

    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Train on one batch. data: [T, B, F], target: [B]."""
        num_timesteps, batch_size, _ = data.shape
        device = data.device
        n_classes = self.network.n_classes

        target_onehot = torch.zeros(batch_size, n_classes, device=device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

        self.network.reset()

        spk_sum = torch.zeros(batch_size, n_classes, device=device)
        accumulated_loss = None

        for t in range(num_timesteps):
            layer_outputs = self.network.forward_step_all(data[t])

            for layer_idx, (spike_out, y_hat_spike) in enumerate(layer_outputs):
                loss_sup = F.mse_loss(y_hat_spike, target_onehot.detach())
                if accumulated_loss is None:
                    accumulated_loss = loss_sup
                else:
                    accumulated_loss = accumulated_loss + loss_sup

            spk_sum = spk_sum + layer_outputs[-1][1].detach()

        for layer_idx in range(len(self.network.blocks)):
            self.optimizers[layer_idx].zero_grad()
        accumulated_loss.backward(retain_graph=False)
        for layer_idx in range(len(self.network.blocks)):
            self.optimizers[layer_idx].step()

        loss = accumulated_loss / (num_timesteps * len(self.network.blocks))
        pred = spk_sum.argmax(dim=1)
        return loss, pred

    def reset(self) -> None:
        self.network.reset()

    def to(self, device):
        """Move trainer and network to device, recreating optimizers."""
        super().to(device)
        self.optimizers = [
            torch.optim.Adam(block.parameters(), lr=self.lr, weight_decay=0.0)
            for block in self.network.blocks
        ]
        return self
