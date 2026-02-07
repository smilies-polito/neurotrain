"""
FELL (Full Event-based Local Learning) Trainer.

Per-layer local classifiers, MSE to one-hot. No detach; gradients flow through
time. Per-step, per-layer backward(retain_graph=True) and optimizer.step().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from networks.local_classifier_network import LocalClassifierNetwork


class FELLTrainer(BaseTrainer):
    """
    FELL trainer: per-step, per-layer backward with retain_graph.

    Block uses mode='fell' (no detach). Trainer computes local MSE,
    backward(retain_graph=True), optimizer.step() each timestep for each layer.
    """

    def __init__(
        self,
        network: LocalClassifierNetwork,
        lr: float,
        batch_size: int,
        use_raw_input: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.use_raw_input = use_raw_input

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

        # Paper-identical: same input every timestep (constant current into first layer)
        x_const = data.mean(dim=0)
        if not self.use_raw_input and x_const.shape[1] == 784:
            x_const = (x_const * 0.3081 + 0.1307).clamp(0.0, 1.0)

        spk_sum = torch.zeros(batch_size, n_classes, device=device)
        total_loss = 0.0

        for t in range(num_timesteps):
            layer_outputs = self.network.forward_step_all(x_const)

            for layer_idx, (spike_out, y_hat_spike) in enumerate(layer_outputs):
                loss_sup = F.mse_loss(y_hat_spike, target_onehot.detach())
                total_loss = total_loss + loss_sup.item()

                self.optimizers[layer_idx].zero_grad()
                loss_sup.backward(retain_graph=True)
                self.optimizers[layer_idx].step()

            spk_sum = spk_sum + layer_outputs[-1][1].detach()

        loss = torch.tensor(
            total_loss / (num_timesteps * len(self.network.blocks)), device=device
        )
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
