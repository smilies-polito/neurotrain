"""
ETLP (Event-based Three-factor Local Plasticity) trainer.

Implements the local learning rule described in:
Quintana et al., "ETLP: Event-based Three-factor Local Plasticity",
Neuromorphic Computing and Engineering, 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from networks.etlp_network import ETLPNetwork, SpikeFunction
from trainers.base_trainer import BaseTrainer


class ETLPTrainer(BaseTrainer):
    """ETLP trainer with online, local updates."""

    def __init__(
        self,
        network: ETLPNetwork,
        lr: float,
        batch_size: int,
        voltage_reg: float = 0.0,
        weight_l1: float = 0.0,
        weight_l2: float = 0.0,
        update_rate_hz: float = 100.0,
        quant: bool = False,
        use_optimizer: bool = False,
        optimizer: torch.optim.Optimizer | None = None,
        update_last: bool = False,
        update_every: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.voltage_reg = float(voltage_reg)
        self.weight_l1 = float(weight_l1)
        self.weight_l2 = float(weight_l2)
        self.update_rate_hz = float(update_rate_hz)
        self.quant = bool(quant)
        self.use_optimizer = bool(use_optimizer)
        self.optimizer = optimizer
        self.update_last = bool(update_last)
        self.update_every = int(update_every)

        # Probability of update trigger per timestep
        self.update_probability = max(
            min(self.update_rate_hz * float(self.network.dt) * 1e-3, 1.0), 0.0
        )

        self.loss_fn = nn.MSELoss()

    def _apply_update(
        self,
        output_spikes: torch.Tensor,
        targets_onehot: torch.Tensor,
        batch_size: int,
    ) -> None:
        with torch.no_grad():
            labels = targets_onehot
            error = output_spikes - labels

            v_rec = self.network.state.V_rec
            thr = self.network.thr
            voltage_mask = torch.logical_or(v_rec > thr, v_rec < -thr)
            learning_signals = torch.mm(-labels, self.network.b_out.T)
            learning_signals = (
                learning_signals + self.voltage_reg * v_rec * voltage_mask
            )

            v_scaled = (self.network.state.V_out - thr) / thr
            psi = SpikeFunction.pseudo_derivative(v_scaled)
            e_trace = psi[:, None, :] * self.network.state.epsilon_v_out[:, :, None]

            grad_in = torch.einsum(
                "bj,bij->ij", learning_signals, self.network.state.e_trace_in
            )
            grad_out = torch.einsum("bj,bij->ij", error, e_trace)

            grad_rec = None
            if (
                self.network.recurrent
                and self.network.W_rec is not None
                and self.network.state.e_trace_rec is not None
            ):
                grad_rec = torch.einsum(
                    "bj,bij->ij", learning_signals, self.network.state.e_trace_rec
                )

            if self.weight_l1 != 0.0:
                grad_in = grad_in + self.weight_l1 * torch.sign(self.network.W_in)
                if grad_rec is not None:
                    grad_rec = grad_rec + self.weight_l1 * torch.sign(
                        self.network.W_rec
                    )

            if self.weight_l2 != 0.0:
                grad_in = grad_in + self.weight_l2 * self.network.W_in
                grad_out = grad_out + self.weight_l2 * self.network.W_out
                if grad_rec is not None:
                    grad_rec = grad_rec + self.weight_l2 * self.network.W_rec

            if self.use_optimizer and self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
                self.network.W_in.grad = grad_in
                self.network.W_out.grad = grad_out
                if grad_rec is not None:
                    self.network.W_rec.grad = grad_rec
                self.optimizer.step()
            else:
                scale = self.lr / max(float(batch_size), 1.0)
                self.network.W_in.data -= scale * grad_in
                self.network.W_out.data -= scale * grad_out
                if grad_rec is not None:
                    self.network.W_rec.data -= scale * grad_rec

        self.network.detach()

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on a single batch using ETLP.

        Args:
            data: [timesteps, batch, in_features]
            target: [batch]

        Returns:
            loss: scalar tensor
            pred: [batch, 1]
        """
        num_timesteps = data.shape[0]
        batch_size = data.shape[1]
        device = data.device

        tgt_onehot = torch.zeros(batch_size, self.network.n_out, device=device)
        tgt_onehot.scatter_(1, target.view(-1, 1), 1.0)

        self.network.reset()

        spk_sum = None

        for t in range(num_timesteps):
            spk_rec, _ = self.network(data[t])
            output_spikes = spk_rec[-1]
            spk_sum = output_spikes if spk_sum is None else spk_sum + output_spikes

            if self.update_last and t < num_timesteps - 1:
                continue
            if not ((t + 1) % self.update_every == 0):
                continue

            if self.update_probability < 1.0:
                if torch.rand(1, device=device).item() > self.update_probability:
                    continue

            self._apply_update(output_spikes, tgt_onehot, batch_size)

        loss = self.loss_fn(spk_sum, tgt_onehot)
        pred = spk_sum.argmax(dim=1, keepdim=True)

        return loss, pred

    def reset(self):
        """Reset network state."""
        self.network.reset()
