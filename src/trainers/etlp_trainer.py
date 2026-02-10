"""
ETLP (Event-based Three-factor Local Plasticity) trainer.

Implements the local learning rule described in:
Quintana et al., "ETLP: Event-based Three-factor Local Plasticity",
Neuromorphic Computing and Engineering, 2024.
"""

from __future__ import annotations

from typing import List

import snntorch as snn
import torch
import torch.nn as nn

from networks.fc_network import FCNetwork
from trainers.base_trainer import BaseTrainer


class ETLPTrainer(BaseTrainer):
    """ETLP trainer with online, local updates (FC + snnTorch LIF)."""

    def __init__(
        self,
        network: FCNetwork,
        lr: float,
        batch_size: int,
        trace_decay: float = 0.9,
        surrogate_scale: float = 0.3,
        voltage_reg: float = 0.0,
        weight_l1: float = 0.0,
        weight_l2: float = 0.0,
        update_rate_hz: float = 100.0,
        dt_ms: float = 1.0,
        feedback_distribution: str = "kaiming_uniform",
        feedback_scale: float = 1.0,
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
        self.trace_decay = float(trace_decay)
        self.surrogate_scale = float(surrogate_scale)
        self.voltage_reg = float(voltage_reg)
        self.weight_l1 = float(weight_l1)
        self.weight_l2 = float(weight_l2)
        self.update_rate_hz = float(update_rate_hz)
        self.dt_ms = float(dt_ms)
        self.feedback_distribution = str(feedback_distribution).lower()
        self.feedback_scale = float(feedback_scale)
        self.quant = bool(quant)
        self.use_optimizer = bool(use_optimizer)
        self.optimizer = optimizer
        self.update_last = bool(update_last)
        self.update_every = int(update_every)

        if not hasattr(self.network, "layers"):
            raise TypeError("ETLPTrainer expects an FCNetwork-like model with .layers.")

        self.linear_layers = [
            layer for layer in self.network.layers if isinstance(layer, nn.Linear)
        ]
        self.lif_layers = [
            layer for layer in self.network.layers if isinstance(layer, snn.Leaky)
        ]
        if len(self.linear_layers) != len(self.lif_layers):
            raise ValueError("Mismatch between Linear and LIF layers in network.")
        if len(self.linear_layers) < 1:
            raise ValueError("ETLPTrainer requires at least one Linear layer.")

        self.n_classes = int(
            getattr(self.network, "n_classes", self.linear_layers[-1].out_features)
        )
        self.threshold = float(getattr(self.network, "threshold", 1.0))

        # Fixed random feedback matrices (one per hidden layer)
        self.feedback = nn.ParameterList()
        for layer in self.linear_layers[:-1]:
            fb = torch.empty((layer.out_features, self.n_classes))
            self._init_feedback_(fb)
            self.feedback.append(nn.Parameter(fb, requires_grad=False))

        # Probability of update trigger per timestep
        self.update_probability = max(
            min(self.update_rate_hz * self.dt_ms * 1e-3, 1.0), 0.0
        )

        self.loss_fn = nn.MSELoss()

    def _init_feedback_(self, tensor: torch.Tensor) -> None:
        if self.feedback_distribution == "kaiming_uniform":
            nn.init.kaiming_uniform_(tensor)
        elif self.feedback_distribution == "uniform":
            tensor.uniform_(-1.0, 1.0)
        elif self.feedback_distribution == "normal":
            tensor.normal_(mean=0.0, std=1.0)
        else:
            raise ValueError(
                "feedback_distribution must be one of "
                '("kaiming_uniform", "uniform", "normal")'
            )
        if self.feedback_scale != 1.0:
            tensor.mul_(self.feedback_scale)

    def _surrogate(self, mem: torch.Tensor) -> torch.Tensor:
        v_scaled = (mem - self.threshold) / self.threshold
        return self.surrogate_scale * torch.clamp(1.0 - torch.abs(v_scaled), min=0.0)

    def _apply_update(
        self,
        spk_rec: List[torch.Tensor],
        mem_rec: List[torch.Tensor],
        pre_traces: List[torch.Tensor],
        targets_onehot: torch.Tensor,
        batch_size: int,
    ) -> None:
        with torch.no_grad():
            labels = targets_onehot
            error = spk_rec[-1] - labels

            grads = []

            # Hidden layers
            for idx in range(len(self.linear_layers) - 1):
                mem = mem_rec[idx]
                psi = self._surrogate(mem)
                e_trace = psi[:, None, :] * pre_traces[idx][:, :, None]
                voltage_mask = torch.logical_or(
                    mem > self.threshold, mem < -self.threshold
                )
                learning_signals = torch.mm(-labels, self.feedback[idx].T)
                learning_signals = (
                    learning_signals + self.voltage_reg * mem * voltage_mask
                )
                grad = torch.einsum("bj,bij->ij", learning_signals, e_trace)
                grad_w = grad.transpose(0, 1)
                grads.append(grad_w)

            # Output layer
            mem_out = mem_rec[-1]
            psi_out = self._surrogate(mem_out)
            e_trace_out = psi_out[:, None, :] * pre_traces[-1][:, :, None]
            grad_out = torch.einsum("bj,bij->ij", error, e_trace_out)
            grad_out_w = grad_out.transpose(0, 1)

            grads.append(grad_out_w)

            # Regularization
            if self.weight_l1 != 0.0 or self.weight_l2 != 0.0:
                for layer_idx, layer in enumerate(self.linear_layers):
                    if self.weight_l1 != 0.0:
                        grads[layer_idx] = grads[
                            layer_idx
                        ] + self.weight_l1 * torch.sign(layer.weight)
                    if self.weight_l2 != 0.0:
                        grads[layer_idx] = (
                            grads[layer_idx] + self.weight_l2 * layer.weight
                        )

            if self.use_optimizer and self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
                for layer, grad_w in zip(self.linear_layers, grads):
                    layer.weight.grad = grad_w
                self.optimizer.step()
            else:
                scale = self.lr / max(float(batch_size), 1.0)
                for layer, grad_w in zip(self.linear_layers, grads):
                    layer.weight.data -= scale * grad_w

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

        tgt_onehot = torch.zeros(batch_size, self.n_classes, device=device)
        tgt_onehot.scatter_(1, target.view(-1, 1), 1.0)

        self.network.reset()

        spk_sum = None
        pre_traces = [
            torch.zeros(batch_size, layer.in_features, device=device)
            for layer in self.linear_layers
        ]

        for t in range(num_timesteps):
            spk_rec, mem_rec = self.network(data[t])
            output_spikes = spk_rec[-1]
            spk_sum = output_spikes if spk_sum is None else spk_sum + output_spikes

            # Update presynaptic traces
            for layer_idx in range(len(self.linear_layers)):
                if layer_idx == 0:
                    pre = data[t]
                else:
                    pre = spk_rec[layer_idx - 1]
                pre_traces[layer_idx] = self.trace_decay * pre_traces[layer_idx] + pre

            if self.update_last and t < num_timesteps - 1:
                continue
            if not ((t + 1) % self.update_every == 0):
                continue

            if self.update_probability < 1.0:
                if torch.rand(1, device=device).item() > self.update_probability:
                    continue

            self._apply_update(spk_rec, mem_rec, pre_traces, tgt_onehot, batch_size)

        loss = self.loss_fn(spk_sum, tgt_onehot)
        pred = spk_sum.argmax(dim=1, keepdim=True)

        return loss, pred

    def reset(self):
        """Reset network state."""
        self.network.reset()
