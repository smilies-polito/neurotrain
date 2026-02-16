"""
OSTL (Online Spatio-Temporal Learning) trainer for feed-forward SNNs.

Implements the three-factor update decomposition from:
    Bohnstingl et al., "Online Spatio-Temporal Learning in Deep Neural Networks", 2020.

This trainer targets the framework's snnTorch-based FCNetwork and follows
the trainer API (train_sample/reset/to) used by the main training loop.
"""

from __future__ import annotations

from typing import List, Optional

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class OSTLTrainer(BaseTrainer):
    """
    Online Spatio-Temporal Learning (OSTL) trainer for feed-forward SNNs only.

    Notes on framework adaptation:
    - The paper formulation includes a separate output readout matrix W_out.
      Here we use an identity readout over the selected final-layer output
      (spikes or membrane) so OSTL fits the existing model interface without
      adding extra model parameters.
    - The framework's FCNetwork has no trainable bias by default; this trainer
      therefore applies OSTL updates to weights only.
    - Recurrent weight matrices are intentionally unsupported in this trainer.
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        surrogate_scale: float = 5.0,
        grad_clip: float = 0.0,
        update_last: bool = False,
        update_every: int = 1,
        quant: bool = False,
        use_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        output_mode: str = "spike",
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.surrogate_scale = float(surrogate_scale)
        self.grad_clip = float(grad_clip)
        self.update_last = bool(update_last)
        self.update_every = int(update_every)
        self.quant = quant
        self.use_optimizer = bool(use_optimizer)
        self.output_mode = str(output_mode).lower()

        if self.output_mode not in ("spike", "mem"):
            raise ValueError(
                f"Invalid OSTL output_mode='{output_mode}'. Use 'spike' or 'mem'."
            )

        if not hasattr(self.network, "layers"):
            raise TypeError(
                "OSTLTrainer expects network.layers with alternating "
                "[nn.Linear, snn.Leaky] modules."
            )
        raw_layers = getattr(self.network, "layers")
        if not isinstance(raw_layers, (nn.ModuleList, list, tuple)):
            raise TypeError(
                "OSTLTrainer expects network.layers to be a ModuleList/list/tuple."
            )
        if len(raw_layers) == 0 or len(raw_layers) % 2 != 0:
            raise TypeError(
                "OSTLTrainer expects an even-length alternating [nn.Linear, "
                "snn.Leaky] structure in network.layers."
            )

        self.linear_layers: List[nn.Linear] = []
        self.lif_layers: List[snn.Leaky] = []
        for idx, layer in enumerate(raw_layers):
            if idx % 2 == 0:
                if not isinstance(layer, nn.Linear):
                    raise TypeError(
                        "OSTLTrainer expects alternating [nn.Linear, snn.Leaky] "
                        f"in network.layers; found {type(layer).__name__} at index {idx}."
                    )
                self.linear_layers.append(layer)
            else:
                if not isinstance(layer, snn.Leaky):
                    raise TypeError(
                        "OSTLTrainer expects alternating [nn.Linear, snn.Leaky] "
                        f"in network.layers; found {type(layer).__name__} at index {idx}."
                    )
                self.lif_layers.append(layer)

        recurrent_layers = getattr(self.network, "recurrent_layers", None)
        if recurrent_layers is not None and len(recurrent_layers) > 0:
            raise TypeError(
                "OSTLTrainer is feed-forward only and does not support "
                "network.recurrent_layers."
            )
        if len(self.linear_layers) != len(self.lif_layers):
            raise RuntimeError(
                "OSTLTrainer internal invariant violated: number of Linear and "
                "snn.Leaky layers must match."
            )

        self.num_layers = len(self.linear_layers)
        self.n_classes = int(getattr(self.network, "n_classes"))

        self.layer_decay = [self._to_float(lif.beta) for lif in self.lif_layers]
        self.layer_threshold = [
            self._to_float(getattr(lif, "threshold", 1.0)) for lif in self.lif_layers
        ]

        self._external_optimizer = optimizer
        if self.use_optimizer:
            self.optimizer = (
                optimizer
                if optimizer is not None
                else torch.optim.Adam(self.network.parameters(), lr=self.lr)
            )
        else:
            self.optimizer = None

    @staticmethod
    def _to_float(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)

    def _surrogate_derivative(
        self, membrane_minus_threshold: torch.Tensor
    ) -> torch.Tensor:
        """
        Logistic surrogate derivative for h'(s) in OSTL equations.
        """
        scaled = self.surrogate_scale * membrane_minus_threshold
        sig = torch.sigmoid(scaled)
        return self.surrogate_scale * sig * (1.0 - sig)

    def _accumulate_or_apply_grad(self, layer: nn.Linear, grad_w: torch.Tensor) -> None:
        if self.grad_clip > 0.0:
            grad_w = grad_w.clamp(-self.grad_clip, self.grad_clip)

        if self.use_optimizer and self.optimizer is not None:
            if layer.weight.grad is None:
                layer.weight.grad = grad_w.clone()
            else:
                layer.weight.grad += grad_w
        else:
            layer.weight.data -= self.lr * grad_w

    @torch.no_grad()
    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Train on one batch.

        Args:
            data: [T, B, F]
            target: [B]
        """
        num_timesteps, batch_size, _ = data.shape
        device = data.device

        if len(self.linear_layers) != len(self.lif_layers):
            raise RuntimeError(
                "OSTLTrainer internal invariant violated: len(linear_layers) "
                "must equal len(lif_layers)."
            )

        target_onehot = torch.zeros(
            batch_size, self.n_classes, device=device, dtype=data.dtype
        )
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
        if target_onehot.shape != (batch_size, self.n_classes):
            raise RuntimeError(
                "OSTLTrainer internal invariant violated: target_onehot must "
                f"have shape ({batch_size}, {self.n_classes})."
            )

        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        eps_w = [
            torch.zeros(
                batch_size, layer.out_features, layer.in_features, device=device
            )
            for layer in self.linear_layers
        ]
        prev_mem = [
            torch.zeros(batch_size, layer.out_features, device=device)
            for layer in self.linear_layers
        ]
        prev_spk = [
            torch.zeros(batch_size, layer.out_features, device=device)
            for layer in self.linear_layers
        ]
        prev_h_prime = [
            torch.zeros(batch_size, layer.out_features, device=device)
            for layer in self.linear_layers
        ]

        output_sum = torch.zeros(
            batch_size, self.n_classes, device=device, dtype=data.dtype
        )
        total_loss = torch.tensor(0.0, device=device)

        for t in range(num_timesteps):
            spk_rec, mem_rec = self.network(data[t])
            if len(spk_rec) != self.num_layers or len(mem_rec) != self.num_layers:
                raise ValueError(
                    "OSTLTrainer expects network forward to return per-layer "
                    "spike/membrane lists with length equal to number of layers."
                )

            output_t = spk_rec[-1] if self.output_mode == "spike" else mem_rec[-1]
            if output_t.shape != (batch_size, self.n_classes):
                raise ValueError(
                    "OSTLTrainer expects final selected output shape "
                    f"({batch_size}, {self.n_classes}), got {tuple(output_t.shape)} "
                    f"for output_mode='{self.output_mode}'."
                )
            output_sum += output_t

            e_w_per_layer = []
            h_prime_per_layer = []

            for layer_idx in range(self.num_layers):
                mem_t = mem_rec[layer_idx]
                pre_t = data[t] if layer_idx == 0 else spk_rec[layer_idx - 1]

                h_prime_t = self._surrogate_derivative(
                    mem_t - self.layer_threshold[layer_idx]
                )
                h_prime_per_layer.append(h_prime_t)

                # Feed-forward SNU adaptation from Eq. (16): ds_t / ds_{t-1}
                ds_ds_prev = self.layer_decay[layer_idx] * (
                    (1.0 - prev_spk[layer_idx])
                    - prev_mem[layer_idx] * prev_h_prime[layer_idx]
                )

                # Eq. (14): epsilon recursion for weights (with g' = 1 for Leaky state update)
                eps_w[layer_idx] = ds_ds_prev.unsqueeze(-1) * eps_w[
                    layer_idx
                ] + pre_t.unsqueeze(1)

                # Eq. (12): eligibility trace
                e_w_t = h_prime_t.unsqueeze(-1) * eps_w[layer_idx]
                e_w_per_layer.append(e_w_t)

            learning_signals = [None] * self.num_layers
            # Fixed identity readout adaptation of Eq. (19):
            # use selected output readout (spikes or membrane) at final layer.
            learning_signals[-1] = output_t - target_onehot

            # Eq. (18): recursive learning signal propagation over layers
            # Propagate across depth via feed-forward weights only.
            for layer_idx in range(self.num_layers - 2, -1, -1):
                next_h_prime = h_prime_per_layer[layer_idx + 1]
                w_next = self.linear_layers[layer_idx + 1].weight
                jacobian_like = next_h_prime.unsqueeze(-1) * w_next.unsqueeze(0)
                learning_signals[layer_idx] = torch.einsum(
                    "bi,bij->bj", learning_signals[layer_idx + 1], jacobian_like
                )

            should_update = True
            if self.update_last:
                should_update = t == num_timesteps - 1
            elif self.update_every > 1:
                should_update = (t + 1) % self.update_every == 0

            if should_update:
                for layer_idx, layer in enumerate(self.linear_layers):
                    grad_w = (
                        torch.einsum(
                            "bi,bij->ij",
                            learning_signals[layer_idx],
                            e_w_per_layer[layer_idx],
                        )
                        / batch_size
                    )
                    self._accumulate_or_apply_grad(layer, grad_w)

                if self.use_optimizer and self.optimizer is not None:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            for layer_idx in range(self.num_layers):
                prev_mem[layer_idx] = mem_rec[layer_idx].detach()
                prev_spk[layer_idx] = spk_rec[layer_idx].detach()
                prev_h_prime[layer_idx] = h_prime_per_layer[layer_idx].detach()

            total_loss += F.mse_loss(output_t, target_onehot)

        loss = total_loss / num_timesteps
        pred = output_sum.argmax(dim=1, keepdim=True)
        return loss.detach(), pred

    def reset(self) -> None:
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def to(self, device):
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self
