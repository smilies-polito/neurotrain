"""
OSTL (Online Spatio-Temporal Learning) trainer for feed-forward SNNs.

Implements the deep feed-forward SNU formulation from:
    Bohnstingl et al., "Online Spatio-Temporal Learning in Deep Neural Networks", 2020.
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
    OSTL trainer for feed-forward linear+Leaky SNN stacks.

    Supported structures:
    - `network.synapses` + `network.neurons` (preferred)
    - alternating `[nn.Linear, snn.Leaky, ...]` in `network.layers` (legacy)

    Unsupported:
    - recurrent layers (`network.recurrent_layers`)
    - convolutional trainable layers
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
        del kwargs

        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.surrogate_scale = float(surrogate_scale)
        self.grad_clip = float(grad_clip)
        self.update_last = bool(update_last)
        self.update_every = int(update_every)
        self.quant = bool(quant)
        self.use_optimizer = bool(use_optimizer)
        self.output_mode = str(output_mode).lower()

        if self.lr <= 0.0:
            raise ValueError("OSTLTrainer requires lr > 0.")
        if self.surrogate_scale <= 0.0:
            raise ValueError("OSTLTrainer requires surrogate_scale > 0.")
        if self.grad_clip < 0.0:
            raise ValueError("OSTLTrainer requires grad_clip >= 0.")
        if self.update_every <= 0:
            raise ValueError("OSTLTrainer requires update_every >= 1.")
        if self.output_mode not in ("spike", "mem"):
            raise ValueError(
                f"Invalid output_mode '{output_mode}'. Use 'spike' or 'mem'."
            )

        self.linear_layers, self.lif_layers = self._resolve_layers(self.network)
        if len(self.linear_layers) == 0:
            raise TypeError(
                "OSTLTrainer requires at least one Linear+Leaky layer pair."
            )

        recurrent_layers = getattr(self.network, "recurrent_layers", None)
        if recurrent_layers is not None and len(recurrent_layers) > 0:
            raise TypeError(
                "OSTLTrainer is feed-forward only and does not support recurrent layers."
            )

        self.num_layers = len(self.linear_layers)
        self.n_classes = int(getattr(self.network, "n_classes"))

        self.layer_decay = [
            self._to_scalar(lif.beta, "beta") for lif in self.lif_layers
        ]
        self.layer_threshold = [
            self._to_scalar(getattr(lif, "threshold", 1.0), "threshold")
            for lif in self.lif_layers
        ]

        self._external_optimizer = optimizer
        if self.use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(
                self.network.parameters(), lr=self.lr
            )
        else:
            self.optimizer = None

    @staticmethod
    def _to_scalar(value, name: str) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError(
                    f"OSTLTrainer expects scalar {name}; got tensor with shape {tuple(value.shape)}."
                )
            return float(value.detach().item())
        return float(value)

    @staticmethod
    def _resolve_layers(network: nn.Module) -> tuple[List[nn.Linear], List[snn.Leaky]]:
        synapses = getattr(network, "synapses", None)
        neurons = getattr(network, "neurons", None)

        if (synapses is None) != (neurons is None):
            raise TypeError(
                "OSTLTrainer requires both network.synapses and network.neurons when either is present."
            )

        if synapses is not None and neurons is not None:
            if not isinstance(synapses, (nn.ModuleList, list, tuple)) or not isinstance(
                neurons, (nn.ModuleList, list, tuple)
            ):
                raise TypeError(
                    "OSTLTrainer expects network.synapses and network.neurons to be ModuleList/list/tuple."
                )
            if len(synapses) == 0 or len(synapses) != len(neurons):
                raise TypeError(
                    "OSTLTrainer expects equal non-zero lengths for network.synapses and network.neurons."
                )

            linear_layers: List[nn.Linear] = []
            lif_layers: List[snn.Leaky] = []
            for idx, (syn, neu) in enumerate(zip(synapses, neurons)):
                if not isinstance(syn, nn.Linear):
                    raise TypeError(
                        f"OSTLTrainer expects nn.Linear in network.synapses, got {type(syn).__name__} at index {idx}."
                    )
                if not isinstance(neu, snn.Leaky):
                    raise TypeError(
                        f"OSTLTrainer expects snn.Leaky in network.neurons, got {type(neu).__name__} at index {idx}."
                    )
                linear_layers.append(syn)
                lif_layers.append(neu)
            return linear_layers, lif_layers

        raw_layers = getattr(network, "layers", None)
        if raw_layers is None:
            raise TypeError(
                "OSTLTrainer expects either (network.synapses, network.neurons) or network.layers."
            )
        if not isinstance(raw_layers, (nn.ModuleList, list, tuple)):
            raise TypeError(
                "OSTLTrainer expects network.layers to be a ModuleList/list/tuple."
            )
        if len(raw_layers) == 0 or len(raw_layers) % 2 != 0:
            raise TypeError(
                "OSTLTrainer expects network.layers to be an even-length alternating [Linear, Leaky] list."
            )

        linear_layers = []
        lif_layers = []
        for idx in range(0, len(raw_layers), 2):
            linear_layer = raw_layers[idx]
            lif_layer = raw_layers[idx + 1]
            if not isinstance(linear_layer, nn.Linear) or not isinstance(
                lif_layer, snn.Leaky
            ):
                raise TypeError(
                    "OSTLTrainer expects alternating [nn.Linear, snn.Leaky] entries in network.layers."
                )
            linear_layers.append(linear_layer)
            lif_layers.append(lif_layer)
        return linear_layers, lif_layers

    def _surrogate_derivative(
        self, membrane_minus_threshold: torch.Tensor
    ) -> torch.Tensor:
        # Logistic surrogate derivative for h'(s).
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
                layer.weight.grad.add_(grad_w)
            return

        layer.weight.add_(-self.lr * grad_w)

    @torch.no_grad()
    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Train one mini-batch.

        Args:
            data: [T, B, ...]
            target: [B]
        """
        if data.dim() < 3:
            raise ValueError(
                f"OSTLTrainer expects input shape [T, B, ...], got {tuple(data.shape)}."
            )

        num_timesteps = int(data.shape[0])
        batch_size = int(data.shape[1])
        device = data.device

        if target.dim() != 1 or int(target.shape[0]) != batch_size:
            raise ValueError(
                "OSTLTrainer expects target shape [B], "
                f"got {tuple(target.shape)} for batch size {batch_size}."
            )

        value_dtype = data.dtype if data.is_floating_point() else torch.float32
        target_onehot = torch.zeros(
            batch_size, self.n_classes, device=device, dtype=value_dtype
        )
        target_onehot.scatter_(1, target.long().unsqueeze(1), 1.0)

        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        eps_w = [
            layer.weight.new_zeros(batch_size, layer.out_features, layer.in_features)
            for layer in self.linear_layers
        ]
        prev_mem = [
            layer.weight.new_zeros(batch_size, layer.out_features)
            for layer in self.linear_layers
        ]
        prev_spk = [
            layer.weight.new_zeros(batch_size, layer.out_features)
            for layer in self.linear_layers
        ]
        prev_h_prime = [
            layer.weight.new_zeros(batch_size, layer.out_features)
            for layer in self.linear_layers
        ]

        grad_buffer = [torch.zeros_like(layer.weight) for layer in self.linear_layers]
        pending_grads = 0

        output_sum = torch.zeros(
            batch_size, self.n_classes, device=device, dtype=value_dtype
        )
        total_loss = torch.zeros((), device=device, dtype=value_dtype)
        loss_steps = 0

        for t in range(num_timesteps):
            spk_rec, mem_rec = self.network(data[t])
            if len(spk_rec) != self.num_layers or len(mem_rec) != self.num_layers:
                raise ValueError(
                    "OSTLTrainer expects network(data[t]) to return spike/membrane lists "
                    f"of length {self.num_layers}."
                )

            output_t = spk_rec[-1] if self.output_mode == "spike" else mem_rec[-1]
            if output_t.shape != (batch_size, self.n_classes):
                raise ValueError(
                    f"OSTLTrainer expects output shape {(batch_size, self.n_classes)}, "
                    f"got {tuple(output_t.shape)} with output_mode='{self.output_mode}'."
                )
            output_sum.add_(output_t)

            supervised_step = (not self.update_last) or (t == num_timesteps - 1)
            if supervised_step:
                total_loss = total_loss + F.mse_loss(output_t, target_onehot)
                loss_steps += 1

            e_w_per_layer = []
            h_prime_per_layer = []

            for layer_idx, layer in enumerate(self.linear_layers):
                mem_t = mem_rec[layer_idx]
                if mem_t.shape != (batch_size, layer.out_features):
                    raise ValueError(
                        "OSTLTrainer expects membrane tensors shaped "
                        f"(B, {layer.out_features}) for layer {layer_idx}, got {tuple(mem_t.shape)}."
                    )

                pre_t = data[t] if layer_idx == 0 else spk_rec[layer_idx - 1]
                if pre_t.dim() > 2:
                    pre_t = pre_t.flatten(1)
                if pre_t.shape != (batch_size, layer.in_features):
                    raise ValueError(
                        "OSTLTrainer expects presynaptic activity shaped "
                        f"(B, {layer.in_features}) for layer {layer_idx}, got {tuple(pre_t.shape)}."
                    )
                pre_t = pre_t.to(dtype=layer.weight.dtype)

                h_prime_t = self._surrogate_derivative(
                    mem_t - self.layer_threshold[layer_idx]
                )
                h_prime_per_layer.append(h_prime_t)

                # Eq. (16), feed-forward case: g' = 1 for SNU/Leaky state update.
                ds_ds_prev = self.layer_decay[layer_idx] * (
                    (1.0 - prev_spk[layer_idx])
                    - prev_mem[layer_idx] * prev_h_prime[layer_idx]
                )

                # Eq. (14): eligibility-vector recursion for W.
                eps_w[layer_idx] = ds_ds_prev.unsqueeze(-1) * eps_w[
                    layer_idx
                ] + pre_t.unsqueeze(1)

                # Eq. (12): eligibility trace for W.
                e_w_per_layer.append(h_prime_t.unsqueeze(-1) * eps_w[layer_idx])

            if supervised_step:
                learning_signals: List[torch.Tensor] = [
                    torch.empty(0, device=device) for _ in range(self.num_layers)
                ]
                # Eq. (19): L_k = (W_out^T y_k - y*) with identity readout W_out = I.
                learning_signals[-1] = output_t - target_onehot

                # Eq. (18): recursive learning-signal propagation through depth.
                for layer_idx in range(self.num_layers - 2, -1, -1):
                    w_next = self.linear_layers[layer_idx + 1].weight
                    jacobian = h_prime_per_layer[layer_idx + 1].unsqueeze(
                        -1
                    ) * w_next.unsqueeze(0)
                    learning_signals[layer_idx] = torch.einsum(
                        "bi,bij->bj", learning_signals[layer_idx + 1], jacobian
                    )

                for layer_idx in range(self.num_layers):
                    grad_t = (
                        torch.einsum(
                            "bi,bij->ij",
                            learning_signals[layer_idx],
                            e_w_per_layer[layer_idx],
                        )
                        / batch_size
                    )
                    grad_buffer[layer_idx].add_(grad_t)
                pending_grads += 1

            if self.update_last:
                should_update = t == num_timesteps - 1
            elif self.update_every > 1:
                should_update = (t + 1) % self.update_every == 0
            else:
                should_update = True

            if should_update and pending_grads > 0:
                for layer_idx, layer in enumerate(self.linear_layers):
                    self._accumulate_or_apply_grad(layer, grad_buffer[layer_idx])
                    grad_buffer[layer_idx].zero_()
                pending_grads = 0

                if self.use_optimizer and self.optimizer is not None:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            for layer_idx in range(self.num_layers):
                prev_mem[layer_idx] = mem_rec[layer_idx]
                prev_spk[layer_idx] = spk_rec[layer_idx]
                prev_h_prime[layer_idx] = h_prime_per_layer[layer_idx]

        # Flush trailing accumulated gradients for non-divisible update_every windows.
        if pending_grads > 0:
            for layer_idx, layer in enumerate(self.linear_layers):
                self._accumulate_or_apply_grad(layer, grad_buffer[layer_idx])
            if self.use_optimizer and self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        if loss_steps == 0:
            raise RuntimeError(
                "OSTLTrainer internal error: no supervised step was processed."
            )

        loss = total_loss / float(loss_steps)
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
