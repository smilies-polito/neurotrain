"""
S-TLLR (STDP-inspired Temporal Local Learning Rule) Trainer.

Implements the three-factor weight update manually in the trainer.
Uses snntorch.functional for loss; weight updates follow the S-TLLR formula.
Reference: Apolinario & Roy, TMLR 2025.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from networks.stllr_layers import LinearSTLLR


def _ce_grad_output(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Gradient of CE loss w.r.t. output (logits).

    For cross_entropy(output, target), d(loss)/d(output) = (softmax - one_hot) / B.
    """
    C = output.shape[1]
    one_hot = F.one_hot(target, num_classes=C).float()
    softmax = F.softmax(output.float(), dim=1)
    return (softmax - one_hot) / output.shape[0]


class STLLRTrainer(BaseTrainer):
    """
    S-TLLR trainer: manual three-factor weight update.

    Forward pass is no-grad; trainer applies the S-TLLR formula for weight updates.
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        delay_ls: int = 5,
        factors_stdp: Optional[List[float]] = None,
        use_optimizer: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.delay_ls = delay_ls
        self.factors_stdp = factors_stdp or [0.2, 0.75, -1.0, 1.0]

        self._external_optimizer = optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on a single batch using S-TLLR.

        Args:
            data: [T, B, F]
            target: [B]

        Returns:
            loss: scalar
            pred: [B] class indices
        """
        T, B, _ = data.shape
        n_classes = self.network.n_classes

        self.network.reset()
        self.optimizer.zero_grad()

        # Accumulate predictions (no grad)
        pred_sum = torch.zeros(B, n_classes, device=data.device, dtype=data.dtype)

        # Accumulate weight gradients for last delay_ls steps
        layers: List[LinearSTLLR] = [
            layer for layer in self.network.layers if isinstance(layer, LinearSTLLR)
        ]
        factors = layers[0].factors  # [λ_post, λ_pre, α_post, α_pre]
        alpha_post, alpha_pre = factors[2], factors[3]

        grad_weight_acc = [torch.zeros_like(layer.weight) for layer in layers]
        grad_bias_acc = [
            torch.zeros_like(layer.bias) if layer.bias is not None else None
            for layer in layers
        ]

        with torch.no_grad():
            for t in range(T):
                spk_rec, mem_rec = self.network(data[t])
                out = spk_rec[-1]
                pred_sum = pred_sum + out

                if (T - 1 - t) < self.delay_ls:
                    # Compute gradient of CE w.r.t. output (treating spikes as logits)
                    grad_output = _ce_grad_output(out, target)

                    # Propagate backward through layers and accumulate grad_weight
                    grad_next = grad_output
                    for i in range(len(layers) - 1, -1, -1):
                        layer = layers[i]
                        inp = layer.last_input
                        trace_in = layer.last_trace_in
                        trace_out = layer.last_trace_out
                        psi = layer.last_psi
                        weight = layer.get_weight()

                        # Three-factor rule (reference: STLLRLinearGrad.backward)
                        # grad_weight = α_post * (grad_output * trace_out).T @ input
                        #            + α_pre * (grad_output * psi).T @ trace_in
                        gw = alpha_post * (
                            (grad_next * trace_out).T @ inp
                        ) + alpha_pre * ((grad_next * psi).T @ trace_in)
                        grad_weight_acc[i] = grad_weight_acc[i] + gw

                        if layer.bias is not None:
                            grad = psi * grad_next
                            grad_bias_acc[i] = grad_bias_acc[i] + grad.sum(dim=0)

                        # Propagate: grad_input = (psi * grad_next) @ weight
                        grad_next = (psi * grad_next) @ weight

        # Apply accumulated gradients via optimizer
        for i, layer in enumerate(layers):
            if layer.weight.grad is None:
                layer.weight.grad = grad_weight_acc[i]
            else:
                layer.weight.grad = layer.weight.grad + grad_weight_acc[i]
            if layer.bias is not None and grad_bias_acc[i] is not None:
                if layer.bias.grad is None:
                    layer.bias.grad = grad_bias_acc[i]
                else:
                    layer.bias.grad = layer.bias.grad + grad_bias_acc[i]

        self.optimizer.step()

        # Loss for logging (use accumulated pred, snnTorch-style)
        with torch.no_grad():
            loss = F.cross_entropy(pred_sum, target)
            pred = pred_sum.argmax(dim=1, keepdim=True).squeeze(-1)

        return loss, pred

    def reset(self) -> None:
        self.network.reset()

    def to(self, device):
        super().to(device)
        if self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), lr=self.lr
            )
        return self
