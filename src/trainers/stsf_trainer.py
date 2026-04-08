from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from trainers.base_trainer import BaseTrainer


def SFMatrix(n_classes: int, n_hidden: int) -> torch.Tensor:
    """Sparse feedback matrix [n_classes, n_hidden] per STSF paper (He et al. 2025).

    Connectivity: each output class (row) is connected to a random subset of hidden
    neurons, with the guarantee that every hidden neuron (column) has at least one
    non-zero entry — no hidden neuron is left without a feedback signal.

    Init bound follows the reference implementation:
        bd = sqrt(n_hidden / n_classes)
    Values are drawn uniformly from [-bd, +bd] at non-zero positions.
    """
    # One random hidden-neuron index per class → [n_classes] indices into [0, n_hidden)
    perm = torch.randperm(math.ceil(n_classes / n_hidden + 1) * n_hidden)
    index = (perm % n_hidden)[:n_classes]                   # [n_classes]
    mat = torch.zeros(n_classes, n_hidden)
    mat.scatter_(1, index.unsqueeze(1), 1)                  # [n_classes, n_hidden] connectivity mask

    # Guarantee every column (hidden neuron) has at least one non-zero entry.
    zero_cols = (mat.sum(dim=0) == 0).nonzero(as_tuple=False).squeeze(1)
    if zero_cols.numel() > 0:
        for col in zero_cols:
            row = torch.randint(0, n_classes, (1,))
            mat[row, col] = 1.0

    bd = math.sqrt(n_hidden / n_classes)                    # reference bound
    mat = (2 * bd * torch.rand_like(mat) - bd) * mat        # uniform in [-bd, +bd]
    return mat                                               # [n_classes, n_hidden]


class STSFTrainer(BaseTrainer):
    """Spiking Time Sparse Feedback (STSF) — He et al. 2025.

    Hybrid local/global learning rule for feedforward SNNs:
      - Local term  (Vanilla STDP): e_stdp(t) = s_post(t) * s_pre(t)
      - Global term (sparse DFA):   δ_h(t)    = G_h @ (s_o(t) - s_d(t))
      - Weight update:              Δw(t)      = δ(t) * e_stdp(t)

    The output layer uses the "s_o = 1" trick from the paper: the post-synaptic
    output spike is assumed to be 1 so only the pre-synaptic (hidden) spike
    gates the update. This allows weight changes even when the output is silent.

    Requires at least one hidden layer.
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        use_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,          # absorbs unknown args from the benchmark runner (e.g. quant=False)
    ):
        super().__init__()

        self.network       = network
        self.lr            = float(lr)
        self.batch_size    = int(batch_size)
        self.loss_fn       = nn.MSELoss()
        self.use_optimizer = bool(use_optimizer)

        # Combined scaling factor applied to every weight update (manual SGD path):
        #   lr * (2 / n_classes) / batch_size
        # The (2 / n_classes) term matches the MSE gradient scale so that the
        # hidden and output layers share the same effective step size.
        self.scale = self.lr * (2.0 / network.n_classes) / self.batch_size

        # Fixed sparse feedback matrices, one per hidden layer.
        # Shape: [n_classes, n_hidden_l] — not updated during training.
        n_out         = network.n_classes
        hidden_sizes  = network.hidden_size
        self.feedback = nn.ParameterList([
            nn.Parameter(SFMatrix(n_out, h), requires_grad=False)
            for h in hidden_sizes
        ])

        # Optimizer setup — mirrors OSTLTrainer convention:
        #   use_optimizer=False  → manual weight.data updates (original paper SGD)
        #   use_optimizer=True, optimizer=None  → Adam created internally
        #   use_optimizer=True, optimizer=<obj> → caller-supplied optimizer used as-is
        self._external_optimizer = optimizer
        if self.use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(self.network.parameters(), lr=self.lr)
        else:
            self.optimizer = None

        # Print initialization summary
        optimizer_name = type(self.optimizer).__name__ if self.optimizer else "None"
        print(f"\n{'='*60}")
        print(f"  STSFTrainer")
        print(f"{'='*60}")
        print(f"  {'Learning Rate':<25} {self.lr}")
        print(f"  {'Batch Size':<25} {self.batch_size}")
        print(f"  {'Use Optimizer':<25} {self.use_optimizer}")
        print(f"  {'Optimizer':<25} {optimizer_name}")
        print(f"{'='*60}\n")

    @torch.no_grad()
    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """Train on one mini-batch.

        Args:
            data:   [T, B, ...] time-major tensor (time, batch, then any shape)
            target: [B]         (class labels)

        Returns:
            loss: scalar MSE loss over accumulated output spikes
            pred: [B, 1] predicted class indices
        """
        T, B = data.shape[:2]
        n_classes  = self.network.n_classes
        device     = data.device

        # One-hot target: [B, n_classes]
        tgt = torch.zeros(B, n_classes, device=device)
        tgt.scatter_(1, target.unsqueeze(1), 1.0)

        self.network.reset()
        if self.use_optimizer:
            self.optimizer.zero_grad()
            # Initialise gradient accumulators to zero (shape not yet known).
            for p in self.network.parameters():
                p.grad = None

        spk_sum = None

        for t in range(T):
            spks, _ = self.network(data[t])         # spks: list of [B, hidden_l] per layer
            spk_sum = spks[-1] if spk_sum is None else spk_sum + spks[-1]

            # Output error: δ_o = s_o - s_d   [B, n_classes]
            error = spks[-1] - tgt

            # --- Hidden layer updates ---
            for l in range(len(self.network.hidden_size)):
                x_pre  = data[t].reshape(data[t].shape[0], -1) if l == 0 else spks[l - 1]  # pre-synaptic spikes (flattened for l=0)
                x_post = spks[l]                               # post-synaptic spikes

                # Sparse feedback projects output error into hidden space (DFA)
                delta_h = error @ self.feedback[l]             # [B, hidden_l]

                # Local × global: Δw = (δ_h ⊙ s_post)ᵀ @ s_pre   [hidden_l, in_l]
                dw = (delta_h * x_post).T @ x_pre

                if self.use_optimizer:
                    # Accumulate mean gradient over B*T; overwriting was the bug.
                    w = self.network.layers[l * 2].weight
                    if w.grad is None:
                        w.grad = dw / (B * T)
                    else:
                        w.grad += dw / (B * T)
                else:
                    self.network.layers[l * 2].weight.data -= self.scale * dw

            # --- Output layer update (s_o = 1 trick) ---
            # Pre-synaptic = last hidden spikes; post-synaptic treated as 1.
            # Δw_out = δ_o.T @ s_h_last   [n_classes, hidden_last]
            dw_out = error.T @ spks[len(self.network.hidden_size) - 1]

            if self.use_optimizer:
                w_out = self.network.layers[-2].weight
                if w_out.grad is None:
                    w_out.grad = dw_out / (B * T)
                else:
                    w_out.grad += dw_out / (B * T)
            else:
                self.network.layers[-2].weight.data -= self.scale * dw_out

        if self.use_optimizer:
            self.optimizer.step()

        # Compare accumulated spike counts against the rate-scaled target (tgt * T).
        # Without this scaling the loss *increases* as the correctly-firing class
        # accumulates more spikes than the one-hot 1.0 target.
        loss = self.loss_fn(spk_sum, tgt * T)
        pred = spk_sum.argmax(dim=1, keepdim=True)
        return loss, pred

    def reset(self):
        """Reset all LIF membrane states."""
        self.network.reset()

    def to(self, device):
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self
