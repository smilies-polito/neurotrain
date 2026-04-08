"""
ETLP (Event-based Three-factor Local Plasticity) trainer.

Implements the local learning rule described in:
Quintana et al., "ETLP: Event-based Three-factor Local Plasticity for
Online Learning with Neuromorphic Hardware", NCE 2024.
"""

from __future__ import annotations

import math
from typing import List, Optional

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class ETLPTrainer(BaseTrainer):
    """
    ETLP: Event-based Three-factor Local Plasticity — Quintana et al. (2024).

    Three factors per weight update (Eq. 4d / Algorithm 1):
      1. Pre-synaptic spike trace  ε_pre  (Eq. 7):   β · ε_pre + x
      2. Surrogate derivative      φ      (Eq. 8):   triangular pseudo-derivative
      3. Learning signal           L      (Eq. 11):  DRTP projection for hidden layers,
                                                     direct MSE gradient for output layer

    Eligibility trace for LIF (Eq. 10):  e = φ · ε_pre

    Weight update — online, stochastic trigger (Algorithm 1):
        ΔW_l = (1/B) · L_l · e_l   when a Poisson teaching event fires at update_rate_hz Hz.

    Args:
        network:              nn.Module with a .layers attribute (alternating Linear, Leaky).
        lr:                   Learning rate (used when use_optimizer=False).
        batch_size:           Mini-batch size (used to normalise gradient accumulation).
        surrogate_scale:      Scale for the triangular pseudo-derivative (Eq. 8, default 0.3).
        weight_l1:            L1 weight regularisation coefficient (default 0.0).
        weight_l2:            L2 weight regularisation coefficient (default 0.0).
        update_rate_hz:       Teaching-event firing rate (Hz).  Algorithm 1 Poisson trigger.
                              Setting this ≥ 1 / (dt_ms * 1e-3) gives probability = 1
                              (deterministic per-timestep updates, useful for benchmarking).
        dt_ms:                Simulation timestep in milliseconds (default 1.0).
        feedback_distribution: Init distribution for fixed random matrices B_l:
                              "normal" (default, N(0, scale/√n_out)) | "uniform" | "kaiming_uniform".
        feedback_scale:       Scale factor applied to B_l initialisation (default 1.0).
        use_optimizer:        If True, accumulate .grad and call optimizer.step().
        optimizer:            External optimizer; Adam is created when use_optimizer=True and None.
        grad_clip:            If > 0, clamp each gradient entry to ±grad_clip.
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        surrogate_scale: float = 0.3,
        weight_l1: float = 0.0,
        weight_l2: float = 0.0,
        update_rate_hz: float = 1000.0,
        dt_ms: float = 1.0,
        feedback_distribution: str = "normal",
        feedback_scale: float = 1.0,
        use_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        grad_clip: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.surrogate_scale = float(surrogate_scale)
        self.weight_l1 = float(weight_l1)
        self.weight_l2 = float(weight_l2)
        self.update_rate_hz = float(update_rate_hz)
        self.dt_ms = float(dt_ms)
        self.feedback_distribution = str(feedback_distribution).lower()
        self.feedback_scale = float(feedback_scale)
        self.use_optimizer = bool(use_optimizer)
        self.grad_clip = float(grad_clip)

        if not hasattr(self.network, "layers"):
            raise TypeError("ETLPTrainer expects a network with a .layers attribute.")

        self.linear_layers: List[nn.Linear] = [
            layer for layer in self.network.layers if isinstance(layer, nn.Linear)
        ]
        self.lif_layers: List[snn.Leaky] = [
            layer for layer in self.network.layers if isinstance(layer, (snn.Leaky, snn.RLeaky))
        ]
        if len(self.linear_layers) != len(self.lif_layers):
            raise ValueError("Mismatch between Linear and LIF layers in network.")
        if len(self.linear_layers) < 1:
            raise ValueError("ETLPTrainer requires at least one Linear layer.")

        self.n_classes: int = int(
            getattr(self.network, "n_classes", self.linear_layers[-1].out_features)
        )
        self.layer_threshold: List[float] = [
            self._to_scalar(getattr(lif, "threshold", 1.0), "threshold")
            for lif in self.lif_layers
        ]
        # Pre-synaptic trace decay = LIF membrane decay constant β (Eq. 7)
        self.layer_decay: List[float] = [
            self._to_scalar(lif.beta, "beta") for lif in self.lif_layers
        ]

        # Detect recurrent layers and cache their recurrent weight matrices.
        # RLeaky has a .recurrent sub-module (nn.Linear) whose .weight [n_out, n_out]
        # is the recurrent connection updated alongside the input weight.
        self.is_recurrent_layer: List[bool] = [
            isinstance(lif, snn.RLeaky) for lif in self.lif_layers
        ]
        self.rec_weights: List[Optional[nn.Parameter]] = [
            lif.recurrent.weight if self.is_recurrent_layer[l] else None
            for l, lif in enumerate(self.lif_layers)
        ]

        # Update probability (Algorithm 1): Poisson teaching events at update_rate_hz Hz
        self.update_probability: float = max(
            min(self.update_rate_hz * self.dt_ms * 1e-3, 1.0), 0.0
        )

        # Fixed random feedback matrices B_l for hidden layers l = 0 … K-2 (Eq. 11, DRTP).
        # Shape: [n_out_l, n_classes], so that  L_l = −y* @ B_l.T  →  [B, n_out_l].
        # Default init: N(0, feedback_scale / √n_out_l)  — matches official implementation.
        self.feedback = nn.ParameterList()
        for layer in self.linear_layers[:-1]:
            fb = torch.empty(layer.out_features, self.n_classes)
            self._init_feedback_(fb, layer.out_features)
            self.feedback.append(nn.Parameter(fb, requires_grad=False))

        self._external_optimizer = optimizer
        if self.use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(network.parameters(), lr=self.lr)
        else:
            self.optimizer = None

        optimizer_name = type(self.optimizer).__name__ if self.optimizer else "None"
        print(f"\n{'='*60}")
        print(f"  ETLPTrainer")
        print(f"{'='*60}")
        print(f"  {'Learning Rate':<25} {self.lr}")
        print(f"  {'Batch Size':<25} {self.batch_size}")
        print(f"  {'Surrogate Scale':<25} {self.surrogate_scale}")
        print(f"  {'Update Rate (Hz)':<25} {self.update_rate_hz}")
        print(f"  {'Timestep (ms)':<25} {self.dt_ms}")
        print(f"  {'Update Probability':<25} {self.update_probability:.4f}")
        print(f"  {'Feedback Distribution':<25} {self.feedback_distribution}")
        print(f"  {'Feedback Scale':<25} {self.feedback_scale}")
        print(f"  {'Gradient Clipping':<25} {self.grad_clip}")
        print(f"  {'Use Optimizer':<25} {self.use_optimizer}")
        print(f"  {'Optimizer':<25} {optimizer_name}")
        print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Static helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_scalar(value, name: str) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError(
                    f"ETLPTrainer expects scalar {name}; got tensor with shape {tuple(value.shape)}."
                )
            return float(value.detach().item())
        return float(value)

    def _init_feedback_(self, tensor: torch.Tensor, n_out: int) -> None:
        if self.feedback_distribution == "normal":
            # N(0, scale / √n_out) — matches official implementation (models.py, b_out init)
            tensor.normal_(mean=0.0, std=self.feedback_scale / math.sqrt(max(n_out, 1)))
        elif self.feedback_distribution == "uniform":
            tensor.uniform_(-self.feedback_scale, self.feedback_scale)
        elif self.feedback_distribution == "kaiming_uniform":
            nn.init.kaiming_uniform_(tensor)
            if self.feedback_scale != 1.0:
                tensor.mul_(self.feedback_scale)
        else:
            raise ValueError(
                "feedback_distribution must be one of ('normal', 'uniform', 'kaiming_uniform')"
            )

    def _surrogate(self, mem: torch.Tensor, thresh: float) -> torch.Tensor:
        """Triangular pseudo-derivative  φ = scale · max(1 − |(v−θ)/θ|, 0)  (Eq. 8)."""
        v_scaled = (mem - thresh) / thresh
        return self.surrogate_scale * torch.clamp(1.0 - torch.abs(v_scaled), min=0.0)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on a single batch using ETLP.

        Args:
            data:   [T, B, ...] — time-first input
            target: [B]         — class labels
        Returns:
            loss: scalar tensor, pred: [B, 1]
        """
        if data.dim() < 3:
            raise ValueError(
                f"ETLPTrainer expects input shape [T, B, ...], got {tuple(data.shape)}."
            )

        T = data.shape[0]
        B = data.shape[1]
        device = data.device
        dtype = data.dtype if data.is_floating_point() else torch.float32

        if target.dim() != 1 or target.shape[0] != B:
            raise ValueError(
                f"ETLPTrainer expects target shape [B], got {tuple(target.shape)} for batch size {B}."
            )

        # One-hot targets [B, C]
        y_star = torch.zeros(B, self.n_classes, device=device, dtype=dtype)
        y_star.scatter_(1, target.long().unsqueeze(1), 1.0)

        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # Pre-synaptic eligibility traces ε_pre[l]: [B, n_in_l]  (Eq. 7)
        # Initialised to zero at the start of each sequence.
        eps_pre = [
            torch.zeros(B, layer.in_features, device=device, dtype=dtype)
            for layer in self.linear_layers
        ]

        # Recurrent pre-synaptic traces ε_rec[l]: [B, n_out_l]  (Eq. 7, recurrent variant)
        # Tracks the layer's own output spikes — the pre-synaptic activity for the
        # recurrent weight W_rec at each timestep.  None for non-recurrent layers.
        eps_rec: List[Optional[torch.Tensor]] = [
            torch.zeros(B, self.linear_layers[l].out_features, device=device, dtype=dtype)
            if self.is_recurrent_layer[l] else None
            for l in range(len(self.linear_layers))
        ]

        spk_sum = torch.zeros(B, self.n_classes, device=device, dtype=dtype)
        total_loss = torch.zeros((), device=device, dtype=dtype)

        # -----------------------------------------------------------------------
        # TIME LOOP — Algorithm 1 of Quintana et al. (2024)
        # -----------------------------------------------------------------------
        for t in range(T):
            spk_rec, mem_rec = self.network(data[t])
            spk_sum.add_(spk_rec[-1])
            total_loss = total_loss + F.mse_loss(spk_rec[-1], y_star)

            # -------------------------------------------------------------------
            # Eq. 7: update pre-synaptic traces  ε_pre ← β · ε_pre + x
            # The trace decays with the same time constant as the LIF membrane.
            # -------------------------------------------------------------------
            for l in range(len(self.linear_layers)):
                x_l = data[t] if l == 0 else spk_rec[l - 1]
                if x_l.dim() > 2:
                    x_l = x_l.flatten(1)
                eps_pre[l] = self.layer_decay[l] * eps_pre[l] + x_l.to(dtype=dtype)

            # Eq. 7 (recurrent variant): ε_rec ← β · ε_rec + spk[l]
            # spk_rec[l] is the current output of lif_layers[l], which is the
            # pre-synaptic spike for the recurrent weight at the next timestep.
            # Always updated (before the stochastic gate) so traces are continuous.
            for l in range(len(self.linear_layers)):
                if self.is_recurrent_layer[l]:
                    eps_rec[l] = self.layer_decay[l] * eps_rec[l] + spk_rec[l].to(dtype=dtype)

            # -------------------------------------------------------------------
            # Stochastic teaching event (Algorithm 1):
            # Update with probability p = update_rate_hz * dt_ms * 1e-3.
            # When update_rate_hz is large enough that p = 1.0, every timestep updates.
            # -------------------------------------------------------------------
            if self.update_probability < 1.0:
                if torch.rand(1, device=device).item() > self.update_probability:
                    continue

            # -------------------------------------------------------------------
            # Compute eligibility traces (Eq. 10) and learning signals (Eq. 11),
            # then apply weight updates (Algorithm 1).
            # -------------------------------------------------------------------
            for l, layer in enumerate(self.linear_layers):
                thresh = self.layer_threshold[l]

                # Eq. 8: surrogate derivative  φ_l = scale · max(1 − |(v−θ)/θ|, 0)
                psi = self._surrogate(mem_rec[l], thresh)             # [B, n_out_l]

                # Eq. 10 (LIF): eligibility trace  e = φ · ε_pre
                # (ε_adapt = 0 for standard snnTorch Leaky)
                # Shape: [B, n_out_l, n_in_l]
                e_trace = psi.unsqueeze(-1) * eps_pre[l].unsqueeze(1)

                # Eq. 10 (recurrent): e_rec = φ · ε_rec — shape [B, n_out_l, n_out_l]
                # psi:[B,n_out,1] * eps_rec:[B,1,n_out] → outer product per sample
                e_trace_rec_l: Optional[torch.Tensor] = None
                if self.is_recurrent_layer[l] and eps_rec[l] is not None:
                    e_trace_rec_l = psi.unsqueeze(-1) * eps_rec[l].unsqueeze(1)

                # Eq. 11: learning signal L_l
                if l < len(self.linear_layers) - 1:
                    # Hidden layer: DRTP — target directly projected via fixed random B_l
                    L = torch.mm(-y_star, self.feedback[l].T)         # [B, n_out_l]
                else:
                    # Output layer: analytic MSE gradient  L_K = y_out − y*
                    L = spk_rec[-1] - y_star                          # [B, n_out_K]

                # ΔW = (1/B) Σ_b L_b · e_b  →  shape [n_out_l, n_in_l] = weight shape
                g = torch.einsum("bi,bij->ij", L, e_trace) / float(max(B, 1))

                if self.weight_l1 != 0.0:
                    g = g + self.weight_l1 * torch.sign(layer.weight)
                if self.weight_l2 != 0.0:
                    g = g + self.weight_l2 * layer.weight
                if self.grad_clip > 0.0:
                    g = g.clamp(-self.grad_clip, self.grad_clip)

                if self.use_optimizer and self.optimizer is not None:
                    layer.weight.grad = (
                        g if layer.weight.grad is None else layer.weight.grad.add_(g)
                    )
                else:
                    layer.weight.add_(-self.lr * g)

                # Recurrent weight update — mirrors the input weight update above.
                if e_trace_rec_l is not None and self.rec_weights[l] is not None:
                    g_rec = torch.einsum("bi,bij->ij", L, e_trace_rec_l) / float(max(B, 1))
                    if self.weight_l1 != 0.0:
                        g_rec = g_rec + self.weight_l1 * torch.sign(self.rec_weights[l])
                    if self.weight_l2 != 0.0:
                        g_rec = g_rec + self.weight_l2 * self.rec_weights[l]
                    if self.grad_clip > 0.0:
                        g_rec = g_rec.clamp(-self.grad_clip, self.grad_clip)
                    if self.use_optimizer and self.optimizer is not None:
                        rw = self.rec_weights[l]
                        rw.grad = g_rec if rw.grad is None else rw.grad.add_(g_rec)
                    else:
                        self.rec_weights[l].add_(-self.lr * g_rec)

            if self.use_optimizer and self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        loss = total_loss / float(T)
        pred = spk_sum.argmax(dim=1, keepdim=True)
        return loss.detach(), pred

    def reset(self) -> None:
        """Reset network state."""
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def to(self, device):
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self
