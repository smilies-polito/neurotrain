"""
DECOLLE (Deep Continuous Local Learning) Trainer.

Implements the local learning rule from Kaiser et al., 2020.
Uses FCNetwork for forward pass and manages external eligibility traces.

References:
- Paper: https://www.frontiersin.org/articles/10.3389/fnins.2020.00424
- Original: https://github.com/NeuromorphicProcessorProject/decolle-public
- Lava: https://github.com/kclip/lava-decolle
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import snntorch as snn

from trainers.base_trainer import BaseTrainer
from networks.fc_network import FCNetwork


def _expand_param(
    value: float | Sequence[float], n_layers: int, name: str
) -> List[float]:
    """Expand scalar to per-layer list."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != n_layers:
            raise ValueError(f"{name} must have length {n_layers}, got {len(value)}")
        return [float(v) for v in value]
    return [float(value) for _ in range(n_layers)]


class DECOLLETrainer(BaseTrainer):
    """
    DECOLLE local learning trainer for FCNetwork.

    Maintains external eligibility traces and performs three-factor weight updates:
        ΔW = -η * error * surrogate(U) * P

    The trainer manages P traces independently from FCNetwork's internal
    states, enabling the DECOLLE learning rule with standard snnTorch networks.
    """

    def __init__(
        self,
        network: FCNetwork,
        lr: float,
        batch_size: int,
        quant: bool = False,
        use_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        update_last: bool = False,
        update_every: int = 1,
        seq_batch_size: int = 1,
        # DECOLLE-specific parameters
        g_scale: float = 0.5,
        burn_in: int = 0,
        h_with_noise: bool = False,
        omega_std: float = 0.0,
        surrogate: str = "sigmoid",
        surrogate_scale: float = 5.0,
        delta: float = 0.5,
        lr_scale_per_layer: bool = False,  # Optional heuristic; off by default
        activity_regularizer: float = 0.0,  # L2 penalty on spikes (optional; off by default)
    ):
        super().__init__()
        if use_optimizer or optimizer is not None:
            raise ValueError("DECOLLETrainer uses manual updates; optimizer not supported.")

        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.burn_in = burn_in
        self.h_with_noise = h_with_noise
        self.omega_std = omega_std
        self.surrogate = surrogate
        self.surrogate_scale = surrogate_scale
        self.lr_scale_per_layer = lr_scale_per_layer
        self.activity_regularizer = activity_regularizer

        # Extract network structure
        # FCNetwork.layers is [Linear, Leaky, Linear, Leaky, ...]
        self.linear_layers = [layer for layer in network.layers if isinstance(layer, nn.Linear)]
        self.lif_layers = [layer for layer in network.layers if isinstance(layer, snn.Leaky)]
        self.n_layers = len(self.linear_layers)
        self.n_classes = network.n_classes

        if len(self.lif_layers) != self.n_layers:
            raise ValueError("FCNetwork is expected to alternate Linear and Leaky layers for DECOLLE.")

        # Use the actual neuron parameters to keep traces aligned with the forward dynamics.
        # Each LIF layer shares beta with its preceding Linear layer.
        self.layer_beta = []
        self.layer_threshold = []
        for lif in self.lif_layers:
            beta_val = lif.beta
            thr_val = lif.threshold
            beta_float = float(beta_val.item()) if isinstance(beta_val, torch.Tensor) else float(beta_val)
            thr_float = float(thr_val.item()) if isinstance(thr_val, torch.Tensor) else float(thr_val)
            self.layer_beta.append(beta_float)
            self.layer_threshold.append(thr_float)

        self.delta = _expand_param(delta, self.n_layers, "delta")

        # Fixed random readout (G) and feedback (H) matrices per layer
        self.register_buffer("_dummy", torch.tensor(0.0), persistent=False)
        self.G = nn.ParameterList()
        self.H = nn.ParameterList()

        for idx, linear in enumerate(self.linear_layers):
            n_post = linear.out_features
            g = (2 * g_scale * torch.rand(self.n_classes, n_post) - g_scale)
            self.G.append(nn.Parameter(g, requires_grad=False))

            if h_with_noise:
                omega = torch.normal(mean=torch.ones_like(g).t(), std=omega_std)
                omega = torch.clamp(omega, min=0)
                h = g.t() * omega
            else:
                h = g.t()
            self.H.append(nn.Parameter(h, requires_grad=False))

        # Eligibility trace state (initialized lazily)
        self._trace_initialized = False
        self._trace_batch_size: Optional[int] = None
        self.P: List[torch.Tensor] = []

        self.loss_fn = nn.MSELoss()

    def _ensure_traces(self, batch_size: int, device: torch.device) -> None:
        """Allocate eligibility trace buffers if needed."""
        if self._trace_initialized and self._trace_batch_size == batch_size:
            return

        self.P = []
        for linear in self.linear_layers:
            n_pre = linear.in_features
            self.P.append(torch.zeros(batch_size, n_pre, device=device))

        self._trace_initialized = True
        self._trace_batch_size = batch_size

    def _reset_traces(self) -> None:
        """Reset trace state."""
        self._trace_initialized = False
        self._trace_batch_size = None
        self.P = []

    def _update_traces(self, layer_idx: int, s_pre: torch.Tensor) -> torch.Tensor:
        """
        Update eligibility trace P for a layer and return it.

        For FCNetwork, synapses are instantaneous and membrane follows the LIF
        decay beta. To stay consistent with the forward dynamics, we reuse the
        layer's beta instead of introducing separate tau_syn/tau_mem.
        """
        beta_lif = self.layer_beta[layer_idx]
        self.P[layer_idx] = beta_lif * self.P[layer_idx] + (1.0 - beta_lif) * s_pre

        return self.P[layer_idx]

    @torch.no_grad()
    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on one batch of shape [T, B, F].

        Returns:
            loss: Scalar loss tensor
            pred: Predictions of shape [B, 1]
        """
        num_timesteps, batch_size, _ = data.shape
        device = data.device

        # One-hot targets
        tgt = torch.zeros(batch_size, self.n_classes, device=device)
        tgt.scatter_(1, target.unsqueeze(1), 1.0)

        # Reset states
        self.network.reset()
        self._reset_traces()
        self._ensure_traces(batch_size, device)

        spk_sum = torch.zeros(batch_size, self.n_classes, device=device)
        total_loss = 0.0
        total_counts = 0

        # Diagnostics (accumulated over timesteps)
        self._diag_spike_rates = [0.0] * self.n_layers
        self._diag_surr_grad_mean = [0.0] * self.n_layers
        self._diag_weight_update_norm = [0.0] * self.n_layers
        self._diag_p_trace_mean = [0.0] * self.n_layers

        for t in range(num_timesteps):
            # Forward pass through FCNetwork
            spk_list, mem_list = self.network(data[t])
            spk_sum = spk_sum + spk_list[-1]

            apply_learning = t >= self.burn_in

            # Process each layer for local learning
            s_pre = data[t]  # Input to first layer

            for layer_idx in range(self.n_layers):
                spk_l = spk_list[layer_idx]
                mem_l = mem_list[layer_idx]
                threshold = self.layer_threshold[layer_idx]

                # Update traces and get P for this layer
                p_l = self._update_traces(layer_idx, s_pre)

                # Accumulate diagnostics
                self._diag_spike_rates[layer_idx] += spk_l.mean().item() / num_timesteps
                self._diag_p_trace_mean[layer_idx] += p_l.mean().item() / num_timesteps

                # Get matching G/H matrices
                g_mat = self.G[layer_idx]
                h_mat = self.H[layer_idx]

                # Local readout and loss
                y_l = torch.matmul(spk_l, g_mat.t())  # [B, C]
                delta_y = y_l - tgt
                total_loss += 0.5 * (delta_y.pow(2)).mean()
                if self.activity_regularizer > 0.0:
                    total_loss += 0.5 * self.activity_regularizer * spk_l.pow(2).mean()
                total_counts += 1

                if not apply_learning:
                    s_pre = spk_l
                    continue

                # Local error via feedback alignment
                err_l = torch.matmul(delta_y, h_mat.t())  # [B, n_post]

                # Surrogate gradient centered around the layer threshold
                u_centered = mem_l - threshold

                if self.surrogate == "sigmoid":
                    sig = torch.sigmoid(self.surrogate_scale * u_centered)
                    g_l = self.surrogate_scale * sig * (1.0 - sig)
                else:
                    g_l = ((u_centered >= -self.delta[layer_idx]) &
                           (u_centered <= self.delta[layer_idx])).float()

                # Accumulate surrogate gradient diagnostic
                self._diag_surr_grad_mean[layer_idx] += g_l.mean().item() / num_timesteps

                # Three-factor modulation
                mod = err_l * g_l
                if self.activity_regularizer > 0.0:
                    # Optional spike L2 regularizer: d/dmem (0.5 * λ * s^2) ≈ λ * s * g
                    mod = mod + self.activity_regularizer * spk_l * g_l

                # Weight update: ΔW = -η * (mod^T @ P) / batch_size
                dw = torch.einsum("bi,bj->ij", mod, p_l) / batch_size
                
                # Optional heuristic LR scaling (not part of vanilla DECOLLE)
                if self.lr_scale_per_layer:
                    # Later layers can receive smaller P traces due to sparse intermediate spikes
                    # Scale LR exponentially: layer 0 gets 1x, layer 1 gets 10x, layer 2 gets 100x
                    layer_lr = self.lr * (10 ** layer_idx)
                else:
                    layer_lr = self.lr
                
                self.linear_layers[layer_idx].weight.data.add_(-layer_lr * dw)

                # Track weight update magnitude
                self._diag_weight_update_norm[layer_idx] += (layer_lr * dw.norm().item()) / num_timesteps

                # Prepare input for next layer
                s_pre = spk_l

        loss = total_loss / max(total_counts, 1)
        pred = spk_sum.argmax(dim=1, keepdim=True)
        return loss, pred

    def get_diagnostics(self) -> dict:
        """Return training diagnostics from last batch."""
        return {
            "spike_rates": self._diag_spike_rates if hasattr(self, '_diag_spike_rates') else [],
            "surr_grad_mean": self._diag_surr_grad_mean if hasattr(self, '_diag_surr_grad_mean') else [],
            "weight_update_norm": self._diag_weight_update_norm if hasattr(self, '_diag_weight_update_norm') else [],
            "p_trace_mean": self._diag_p_trace_mean if hasattr(self, '_diag_p_trace_mean') else [],
        }

    def reset(self):
        """Reset network and trace states."""
        self.network.reset()
        self._reset_traces()
