from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from trainers.base_trainer import BaseTrainer
from networks.decolle_network import DecolleNetwork


def _expand_param(
    value: float | Sequence[float], n_layers: int, name: str
) -> List[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != n_layers:
            raise ValueError(f"{name} must have length {n_layers}, got {len(value)}")
        return [float(v) for v in value]
    return [float(value) for _ in range(n_layers)]


class DECOLLETrainer(BaseTrainer):
    """
    Implementation of the DECOLLE local learning rule (Kaiser et al., 2020).

    The trainer maintains fixed random local readouts (G, H) and performs
    three-factor weight updates online at every timestep:
        ΔW = -η * error * surrogate(U) * P

    Based on the original implementation:
    https://github.com/NeuromorphicProcessorProject/decolle-public
    and lava-decolle: https://github.com/kclip/lava-decolle
    """

    def __init__(
        self,
        network: DecolleNetwork,
        lr: float,
        batch_size: int,
        quant: bool = False,  # interface compatibility
        use_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        update_last: bool = False,  # kept for config parity; ignored
        update_every: int = 1,  # kept for config parity; ignored
        seq_batch_size: int = 1,  # kept for config parity; ignored
        g_scale: float = 0.5,
        burn_in: int = 0,
        eta_bias: Optional[float] = None,
        h_with_noise: bool = False,
        omega_std: float = 0.0,
        surrogate: str = "sigmoid",  # "sigmoid" (original) or "boxcar"
        surrogate_scale: float = 1.0,  # scale for sigmoid surrogate
    ):
        super().__init__()
        if use_optimizer or optimizer is not None:
            raise ValueError("DECOLLETrainer uses manual updates; optimizer not supported.")
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.burn_in = burn_in
        self.eta_bias = eta_bias if eta_bias is not None else lr
        self.h_with_noise = h_with_noise
        self.omega_std = omega_std
        self.surrogate = surrogate
        self.surrogate_scale = surrogate_scale

        self.n_layers = self.network.n_layers
        self.n_classes = self.network.n_classes

        # Fixed random readout and feedback matrices
        self.register_buffer("_dummy", torch.tensor(0.0), persistent=False)
        self.G = nn.ParameterList()
        self.H = nn.ParameterList()
        for idx in range(self.n_layers):
            n_post = self.network.weights[idx].out_features
            g = (2 * g_scale * torch.rand(self.n_classes, n_post) - g_scale)
            self.G.append(nn.Parameter(g, requires_grad=False))

            if h_with_noise:
                omega = torch.normal(
                    mean=torch.ones_like(g).t(), std=omega_std
                )  # shape [n_post, n_classes]
                omega = torch.clamp(omega, min=0)  # sign-concordant
                h = g.t() * omega
            else:
                h = g.t()
            self.H.append(nn.Parameter(h, requires_grad=False))

        # Surrogate gradient window per layer
        self.delta = _expand_param(self.network.delta, self.n_layers, "delta")
        self.loss_fn = nn.MSELoss()

    @torch.no_grad()
    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train the network on one sample (or batch) of shape [T, B, F].
        Returns (loss, pred) where loss is scalar tensor and pred shape [B, 1].
        """
        num_timesteps, batch_size, _ = data.shape
        device = data.device
        num_classes = self.n_classes

        # One-hot targets per batch
        tgt = torch.zeros(batch_size, num_classes, device=device)
        tgt.scatter_(1, target.unsqueeze(1), 1.0)

        # Reset network state
        self.network.reset()
        spk_sum = torch.zeros(batch_size, num_classes, device=device)

        total_loss = 0.0
        total_counts = 0

        for t in range(num_timesteps):
            # Forward one timestep
            spk_list, u_list, p_list = self.network(data[t])
            spk_out = spk_list[-1]
            spk_sum = spk_sum + spk_out

            # Skip updates during burn-in if requested
            apply_learning = t >= self.burn_in

            # Local losses and updates for each layer
            for layer_idx in range(self.n_layers):
                spk_l = spk_list[layer_idx]
                u_l = u_list[layer_idx]
                p_l = p_list[layer_idx]

                # Select matching readout/feedback by postsynaptic width
                g_mat = None
                h_mat = None
                for g_candidate, h_candidate in zip(self.G, self.H):
                    if g_candidate.shape[1] == spk_l.shape[1]:
                        g_mat = g_candidate
                        h_mat = h_candidate
                        break
                if g_mat is None or h_mat is None:
                    raise RuntimeError(
                        f"No matching readout for spikes shape {spk_l.shape}; "
                        f"available G shapes {[g.shape for g in self.G]}"
                    )

                y_l = torch.matmul(spk_l, g_mat.t())  # [B, C]
                delta_y = y_l - tgt
                total_loss += 0.5 * (delta_y.pow(2)).mean()
                total_counts += 1

                if not apply_learning:
                    continue

                # Local error signal via feedback alignment
                # h_mat is [n_post, C], need [B, C] @ [C, n_post] = [B, n_post]
                err_l = torch.matmul(delta_y, h_mat.t())  # [B, n_post]

                # Surrogate gradient gate
                if self.surrogate == "sigmoid":
                    # Sigmoid surrogate (original DECOLLE implementation)
                    # σ'(U) = σ(U) * (1 - σ(U)) scaled
                    sig = torch.sigmoid(self.surrogate_scale * u_l)
                    g_l = self.surrogate_scale * sig * (1.0 - sig)
                else:
                    # Boxcar surrogate (piecewise linear, per paper spec)
                    g_l = ((u_l >= -self.delta[layer_idx]) & (u_l <= self.delta[layer_idx])).float()

                mod = err_l * g_l  # modulatory factor

                # Weight update
                dw = torch.einsum("bi,bj->ij", mod, p_l) / batch_size
                self.network.weights[layer_idx].weight.data.add_(-self.lr * dw)

                # Optional bias update
                if self.network.biases is not None:
                    db = mod.mean(dim=0)
                    self.network.biases[layer_idx].data.add_(-self.eta_bias * db)

        loss = total_loss / max(total_counts, 1)
        pred = spk_sum.argmax(dim=1, keepdim=True)
        return loss, pred

    def reset(self):
        """Reset network state."""
        self.network.reset()

