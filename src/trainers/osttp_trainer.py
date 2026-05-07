from __future__ import annotations

import math
from typing import List, Optional

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class OSTTPTrainer(BaseTrainer):
    """
    Online Spatio-Temporal Learning with Target Projection (OSTTP) — Ortner et al. (2023).

    OSTTP = OSTL eligibility traces (Eqs. 8–9) + DRTP learning signals (Eq. 11).

    Eligibility traces (Eqs. 8–9, split-Jacobian form):
        ε_{t,W}^l = j_s · ε_{t-1,W}^l  +  x_t  +  J_y · e_{t-1,W}^l
        e_{t,W}^l = ψ_t · ε_{t,W}^l

    where  j_s = ∂s_t/∂s_{t-1}  (diagonal, per reset mechanism)
           J_y = ∂s_t/∂y_{t-1}  (diagonal for Leaky, full [n,n] for RLeaky)
           ψ_t = pseudo-derivative of the spiking function at s_t

    Jacobian components per reset mechanism:
        zero     : j_s = β(1−y_{t-1}),  J_y[diag] = −β·s_{t-1}  [+H for RLeaky]
        subtract : j_s = β,              J_y[diag] = −thresh      [+H for RLeaky]
        none     : j_s = β,              J_y = H only (0 for Leaky)

    Learning signals (Eq. 11 — DRTP):
        output layer K : L_K^t = ∂E_t / ∂y_t^K  (analytic)
        hidden layers  : L_l^t = B_l · y*_t        (fixed random projection)

    Weight update (Eq. 6):
        ΔW_l = (1/B) Σ_t  L_l^t · e_{t,W}^l

    Args:
        network:        nn.Module with `synapses`/`neurons` or `layers` attributes.
        lr:             Learning rate (used when use_optimizer=False).
        batch_size:     Mini-batch size (used to normalise gradient accumulation).
        pseudo:         Surrogate derivative: "tanh" or "fast_sigmoid".
        output_loss:    Loss at output layer: "mse" (default) or "ce".
        feedback_scale: Scale factor for B_l initialisation (default 1.0).
        feedback_seed:  RNG seed for reproducible B_l (default 42).
        grad_clip:      If > 0, clamp gradient entries to ±grad_clip.
        use_optimizer:  If True, accumulate .grad and call optimizer.step().
        optimizer:      Optional external optimizer (Adam is created if None).
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        pseudo: str = "tanh",
        output_loss: str = "mse",
        feedback_scale: float = 1.0,
        feedback_seed: int = 42,
        grad_clip: float = 0.0,
        use_optimizer: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        assert pseudo in ("tanh", "fast_sigmoid"), f"Unknown pseudo: {pseudo!r}"
        assert output_loss in ("mse", "ce"), f"Unknown output_loss: {output_loss!r}"

        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.pseudo = pseudo
        self.output_loss = output_loss
        self.feedback_scale = float(feedback_scale)
        self.feedback_seed = int(feedback_seed)
        self.grad_clip = float(grad_clip)
        self.use_optimizer = bool(use_optimizer)

        if self.lr <= 0.0:
            raise ValueError("OSTTPTrainer requires lr > 0.")
        if self.grad_clip < 0.0:
            raise ValueError("OSTTPTrainer requires grad_clip >= 0.")

        # Resolve layers (same convention as OSTLTrainer)
        self.linear_layers, self.lif_layers = self._resolve_layers(network)
        if len(self.linear_layers) == 0:
            raise TypeError("OSTTPTrainer requires at least one Linear+Leaky layer pair.")

        self.num_layers = len(self.linear_layers)
        self.n_classes = int(getattr(network, "n_classes"))

        # Per-layer metadata
        self.layer_reset = [
            str(getattr(lif, "reset_mechanism", "zero")) for lif in self.lif_layers
        ]
        for idx, rm in enumerate(self.layer_reset):
            if rm not in ("zero", "subtract", "none"):
                raise TypeError(
                    f"OSTTPTrainer supports reset_mechanism in ('zero','subtract','none'); "
                    f"layer {idx} has {rm!r}."
                )

        self.layer_decay = [self._to_scalar(lif.beta, "beta") for lif in self.lif_layers]
        self.layer_threshold = [
            self._to_scalar(getattr(lif, "threshold", 1.0), "threshold")
            for lif in self.lif_layers
        ]

        # Recurrent layers (snn.RLeaky)
        self.is_recurrent_layer = [isinstance(lif, snn.RLeaky) for lif in self.lif_layers]
        self.rec_weights = [
            lif.recurrent.weight if isinstance(lif, snn.RLeaky) else None
            for lif in self.lif_layers
        ]

        # DRTP feedback matrices B_l for hidden layers (l < K).
        # Shape: [C, n_out_l] so that  y* @ B_l  →  [B, n_out_l].
        # Registered as buffers for automatic device/dtype migration.
        self._n_hidden_fb = self.num_layers - 1   # output layer uses analytic signal
        self._fb_names: List[str] = []
        self._fb_ready = False

        # Optimizer
        self._external_optimizer = optimizer
        if self.use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(network.parameters(), lr=self.lr)
        else:
            self.optimizer = None

        # Print initialization summary
        optimizer_name = type(self.optimizer).__name__ if self.optimizer else "None"
        print(f"\n{'='*60}")
        print(f"  OSTTPTrainer")
        print(f"{'='*60}")
        print(f"  {'Learning Rate':<25} {self.lr}")
        print(f"  {'Batch Size':<25} {self.batch_size}")
        print(f"  {'Pseudo Derivative':<25} {self.pseudo}")
        print(f"  {'Output Loss':<25} {self.output_loss}")
        print(f"  {'Feedback Scale':<25} {self.feedback_scale}")
        print(f"  {'Use Optimizer':<25} {self.use_optimizer}")
        print(f"  {'Optimizer':<25} {optimizer_name}")
        print(f"  {'Gradient Clipping':<25} {self.grad_clip}")
        print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Static helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_scalar(value, name: str) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError(
                    f"OSTTPTrainer expects scalar {name}; got tensor with shape {tuple(value.shape)}."
                )
            return float(value.detach().item())
        return float(value)

    @staticmethod
    def _resolve_layers(network: nn.Module):
        """Extract paired (nn.Linear, snn.Leaky/RLeaky) lists from the network.

        Accepts the same two conventions as OSTLTrainer:
          - network.synapses / network.neurons  (explicit ModuleList pair)
          - network.layers                       (alternating [Linear, LIF, ...])
        """
        synapses = getattr(network, "synapses", None)
        neurons = getattr(network, "neurons", None)

        if (synapses is None) != (neurons is None):
            raise TypeError(
                "OSTTPTrainer requires both network.synapses and network.neurons when either is present."
            )

        if synapses is not None and neurons is not None:
            if not isinstance(synapses, (nn.ModuleList, list, tuple)) or not isinstance(
                neurons, (nn.ModuleList, list, tuple)
            ):
                raise TypeError(
                    "OSTTPTrainer expects network.synapses and network.neurons to be ModuleList/list/tuple."
                )
            if len(synapses) == 0 or len(synapses) != len(neurons):
                raise TypeError(
                    "OSTTPTrainer expects equal non-zero lengths for network.synapses and network.neurons."
                )
            linear_layers: List[nn.Linear] = []
            lif_layers = []
            for idx, (syn, neu) in enumerate(zip(synapses, neurons)):
                if not isinstance(syn, nn.Linear):
                    raise TypeError(
                        f"OSTTPTrainer expects nn.Linear in network.synapses, got {type(syn).__name__} at index {idx}."
                    )
                if not isinstance(neu, (snn.Leaky, snn.RLeaky)):
                    raise TypeError(
                        f"OSTTPTrainer expects snn.Leaky or snn.RLeaky in network.neurons, got {type(neu).__name__} at index {idx}."
                    )
                linear_layers.append(syn)
                lif_layers.append(neu)
            return linear_layers, lif_layers

        raw_layers = getattr(network, "layers", None)
        if raw_layers is None:
            raise TypeError(
                "OSTTPTrainer expects either (network.synapses, network.neurons) or network.layers."
            )
        if not isinstance(raw_layers, (nn.ModuleList, list, tuple)):
            raise TypeError("OSTTPTrainer expects network.layers to be a ModuleList/list/tuple.")
        if len(raw_layers) == 0 or len(raw_layers) % 2 != 0:
            raise TypeError(
                "OSTTPTrainer expects network.layers to be an even-length alternating [Linear, LIF] list."
            )
        linear_layers = []
        lif_layers = []
        for idx in range(0, len(raw_layers), 2):
            lin, lif = raw_layers[idx], raw_layers[idx + 1]
            if not isinstance(lin, nn.Linear) or not isinstance(lif, (snn.Leaky, snn.RLeaky)):
                raise TypeError(
                    "OSTTPTrainer expects alternating [nn.Linear, snn.Leaky/snn.RLeaky] entries in network.layers."
                )
            linear_layers.append(lin)
            lif_layers.append(lif)
        return linear_layers, lif_layers

    # -------------------------------------------------------------------------
    # Pseudo-derivative  ψ(s − θ)
    # -------------------------------------------------------------------------

    def _psi(self, mem: torch.Tensor, thresh: float) -> torch.Tensor:
        x = mem - thresh
        if self.pseudo == "tanh":
            return 1.0 - torch.tanh(x).pow(2)
        # fast_sigmoid: 1 / (100|x| + 1)²
        return 1.0 / (100.0 * x.abs() + 1.0).pow(2)

    # -------------------------------------------------------------------------
    # Output loss / learning signal
    # -------------------------------------------------------------------------

    def _loss_value(self, y_out: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
        if self.output_loss == "ce":
            return -(y_star * F.log_softmax(y_out, dim=1)).sum(1).mean()
        return F.mse_loss(y_out, y_star)

    def _output_signal(self, y_out: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
        """L_K^t = ∂E_t / ∂y_t^K  (analytic gradient of loss)."""
        if self.output_loss == "ce":
            return torch.softmax(y_out, dim=1) - y_star
        return y_out - y_star   # MSE gradient

    # -------------------------------------------------------------------------
    # DRTP feedback matrices
    # -------------------------------------------------------------------------

    def _ensure_feedback(self, device, dtype) -> None:
        if self._fb_ready:
            return
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.feedback_seed)
        std = self.feedback_scale / math.sqrt(max(self.n_classes, 1))
        for l in range(self._n_hidden_fb):
            n_out = self.linear_layers[l].out_features
            B = torch.empty(self.n_classes, n_out, dtype=dtype)
            B.normal_(mean=0.0, std=std, generator=gen)
            name = f"_fb_{l}"
            self.register_buffer(name, B.to(device), persistent=True)
            self._fb_names.append(name)
        self._fb_ready = True

    def _get_fb(self, l: int) -> torch.Tensor:
        return getattr(self, self._fb_names[l])

    # -------------------------------------------------------------------------
    # Gradient application
    # -------------------------------------------------------------------------

    def _apply_update(
        self,
        layer: nn.Linear,
        g: torch.Tensor,
        rec_w: Optional[torch.Tensor],
        g_r: Optional[torch.Tensor],
    ) -> None:
        if self.grad_clip > 0.0:
            g = g.clamp(-self.grad_clip, self.grad_clip)
            if g_r is not None:
                g_r = g_r.clamp(-self.grad_clip, self.grad_clip)
        if self.use_optimizer and self.optimizer is not None:
            layer.weight.grad = g if layer.weight.grad is None else layer.weight.grad.add_(g)
            if rec_w is not None and g_r is not None:
                rec_w.grad = g_r if rec_w.grad is None else rec_w.grad.add_(g_r)
        else:
            layer.weight.add_(-self.lr * g)
            if rec_w is not None and g_r is not None:
                rec_w.add_(-self.lr * g_r)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Train one mini-batch.

        Args:
            data:   [T, B, ...]  — time-first input
            target: [B]          — class labels
        Returns:
            loss: scalar, pred: [B, 1]
        """
        if data.dim() < 3:
            raise ValueError(
                f"OSTTPTrainer expects input shape [T, B, ...], got {tuple(data.shape)}."
            )

        T = data.shape[0]
        B = data.shape[1]
        device = data.device
        dtype = data.dtype if data.is_floating_point() else torch.float32

        if target.dim() != 1 or target.shape[0] != B:
            raise ValueError(
                f"OSTTPTrainer expects target shape [B], got {tuple(target.shape)} for batch size {B}."
            )

        # One-hot targets [B, C]
        y_star = torch.zeros(B, self.n_classes, device=device, dtype=dtype)
        y_star.scatter_(1, target.long().unsqueeze(1), 1.0)

        self._ensure_feedback(device, dtype)
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # -----------------------------------------------------------------------
        # Per-layer eligibility state
        #   ε_W[l]: [B, n_out, n_in]  — current eligibility trace for W
        #   e_W[l]: [B, n_out, n_in]  — e from PREVIOUS timestep (used in J_y term)
        #   ε_H[l]: [B, n_out, n_out] — eligibility trace for H (RLeaky only)
        #   e_H[l]: [B, n_out, n_out] — e_H from previous timestep
        # -----------------------------------------------------------------------
        eps_W: List[torch.Tensor] = []
        e_W: List[torch.Tensor] = []
        dW: List[torch.Tensor] = []

        eps_H: List[Optional[torch.Tensor]] = []
        e_H: List[Optional[torch.Tensor]] = []
        dH: List[Optional[torch.Tensor]] = []

        for l, layer in enumerate(self.linear_layers):
            n_i, n_o = layer.in_features, layer.out_features
            eps_W.append(layer.weight.new_zeros(B, n_o, n_i))
            e_W.append(layer.weight.new_zeros(B, n_o, n_i))
            dW.append(torch.zeros(n_o, n_i, device=device, dtype=dtype))

            if self.is_recurrent_layer[l] and self.rec_weights[l] is not None:
                eps_H.append(layer.weight.new_zeros(B, n_o, n_o))
                e_H.append(layer.weight.new_zeros(B, n_o, n_o))
                dH.append(torch.zeros(n_o, n_o, device=device, dtype=dtype))
            else:
                eps_H.append(None)
                e_H.append(None)
                dH.append(None)

        # Stored membrane / spike from previous timestep (for Jacobian)
        prev_spk = [layer.weight.new_zeros(B, layer.out_features) for layer in self.linear_layers]
        prev_mem = [layer.weight.new_zeros(B, layer.out_features) for layer in self.linear_layers]

        output_sum = torch.zeros(B, self.n_classes, device=device, dtype=dtype)
        total_loss = torch.zeros((), device=device, dtype=dtype)
        last_mem_out = output_sum  # will be overwritten at each timestep

        # -----------------------------------------------------------------------
        # TIME LOOP — Algorithm from Ortner et al. (2023)
        # -----------------------------------------------------------------------
        for t in range(T):

            spk_rec, mem_rec = self.network(data[t])
            if len(spk_rec) != self.num_layers or len(mem_rec) != self.num_layers:
                raise ValueError(
                    "OSTTPTrainer expects network(data[t]) to return spike/membrane lists "
                    f"of length {self.num_layers}."
                )

            # Output readout: membrane of the last spiking layer
            y_out = mem_rec[-1]
            spk_out = spk_rec[-1]  # Extract the actual spikes
            last_mem_out = y_out   # Track final-timestep membrane for integrator predictions
            output_sum.add_(spk_out)  # Accumulate spikes for the prediction metric
            total_loss = total_loss + self._loss_value(y_out, y_star) # Keep using membrane for the loss

            # -------------------------------------------------------------------
            # Per-layer eligibility update (Eqs. 8–9)
            # -------------------------------------------------------------------
            for l, layer in enumerate(self.linear_layers):
                beta = self.layer_decay[l]
                thresh = self.layer_threshold[l]
                s_t = mem_rec[l]                   # [B, n_out] current membrane
                y_prev = prev_spk[l]               # [B, n_out] spike from t-1
                s_prev = prev_mem[l]               # [B, n_out] membrane from t-1

                # Presynaptic input x_t: data for layer 0, spikes from layer l-1 otherwise
                x_t = data[t] if l == 0 else spk_rec[l - 1]
                if x_t.dim() > 2:
                    x_t = x_t.flatten(1)
                x_t = x_t.to(dtype=dtype)

                # Surrogate derivative ψ_t = ∂y_t/∂s_t
                psi_t = self._psi(s_t, thresh)     # [B, n_out]

                # Jacobian components (split form, Eq. 8):
                #   j_s: diagonal ∂s_t/∂s_{t-1}  [B, n_out]
                #   diag_y: diagonal part of ∂s_t/∂y_{t-1}  [B, n_out]
                #   H: recurrent weight matrix [n_out, n_out] (RLeaky only)
                rm = self.layer_reset[l]
                if rm == "zero":
                    j_s = beta * (1.0 - y_prev)        # [B, n_out]
                    diag_y = -beta * s_prev             # [B, n_out]
                elif rm == "subtract":
                    j_s = torch.full_like(y_prev, beta) # [B, n_out]
                    diag_y = torch.full_like(y_prev, -thresh)
                else:  # "none"
                    j_s = torch.full_like(y_prev, beta)
                    diag_y = torch.zeros_like(y_prev)

                H = self.rec_weights[l]

                # --- Weight (W) eligibility ---
                # J_y · e_{t-1,W}:
                #   Leaky: diagonal multiply  →  diag_y[b,i] * e_W[l][b,i,j]
                #   RLeaky: (H + diag(diag_y)) @ e_{t-1,W}  →  [B, n_out, n_in]
                if H is not None:
                    # J_y = H + diag(diag_y) as [B, n_out, n_out]
                    J_y = H.detach().unsqueeze(0) + torch.diag_embed(diag_y)
                    Jy_ew = torch.bmm(J_y, e_W[l])         # [B, n_out, n_in]
                else:
                    Jy_ew = diag_y.unsqueeze(-1) * e_W[l]  # [B, n_out, n_in]

                # Eq. 8: ε_t = j_s · ε_{t-1} + x_t + J_y · e_{t-1}
                ds_dW = x_t.unsqueeze(1).expand(-1, layer.out_features, -1)  # [B, n_out, n_in]
                eps_W[l] = j_s.unsqueeze(-1) * eps_W[l] + ds_dW + Jy_ew

                # Eq. 9: e_t = ψ_t · ε_t
                e_W[l] = psi_t.unsqueeze(-1) * eps_W[l]    # [B, n_out, n_in]

                # --- Recurrent weight (H) eligibility (RLeaky only) ---
                if H is not None and eps_H[l] is not None:
                    Jy_eh = torch.bmm(J_y, e_H[l])         # [B, n_out, n_out]
                    rec_pre = y_prev.to(dtype=dtype)        # [B, n_out]  = y_{t-1}
                    ds_dH = rec_pre.unsqueeze(1).expand(-1, layer.out_features, -1)
                    eps_H[l] = j_s.unsqueeze(-1) * eps_H[l] + ds_dH + Jy_eh
                    e_H[l] = psi_t.unsqueeze(-1) * eps_H[l]

                # --- Learning signal L_l^t (Eq. 11) ---
                if l == self.num_layers - 1:
                    # Output layer: analytic gradient ∂E/∂y_K
                    L = self._output_signal(y_out, y_star)  # [B, n_out_K]
                else:
                    # Hidden layer: DRTP projection B_l · y*  →  [B, n_out]
                    L = torch.mm(y_star, self._get_fb(l))   # [B, n_out]

                # --- Gradient accumulation (Eq. 6) ---
                # ΔW += Σ_b L[b] · e_W[b]  (sum over batch, divide by B after loop)
                dW[l].add_(torch.einsum("bi,bij->ij", L, e_W[l]))
                if dH[l] is not None and e_H[l] is not None:
                    dH[l].add_(torch.einsum("bi,bij->ij", L, e_H[l]))

            # Advance stored state
            for l in range(self.num_layers):
                prev_spk[l] = spk_rec[l]
                prev_mem[l] = mem_rec[l]

        # -----------------------------------------------------------------------
        # Apply accumulated gradients (deferred, after full sequence)
        # -----------------------------------------------------------------------
        denom = float(max(B, 1))
        for l, layer in enumerate(self.linear_layers):
            g = dW[l] / denom
            g_r = (dH[l] / denom) if dH[l] is not None else None
            rec_w = self.rec_weights[l]
            self._apply_update(layer, g, rec_w, g_r)

        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        loss = total_loss / float(T)
        use_mem_pred = bool(getattr(self.network, "out_integrator", False))
        pred = last_mem_out.argmax(dim=1, keepdim=True) if use_mem_pred else output_sum.argmax(dim=1, keepdim=True)
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
