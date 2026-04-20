from __future__ import annotations

from typing import List, Optional

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class OSTLTrainer(BaseTrainer):
    """
    Online Spatio-Temporal Learning (OSTL) — Bohnstingl et al. (2023).

    Optional variations (all off by default):

    deferred (bool):
        False → online updates: weights are updated at every timestep (Algorithm 1, online branch).
        True  → deferred updates: gradients are accumulated over the full sequence and applied
                once at the end (Algorithm 1, deferred branch). Equivalent to RTRL with deferred
                updates; gradient-equivalent to BPTT for single-layer SNNs.

    feedback_alignment (bool):
        False → standard OSTL: the learning signal is backpropagated through the forward weights
                W^{l+1} (Eq. 18–19).
        True  → OSTL rnd: fixed random matrices B^{l+1} replace W^{l+1} in the learning signal
                (Eq. 26–27). Removes the weight-transport requirement at the cost of approximate
                gradients. Random matrices are drawn once at init and held fixed.

    ostl_complete (bool):
        False → OSTL w/o H (Section III.C): the state Jacobian ds^t/ds^{t-1} is diagonal (Eq. 17),
                ignoring the recurrent weight H. O(Kn²) complexity.
        True  → OSTL complete (Section III.B): the state Jacobian includes the full H·diag(h'^{t-1})
                term (Eq. 24), making it a full [n,n] matrix. Eligibility tensors become rank-3.
                Gradient-equivalent to BPTT for single-layer recurrent SNNs. O(Kn⁴) complexity.
                Only affects recurrent (RLeaky) layers; FF layers are unchanged.
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        grad_clip: float = 0.0,
        use_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        deferred: bool = False,
        feedback_alignment: bool = False,
        ostl_complete: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.grad_clip = float(grad_clip)
        self.use_optimizer = bool(use_optimizer)
        self.deferred = bool(deferred)
        self.feedback_alignment = bool(feedback_alignment)
        self.ostl_complete = bool(ostl_complete)

        if self.lr <= 0.0:
            raise ValueError("OSTLTrainer requires lr > 0.")
        if self.grad_clip < 0.0:
            raise ValueError("OSTLTrainer requires grad_clip >= 0.")

        # Resolve the layers
        self.linear_layers, self.lif_layers = self._resolve_layers(self.network)
        if len(self.linear_layers) == 0:
            raise TypeError("OSTLTrainer requires at least one Linear+Leaky layer pair.")

        # Detect recurrent layers (Section III.B / III.C)
        self.is_recurrent_layer = [isinstance(lif, snn.RLeaky) for lif in self.lif_layers]
        self.has_recurrent = any(self.is_recurrent_layer)
        self.rec_weights = [
            lif.recurrent.weight if isinstance(lif, snn.RLeaky) else None
            for lif in self.lif_layers
        ]

        self.num_layers = len(self.linear_layers)
        self.n_classes = int(getattr(self.network, "n_classes"))

        # Per-layer LIF parameters needed for Eq. (17)
        self.layer_decay = [self._to_scalar(lif.beta, "beta") for lif in self.lif_layers]
        self.layer_threshold = [
            self._to_scalar(getattr(lif, "threshold", 1.0), "threshold") for lif in self.lif_layers
        ]

        for idx, lif in enumerate(self.lif_layers):
            rm = getattr(lif, "reset_mechanism", None)
            lif_type = type(lif).__name__
            if rm != "zero":
                # OSTL's eligibility trace equations (Eq. 17) are derived assuming a
                # zero (hard) reset. Coerce silently so any network can be used.
                import warnings
                warnings.warn(
                    f"OSTLTrainer: coercing reset_mechanism to 'zero' on layer {idx} "
                    f"({lif_type}) (was {rm!r}). OSTL requires zero reset for its "
                    "eligibility trace equations."
                )
                lif.reset_mechanism = "zero"

        # OSTL rnd (Eq. 26): fixed random feedback matrices B^{l+1} for l = 0 … K-2.
        # Same shape as the corresponding forward weight W^{l+1}: [n_{l+1}, n_l].
        # Only the hidden layers need a feedback matrix; the output layer uses Eq. (27) unchanged.
        if self.feedback_alignment:
            self.feedback_weights: Optional[List[torch.Tensor]] = [
                torch.randn(self.linear_layers[l + 1].out_features,
                            self.linear_layers[l + 1].in_features)
                for l in range(self.num_layers - 1)
            ]
        else:
            self.feedback_weights = None

        self._external_optimizer = optimizer
        if self.use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(self.network.parameters(), lr=self.lr)
        else:
            self.optimizer = None

        # Print initialization summary
        optimizer_name = type(self.optimizer).__name__ if self.optimizer else "None"
        print(f"\n{'='*60}")
        print(f"  OSTLTrainer")
        print(f"{'='*60}")
        print(f"  {'Learning Rate':<25} {self.lr}")
        print(f"  {'Batch Size':<25} {self.batch_size}")
        print(f"  {'Gradient Clipping':<25} {self.grad_clip}")
        print(f"  {'Use Optimizer':<25} {self.use_optimizer}")
        print(f"  {'Optimizer':<25} {optimizer_name}")
        print(f"  {'Deferred':<25} {self.deferred}")
        print(f"  {'Feedback Alignment':<25} {self.feedback_alignment}")
        print(f"  {'OSTL Complete':<25} {self.ostl_complete}")
        print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Static helpers
    # -------------------------------------------------------------------------

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
        """Extract paired (nn.Linear, snn.Leaky) lists from the network."""
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
                if not isinstance(neu, (snn.Leaky, snn.RLeaky)):
                    raise TypeError(
                        f"OSTLTrainer expects snn.Leaky or snn.RLeaky in network.neurons, got {type(neu).__name__} at index {idx}."
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
            raise TypeError("OSTLTrainer expects network.layers to be a ModuleList/list/tuple.")
        if len(raw_layers) == 0 or len(raw_layers) % 2 != 0:
            raise TypeError(
                "OSTLTrainer expects network.layers to be an even-length alternating [Linear, Leaky] list."
            )
        linear_layers = []
        lif_layers = []
        for idx in range(0, len(raw_layers), 2):
            lin, lif = raw_layers[idx], raw_layers[idx + 1]
            if not isinstance(lin, nn.Linear) or not isinstance(lif, (snn.Leaky, snn.RLeaky)):
                raise TypeError(
                    "OSTLTrainer expects alternating [nn.Linear, snn.Leaky/snn.RLeaky] entries in network.layers."
                )
            linear_layers.append(lin)
            lif_layers.append(lif)
        return linear_layers, lif_layers

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def _compute_jacobian(
        self, l: int, prev_mem: torch.Tensor, prev_spk: torch.Tensor, prev_h_prime: torch.Tensor
    ) -> torch.Tensor:
        """Compute the state Jacobian ds^t/ds^{t-1} for layer l.

        Returns:
            [B, n] diagonal vector   — FF layers or OSTL w/o H (Eq. 17)
            [B, n, n] full matrix    — OSTL complete on recurrent layers (Eq. 24)
        """
        # Diagonal part (always computed): d · ((1 - y^{t-1}) - s^{t-1} · h'^{t-1})
        diag_part = self.layer_decay[l] * (
            (1.0 - prev_spk) - prev_mem * prev_h_prime
        )

        if self.ostl_complete and self.is_recurrent_layer[l]:
            # Eq. (24): ds^t/ds^{t-1} = H · diag(h'^{t-1}) + d · diag((1-y^{t-1}) - s^{t-1}·h'^{t-1})
            # g' = 1 (identity activation), so diag(g') = I and drops out.
            H = self.rec_weights[l]                                  # [n, n]
            H_term = H.unsqueeze(0) * prev_h_prime.unsqueeze(1)      # [B, n, n]
            return H_term + torch.diag_embed(diag_part)              # [B, n, n]
        else:
            return diag_part  # [B, n]

    def _apply_update(self, layer: nn.Linear, g: torch.Tensor,
                      rec_w: Optional[torch.Tensor], g_r: Optional[torch.Tensor]) -> None:
        """Apply (or accumulate for the optimizer) one gradient step for a layer."""
        if self.use_optimizer and self.optimizer is not None:
            layer.weight.grad = g if layer.weight.grad is None else layer.weight.grad.add_(g)
            if rec_w is not None and g_r is not None:
                rec_w.grad = g_r if rec_w.grad is None else rec_w.grad.add_(g_r)
        else:
            layer.weight.add_(-self.lr * g)
            if rec_w is not None and g_r is not None:
                rec_w.add_(-self.lr * g_r)

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
                f"OSTLTrainer expects input shape [T, B, ...], got {tuple(data.shape)}."
            )

        T = data.shape[0]
        B = data.shape[1]
        device = data.device
        dtype = data.dtype if data.is_floating_point() else torch.float32

        if target.dim() != 1 or target.shape[0] != B:
            raise ValueError(
                f"OSTLTrainer expects target shape [B], got {tuple(target.shape)} for batch size {B}."
            )

        # One-hot targets for MSE output error (Eq. 19 / 27)
        target_oh = torch.zeros(B, self.n_classes, device=device, dtype=dtype)
        target_oh.scatter_(1, target.long().unsqueeze(1), 1.0)

        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # -----------------------------------------------------------------------
        # OSTL running state (reset at the start of each sample/sequence)
        # -----------------------------------------------------------------------
        # elig_vec[l]: eligibility tensor ε^{t,W_l}
        #   FF / w/o H:  [B, out_l, in_l]          (Eq. 15, rank-2)
        #   complete:     [B, out_l, out_l, in_l]   (Eq. 15 with full Jacobian, rank-3)
        elig_vec = []
        for l, layer in enumerate(self.linear_layers):
            if self.ostl_complete and self.is_recurrent_layer[l]:
                elig_vec.append(layer.weight.new_zeros(
                    B, layer.out_features, layer.out_features, layer.in_features))
            else:
                elig_vec.append(layer.weight.new_zeros(
                    B, layer.out_features, layer.in_features))

        # Values from t-1 used in Eq. (17)/(24): stored membrane, spike, surrogate derivative
        prev_mem = [layer.weight.new_zeros(B, layer.out_features) for layer in self.linear_layers]
        prev_spk = [layer.weight.new_zeros(B, layer.out_features) for layer in self.linear_layers]
        prev_h_prime = [layer.weight.new_zeros(B, layer.out_features) for layer in self.linear_layers]

        # Recurrent eligibility state
        #   w/o H:    [B, out_l, out_l]          (Section III.C, rank-2)
        #   complete: [B, out_l, out_l, out_l]   (Section III.B / Eq. 22, rank-3)
        elig_vec_rec = []
        for l, layer in enumerate(self.linear_layers):
            if not self.is_recurrent_layer[l]:
                elig_vec_rec.append(None)
            elif self.ostl_complete:
                elig_vec_rec.append(layer.weight.new_zeros(
                    B, layer.out_features, layer.out_features, layer.out_features))
            else:
                elig_vec_rec.append(layer.weight.new_zeros(
                    B, layer.out_features, layer.out_features))

        # Deferred mode: accumulate gradients here, apply once after the sequence.
        if self.deferred:
            grad_buf = [torch.zeros_like(layer.weight) for layer in self.linear_layers]
            grad_buf_rec = [
                torch.zeros_like(self.rec_weights[l]) if self.is_recurrent_layer[l] else None
                for l in range(self.num_layers)
            ]

        output_sum = torch.zeros(B, self.n_classes, device=device, dtype=dtype)
        total_loss = torch.zeros((), device=device, dtype=dtype)

        # -----------------------------------------------------------------------
        # LOOP OVER TIMESTEPS — Algorithm 1 of Bohnstingl et al. (2023)
        # -----------------------------------------------------------------------
        for t in range(T):

            # Forward pass
            spk_rec, mem_rec = self.network(data[t])
            if len(spk_rec) != self.num_layers or len(mem_rec) != self.num_layers:
                raise ValueError(
                    "OSTLTrainer expects network(data[t]) to return spike/membrane lists "
                    f"of length {self.num_layers}."
                )

            output_t = spk_rec[-1]
            output_sum.add_(output_t)
            total_loss = total_loss + F.mse_loss(output_t, target_oh)

            # -------------------------------------------------------------------
            # Eq. (13, 15): build eligibility vector ε and trace e = h'·ε per layer
            # -------------------------------------------------------------------
            h_prime: List[torch.Tensor] = []                   # surrogate derivative h'_l^t
            elig_trace: List[torch.Tensor] = []                # eligibility trace  e_l^{t,W}
            elig_trace_rec: List[Optional[torch.Tensor]] = []  # eligibility trace  e_l^{t,H}

            for l, layer in enumerate(self.linear_layers):
                mem_t = mem_rec[l]

                # Presynaptic activity x_{l-1}^t (raw input for layer 0, spikes otherwise)
                pre_t = data[t] if l == 0 else spk_rec[l - 1]
                if pre_t.dim() > 2:
                    pre_t = pre_t.flatten(1)
                pre_t = pre_t.to(dtype=layer.weight.dtype)

                # Surrogate derivative h'_l^t = 1 - tanh²(mem_t - θ).
                # mem_t is s^t: the fully updated membrane (decay + input) before any future reset.
                # The zero-reset of s^{t-1} is applied at the start of the s^t computation inside
                # snntorch, so mem_t already reflects correct dynamics with no pre/post-reset ambiguity.
                h_prime_t = 1.0 - torch.tanh(mem_t - self.layer_threshold[l]).pow(2)
                h_prime.append(h_prime_t)

                # State Jacobian ds^t/ds^{t-1}:
                #   Eq. (17) diagonal [B, n]    — FF / w/o H
                #   Eq. (24) full [B, n, n]     — OSTL complete on recurrent layers
                jac = self._compute_jacobian(l, prev_mem[l], prev_spk[l], prev_h_prime[l])

                if self.ostl_complete and self.is_recurrent_layer[l]:
                    # --- OSTL complete: rank-3 eligibility tensors (Section III.B) ---
                    n_out = layer.out_features

                    # Eq. (15) with full Jacobian:
                    # ε^{t,W} = jac @ ε^{t-1,W} + injection_W
                    # jac: [B,n,n], elig_vec[l]: [B,n,n,in]
                    # injection_W[b,o,p,q] = δ_{o,p} · x^t_q
                    eye_n = torch.eye(n_out, device=device, dtype=dtype)
                    injection_W = eye_n.unsqueeze(0).unsqueeze(-1) * pre_t.unsqueeze(1).unsqueeze(1)
                    elig_vec[l] = torch.einsum('bij,bjkl->bikl', jac, elig_vec[l]) + injection_W

                    # Eq. (13): e^{t,W} = diag(h'^t) · ε^{t,W}  → h' broadcasts over last two dims
                    elig_trace.append(h_prime_t.unsqueeze(-1).unsqueeze(-1) * elig_vec[l])

                    # Eq. (22): ε^{t,H} = jac @ ε^{t-1,H} + injection_H
                    # injection_H[b,o,p,q] = δ_{o,p} · y^{t-1}_q
                    rec_pre = prev_spk[l].to(dtype=layer.weight.dtype)
                    injection_H = eye_n.unsqueeze(0).unsqueeze(-1) * rec_pre.unsqueeze(1).unsqueeze(1)
                    elig_vec_rec[l] = torch.einsum('bij,bjkl->bikl', jac, elig_vec_rec[l]) + injection_H

                    # Eq. (21): e^{t,H} = diag(h'^t) · ε^{t,H}
                    elig_trace_rec.append(h_prime_t.unsqueeze(-1).unsqueeze(-1) * elig_vec_rec[l])
                else:
                    # --- OSTL w/o H (default): rank-2 eligibility tensors ---
                    # jac is [B, n] diagonal vector

                    # Eq. (15): ε^{t,W} = jac · ε^{t-1,W} + x^t
                    elig_vec[l] = jac.unsqueeze(-1) * elig_vec[l] + pre_t.unsqueeze(1)

                    # Eq. (13): e^{t,W} = h'^t · ε^{t,W}
                    elig_trace.append(h_prime_t.unsqueeze(-1) * elig_vec[l])

                    # Recurrent weight eligibility (OSTL w/o H, Eq. 22)
                    # Injection term: y_l^{t-1} = prev_spk[l]
                    if self.is_recurrent_layer[l]:
                        rec_pre = prev_spk[l].to(dtype=layer.weight.dtype)
                        elig_vec_rec[l] = jac.unsqueeze(-1) * elig_vec_rec[l] + rec_pre.unsqueeze(1)
                        elig_trace_rec.append(h_prime_t.unsqueeze(-1) * elig_vec_rec[l])
                    else:
                        elig_trace_rec.append(None)

            # -------------------------------------------------------------------
            # Learning signals L_l^t propagated from output to input.
            # Standard OSTL uses forward weights W^{l+1} (Eq. 18).
            # OSTL rnd uses fixed random matrices B^{l+1} instead (Eq. 26).
            # The output learning signal is identical in both cases (Eq. 19 / 27).
            # -------------------------------------------------------------------
            L: List[torch.Tensor] = [torch.empty(0, device=device) for _ in range(self.num_layers)]

            # Output layer error  L_K = ŷ^t - y*  (Eq. 19 / 27)
            L[-1] = output_t - target_oh

            # Hidden layer signals propagated backward through space
            for l in range(self.num_layers - 2, -1, -1):
                # Select feedback matrix: random (OSTL rnd) or forward weight (OSTL)
                fb_weight = (
                    self.feedback_weights[l].to(device)  # B^{l+1}  (Eq. 26)
                    if self.feedback_alignment
                    else self.linear_layers[l + 1].weight  # W^{l+1}  (Eq. 18)
                )
                jac = h_prime[l + 1].unsqueeze(-1) * fb_weight.unsqueeze(0)
                L[l] = torch.einsum("bi,bij->bj", L[l + 1], jac)

            # -------------------------------------------------------------------
            # Eq. (11): Δθ_l = (1/B) Σ_b L_l^t · e_l^{t,θ}
            # Online: apply immediately. Deferred: accumulate into grad_buf.
            # -------------------------------------------------------------------
            for l, layer in enumerate(self.linear_layers):
                if self.ostl_complete and self.is_recurrent_layer[l]:
                    # Rank-3 traces: L:[B,n], e:[B,n,n,in] → contract over b,i → [n,in]
                    g = torch.einsum("bi,bijk->jk", L[l], elig_trace[l]) / B
                else:
                    g = torch.einsum("bi,bij->ij", L[l], elig_trace[l]) / B

                g_r = None
                rec_w = None
                if self.is_recurrent_layer[l] and elig_trace_rec[l] is not None:
                    if self.ostl_complete:
                        g_r = torch.einsum("bi,bijk->jk", L[l], elig_trace_rec[l]) / B
                    else:
                        g_r = torch.einsum("bi,bij->ij", L[l], elig_trace_rec[l]) / B
                    rec_w = self.rec_weights[l]

                if self.deferred:
                    # Accumulate; grad_clip is applied at the end on the full gradient.
                    grad_buf[l].add_(g)
                    if g_r is not None:
                        grad_buf_rec[l].add_(g_r)
                else:
                    # Online: clip and apply now.
                    if self.grad_clip > 0.0:
                        g = g.clamp(-self.grad_clip, self.grad_clip)
                        if g_r is not None:
                            g_r = g_r.clamp(-self.grad_clip, self.grad_clip)
                    self._apply_update(layer, g, rec_w, g_r)

            # Optimizer step after each timestep (online mode only)
            if not self.deferred and self.use_optimizer and self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            # Advance stored state to t for use at t+1
            for l in range(self.num_layers):
                prev_mem[l] = mem_rec[l]
                prev_spk[l] = spk_rec[l]
                prev_h_prime[l] = h_prime[l]

        # -----------------------------------------------------------------------
        # Deferred mode: apply the full accumulated gradient after the sequence.
        # -----------------------------------------------------------------------
        if self.deferred:
            for l, layer in enumerate(self.linear_layers):
                g = grad_buf[l]
                if self.grad_clip > 0.0:
                    g = g.clamp(-self.grad_clip, self.grad_clip)

                g_r = None
                rec_w = None
                if self.is_recurrent_layer[l] and grad_buf_rec[l] is not None:
                    g_r = grad_buf_rec[l]
                    if self.grad_clip > 0.0:
                        g_r = g_r.clamp(-self.grad_clip, self.grad_clip)
                    rec_w = self.rec_weights[l]

                self._apply_update(layer, g, rec_w, g_r)

            if self.use_optimizer and self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        loss = total_loss / float(T)
        pred = output_sum.argmax(dim=1, keepdim=True)
        return loss.detach(), pred

    def reset(self) -> None:
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def to(self, device):
        super().to(device)
        # Move fixed random feedback matrices to the target device.
        if self.feedback_weights is not None:
            self.feedback_weights = [w.to(device) for w in self.feedback_weights]
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self
