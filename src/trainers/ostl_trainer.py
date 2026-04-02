from __future__ import annotations

from typing import List, Optional

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class OSTLTrainer(BaseTrainer):

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        grad_clip: float = 0.0,
        update_last: bool = False,
        update_every: int = 1,
        use_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        output_mode: str = "spike",
        **kwargs,
    ):
        super().__init__()
        del kwargs

        # Save the parameters
        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.grad_clip = float(grad_clip)
        self.update_last = bool(update_last)
        self.update_every = int(update_every)
        self.use_optimizer = bool(use_optimizer)
        self.output_mode = str(output_mode).lower()

        # Validate the parameters
        if self.lr <= 0.0:
            raise ValueError("OSTLTrainer requires lr > 0.")
        if self.grad_clip < 0.0:
            raise ValueError("OSTLTrainer requires grad_clip >= 0.")
        if self.update_every <= 0:
            raise ValueError("OSTLTrainer requires update_every >= 1.")
        if self.output_mode not in ("spike", "mem"):
            raise ValueError(f"Invalid output_mode '{output_mode}'. Use 'spike' or 'mem'.")

        # Resolve the layers
        self.linear_layers, self.lif_layers = self._resolve_layers(self.network)
        if len(self.linear_layers) == 0:
            raise TypeError("OSTLTrainer requires at least one Linear+Leaky layer pair.")

        recurrent_layers = getattr(self.network, "recurrent_layers", None)
        if recurrent_layers is not None and len(recurrent_layers) > 0:
            raise TypeError("OSTLTrainer is feed-forward only and does not support recurrent layers.")

        self.num_layers = len(self.linear_layers)
        self.n_classes = int(getattr(self.network, "n_classes"))

        # Per-layer LIF parameters needed for Eq. (17)
        self.layer_decay = [self._to_scalar(lif.beta, "beta") for lif in self.lif_layers]
        self.layer_threshold = [
            self._to_scalar(getattr(lif, "threshold", 1.0), "threshold") for lif in self.lif_layers
        ]

        for idx, lif in enumerate(self.lif_layers):
            rm = getattr(lif, "reset_mechanism", None)
            if rm != "zero":
                raise TypeError(
                    f"OSTLTrainer requires reset_mechanism='zero' on all snn.Leaky layers "
                    f"(layer {idx} has reset_mechanism={rm!r}). "
                    "Pass reset_mechanism='zero' when constructing the network."
                )

        self._external_optimizer = optimizer
        if self.use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(self.network.parameters(), lr=self.lr)
        else:
            self.optimizer = None

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
            raise TypeError("OSTLTrainer expects network.layers to be a ModuleList/list/tuple.")
        if len(raw_layers) == 0 or len(raw_layers) % 2 != 0:
            raise TypeError(
                "OSTLTrainer expects network.layers to be an even-length alternating [Linear, Leaky] list."
            )
        linear_layers = []
        lif_layers = []
        for idx in range(0, len(raw_layers), 2):
            lin, lif = raw_layers[idx], raw_layers[idx + 1]
            if not isinstance(lin, nn.Linear) or not isinstance(lif, snn.Leaky):
                raise TypeError(
                    "OSTLTrainer expects alternating [nn.Linear, snn.Leaky] entries in network.layers."
                )
            linear_layers.append(lin)
            lif_layers.append(lif)
        return linear_layers, lif_layers

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
        # Control on input shape
        if data.dim() < 3:
            raise ValueError(
                f"OSTLTrainer expects input shape [T, B, ...], got {tuple(data.shape)}."
            )

        # Extract dimensions and device info for later use
        T = data.shape[0]
        B = data.shape[1]
        device = data.device
        dtype = data.dtype if data.is_floating_point() else torch.float32
        # Quick check target shape and type
        if target.dim() != 1 or target.shape[0] != B:
            raise ValueError(
                f"OSTLTrainer expects target shape [B], got {tuple(target.shape)} for batch size {B}."
            )

        # One-hot targets for MSE output error (Eq. 19)
        target_oh = torch.zeros(B, self.n_classes, device=device, dtype=dtype)
        target_oh.scatter_(1, target.long().unsqueeze(1), 1.0)

        # Reset of the network and optimizer
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # -----------------------------------------------------------------------
        # OSTL running state (reset at the start of each sample/sequence)
        # -----------------------------------------------------------------------
        # elig_vec[l]: [B, out_l, in_l] — eligibility tensor ε^{t,W_l} (Eq. 14)
        elig_vec = [
            layer.weight.new_zeros(B, layer.out_features, layer.in_features)
            for layer in self.linear_layers
        ]
        # elig_vec_bias[l]: [B, out_l] — eligibility tensor ε^{t,bias_l} (same recursion, injection=1)
        elig_vec_bias = [
            layer.weight.new_zeros(B, layer.out_features) if layer.bias is not None else None
            for layer in self.linear_layers
        ]
        # Values from t-1 used in Eq. (16): stored membrane, spike, surrogate derivative
        prev_mem = [layer.weight.new_zeros(B, layer.out_features) for layer in self.linear_layers]
        prev_spk = [layer.weight.new_zeros(B, layer.out_features) for layer in self.linear_layers]
        prev_h_prime = [layer.weight.new_zeros(B, layer.out_features) for layer in self.linear_layers]

        # Gradient accumulation buffers (flushed according to update schedule)
        grad_buf = [torch.zeros_like(layer.weight) for layer in self.linear_layers]
        grad_buf_bias = [
            torch.zeros_like(layer.bias) if layer.bias is not None else None
            for layer in self.linear_layers
        ]
        pending = 0   # number of timesteps accumulated in grad_buf since last flush

        output_sum = torch.zeros(B, self.n_classes, device=device, dtype=dtype)
        total_loss = torch.zeros((), device=device, dtype=dtype)
        loss_count = 0

        # -----------------------------------------------------------------------
        # LOOP OVER TIMESTEPS
        # -----------------------------------------------------------------------
        for t in range(T):

            # Forward pass for the network
            spk_rec, mem_rec = self.network(data[t])
            # Quinck sanity check
            if len(spk_rec) != self.num_layers or len(mem_rec) != self.num_layers:
                raise ValueError(
                    "OSTLTrainer expects network(data[t]) to return spike/membrane lists "
                    f"of length {self.num_layers}."
                )

            # Handle different output modes: spike vs. membrane
            output_t = spk_rec[-1] if self.output_mode == "spike" else mem_rec[-1]
            output_sum.add_(output_t)

            # Only compute loss / learning signal on supervised timesteps
            is_supervised = (not self.update_last) or (t == T - 1)
            if is_supervised:
                total_loss = total_loss + F.mse_loss(output_t, target_oh)
                loss_count += 1

            # -------------------------------------------------------------------
            # Eq. (14, 12): build eligibility vector ε and trace e = h'·ε per layer
            # -------------------------------------------------------------------
            h_prime: List[torch.Tensor] = []                    # surrogate derivative h'_l^t
            elig_trace: List[torch.Tensor] = []                 # eligibility trace  e_l^{t,W}
            elig_trace_bias: List[Optional[torch.Tensor]] = []  # eligibility trace  e_l^{t,bias}

            for l, layer in enumerate(self.linear_layers):
                mem_t = mem_rec[l]

                # Presynaptic activity x_{l-1}^t (raw input for layer 0, spikes otherwise)
                pre_t = data[t] if l == 0 else spk_rec[l - 1]
                if pre_t.dim() > 2:
                    pre_t = pre_t.flatten(1)
                pre_t = pre_t.to(dtype=layer.weight.dtype)

                # Surrogate derivative at the current membrane  h'_l^t = h'(V^t - θ)
                # Section V: The pseudoderivative dh(x)/dx = 1 - tanh^2(x) is used.
                h_prime_t = 1.0 - torch.tanh(mem_t - self.layer_threshold[l]).pow(2)
                h_prime.append(h_prime_t)

                # Eq. (17): diagonal state Jacobian  ∂s^t/∂s^{t-1} = d·((1-spk^{t-1}) - s^{t-1}·h'^{t-1})
                # g' = 1 here (identity activation, see module docstring).
                # With zero reset, prev_mem == V_pre at non-spike (exact) and 0 at spike
                # (approximation; h' is near zero there so the error is negligible).
                state_deriv = self.layer_decay[l] * (
                    (1.0 - prev_spk[l]) - prev_mem[l] * prev_h_prime[l]
                )

                # Eq. (15): ε^{t,W} = (∂s^t/∂s^{t-1}) · ε^{t-1,W} + g'·x^t  (g'=1)
                elig_vec[l] = state_deriv.unsqueeze(-1) * elig_vec[l] + pre_t.unsqueeze(1)

                # Eq. (13): e^{t,W} = h'^t · ε^{t,W}
                elig_trace.append(h_prime_t.unsqueeze(-1) * elig_vec[l])

                # Bias eligibility (Eq. 14 / Eq. 15 with injection term = 1)
                if layer.bias is not None:
                    elig_vec_bias[l] = state_deriv * elig_vec_bias[l] + 1.0
                    elig_trace_bias.append(h_prime_t * elig_vec_bias[l])
                else:
                    elig_trace_bias.append(None)

            if is_supervised:
                # ---------------------------------------------------------------
                # Eq. (18, 19): learning signals L_l^t propagated from output to input
                # ---------------------------------------------------------------
                L: List[torch.Tensor] = [torch.empty(0, device=device) for _ in range(self.num_layers)]

                # Eq. (19): output layer error  L_K = ŷ - y*
                L[-1] = output_t - target_oh

                # Eq. (18): L_l = W_{l+1}^T · diag(h'_{l+1}) · L_{l+1}
                for l in range(self.num_layers - 2, -1, -1):
                    jac = h_prime[l + 1].unsqueeze(-1) * self.linear_layers[l + 1].weight.unsqueeze(0)
                    L[l] = torch.einsum("bi,bij->bj", L[l + 1], jac)

                # Eq. (11): Δθ_l = (1/B) Σ_b L_l^t · e_l^{t,θ}
                for l, layer in enumerate(self.linear_layers):
                    grad_buf[l].add_(torch.einsum("bi,bij->ij", L[l], elig_trace[l]) / B)
                    if layer.bias is not None and elig_trace_bias[l] is not None:
                        grad_buf_bias[l].add_(torch.einsum("bi,bi->i", L[l], elig_trace_bias[l]) / B)
                pending += 1

            # -------------------------------------------------------------------
            # Update schedule: flush buffered gradients when the trigger fires
            # -------------------------------------------------------------------
            if self.update_last:
                do_update = t == T - 1
            elif self.update_every > 1:
                do_update = (t + 1) % self.update_every == 0
            else:
                do_update = True

            if do_update and pending > 0:
                for l, layer in enumerate(self.linear_layers):
                    g = grad_buf[l].clamp(-self.grad_clip, self.grad_clip) if self.grad_clip > 0.0 else grad_buf[l]
                    if self.use_optimizer and self.optimizer is not None:
                        layer.weight.grad = g.clone() if layer.weight.grad is None else layer.weight.grad.add_(g)
                    else:
                        layer.weight.add_(-self.lr * g)
                    grad_buf[l].zero_()

                    if layer.bias is not None and grad_buf_bias[l] is not None:
                        g_b = grad_buf_bias[l].clamp(-self.grad_clip, self.grad_clip) if self.grad_clip > 0.0 else grad_buf_bias[l]
                        if self.use_optimizer and self.optimizer is not None:
                            layer.bias.grad = g_b.clone() if layer.bias.grad is None else layer.bias.grad.add_(g_b)
                        else:
                            layer.bias.add_(-self.lr * g_b)
                        grad_buf_bias[l].zero_()

                pending = 0
                if self.use_optimizer and self.optimizer is not None:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            # Advance stored state to t for use at t+1
            for l in range(self.num_layers):
                prev_mem[l] = mem_rec[l]
                prev_spk[l] = spk_rec[l]
                prev_h_prime[l] = h_prime[l]

        # Flush any remaining gradients (non-divisible update_every windows)
        if pending > 0:
            for l, layer in enumerate(self.linear_layers):
                g = grad_buf[l].clamp(-self.grad_clip, self.grad_clip) if self.grad_clip > 0.0 else grad_buf[l]
                if self.use_optimizer and self.optimizer is not None:
                    layer.weight.grad = g.clone() if layer.weight.grad is None else layer.weight.grad.add_(g)
                else:
                    layer.weight.add_(-self.lr * g)

                if layer.bias is not None and grad_buf_bias[l] is not None:
                    g_b = grad_buf_bias[l].clamp(-self.grad_clip, self.grad_clip) if self.grad_clip > 0.0 else grad_buf_bias[l]
                    if self.use_optimizer and self.optimizer is not None:
                        layer.bias.grad = g_b.clone() if layer.bias.grad is None else layer.bias.grad.add_(g_b)
                    else:
                        layer.bias.add_(-self.lr * g_b)

            if self.use_optimizer and self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        if loss_count == 0:
            raise RuntimeError("OSTLTrainer: no supervised step was processed.")

        loss = total_loss / float(loss_count)
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
