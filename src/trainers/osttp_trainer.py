"""
OSTTP (Online Spatio-Temporal Learning with Target Projection) trainer.

Implements the algorithm from:
  Ortner et al., "Online Spatio-Temporal Learning with Target Projection",
  IEEE AICAS 2023.

OSTTP combines OSTL eligibility traces with DRTP learning signals to achieve
fully online, forward-only training of spiking neural networks:

    Δθ_l ≈ Σ_t  L_t^l · e_{t,θ_l}^l          (Eq. 6)

  - Output layer:  L_t^K = dE_t / dy_t^K       (analytic gradient)
  - Hidden layers:  L_t^l = B_l · y*_t          (Eq. 11, random projection)

  - Eligibility traces (Eqs. 8-9):
    ε_t = (ds_t/ds_{t-1}) · ε_{t-1} + ∂s_t/∂θ + (ds_t/dy_{t-1}) · e_{t-1}
    e_t = (∂y_t/∂s_t) · ε_t + ∂y_t/∂θ

Supported network patterns:
  - Spiking layers: nn.Linear → snn.Leaky / snn.RLeaky
  - Output readout: last-layer membrane ("mem"), separate nn.Linear ("logits")
  - Reset mechanisms: "zero" (full), "subtract" (soft), "none"

Neuron dynamics (snntorch with reset_delay=True):
  Leaky, reset='zero':      s_t = β(1-y_{t-1})s_{t-1} + I_t
  Leaky, reset='subtract':  s_t = βs_{t-1} - y_{t-1}·b + I_t
  Leaky, reset='none':      s_t = βs_{t-1} + I_t
  RLeaky adds:              + y_{t-1} @ H  (recurrent, ungated by reset)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from networks.base_snn import BaseSNN


# ---------------------------------------------------------------------------
# Layer metadata
# ---------------------------------------------------------------------------

@dataclass
class _LayerSpec:
    """Metadata for one (Linear → spiking neuron) pair."""
    synapse: nn.Linear
    neuron: nn.Module          # snn.Leaky or snn.RLeaky
    n_in: int
    n_out: int
    reset: str                 # "zero" | "subtract" | "none"
    rec_weight: Optional[nn.Parameter] = None   # RLeaky recurrent weight
    rec_bias: Optional[nn.Parameter] = None
    thresh_param: Optional[nn.Parameter] = None  # trainable threshold


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class OSTTPTrainer(BaseTrainer):
    """
    Online Spatio-Temporal Learning with Target Projection (OSTTP).

    Args:
        network:          nn.Module with Linear→Leaky/RLeaky layers.
        lr:               Learning rate (used when use_optimizer=False).
        pseudo:           Pseudo-derivative: "tanh" or "fast_sigmoid".
        output_loss:      "ce" (cross-entropy) or "mse".
        output_readout:   "mem" (last spiking layer membrane) or
                          "logits" (separate non-spiking Linear readout).
        feedback_scale:   Scaling factor for random feedback matrices B_l.
        feedback_seed:    RNG seed for reproducible B_l generation.
        grad_clip:        If >0, clamp gradient magnitudes.
        use_optimizer:    If True, accumulate .grad and call optimizer.step().
        optimizer:        Optional external optimizer (else Adam is created).
    """

    def __init__(
        self,
        network: BaseSNN,
        lr: float = 1e-3,
        pseudo: str = "tanh",
        output_loss: str = "ce",
        output_readout: str = "mem",
        feedback_scale: float = 1.0,
        feedback_seed: int = 42,
        grad_clip: float = 0.0,
        use_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ):
        super().__init__()

        # @TODO: are these controls needed?
        assert pseudo in ("tanh", "fast_sigmoid"), f"Unknown pseudo: {pseudo}"
        assert output_loss in ("ce", "mse"), f"Unknown output_loss: {output_loss}"
        assert output_readout in ("mem", "logits"), f"Unknown output_readout: {output_readout}"

        self.network = network
        self.lr = float(lr)
        self.pseudo = pseudo
        self.output_loss = output_loss
        self.output_readout = output_readout
        self.feedback_scale = float(feedback_scale)
        self.feedback_seed = int(feedback_seed)
        self.grad_clip = float(grad_clip)
        self.use_optimizer = bool(use_optimizer)

        # --- Discover layers ---
        self.layers: List[_LayerSpec] = self._discover_layers()
        assert self.layers, "No Linear→Leaky/RLeaky pairs found in network."

        # --- Output readout ---
        self.output_synapse: Optional[nn.Linear] = None
        if output_readout == "logits":
            self.output_synapse = self._find_output_linear()

        self.output_size = (
            self.output_synapse.out_features
            if self.output_synapse is not None
            else self.layers[-1].n_out
        )

        # --- Feedback matrices (created lazily on first train_sample) ---
        self._n_hidden_fb = (
            len(self.layers) - 1 if output_readout == "mem" else len(self.layers)
        )
        self._fb_names: List[str] = []
        self._fb_ready = False

        # --- Optimizer ---
        if use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(
                network.parameters(), lr=self.lr
            )
        else:
            self.optimizer = None

        # --- Forward hooks to capture per-layer tensors ---
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._h_input: List[Optional[torch.Tensor]] = [None] * len(self.layers)
        self._h_spk: List[Optional[torch.Tensor]] = [None] * len(self.layers)
        self._h_mem: List[Optional[torch.Tensor]] = [None] * len(self.layers)
        self._h_out_in: Optional[torch.Tensor] = None   # input to output linear
        self._h_out_logits: Optional[torch.Tensor] = None
        self._install_hooks()

    # -----------------------------------------------------------------------
    # Layer discovery
    # -----------------------------------------------------------------------

    def _discover_layers(self) -> List[_LayerSpec]:
        """Find adjacent (nn.Linear, snn.Leaky/RLeaky) pairs."""
        pairs: List[Tuple[nn.Linear, nn.Module]] = []
        seen = set()

        def _try(syn, neu):
            if not isinstance(syn, nn.Linear):
                return
            if not isinstance(neu, (snn.Leaky, snn.RLeaky)):
                return
            key = (id(syn), id(neu))
            if key in seen:
                return
            seen.add(key)
            pairs.append((syn, neu))

        # Scan containers for adjacent children
        for _, container in self.network.named_modules():
            children = list(container.children())
            for i in range(len(children) - 1):
                _try(children[i], children[i + 1])

        specs = []
        for syn, neu in pairs:
            if syn.bias is not None and syn.bias.requires_grad:
                raise NotImplementedError(
                    "Trainable bias on spiking-layer Linear is not supported."
                )
            if getattr(neu, "reset_delay", True) is False:
                raise NotImplementedError(
                    "reset_delay=False is not supported (equations assume y_{t-1})."
                )

            reset = str(getattr(neu, "reset_mechanism", "subtract"))
            assert reset in ("zero", "subtract", "none"), f"Bad reset: {reset}"

            rec_w = rec_b = None
            rec = getattr(neu, "recurrent", None)
            if rec is not None:
                assert hasattr(rec, "weight"), "RLeaky must use all_to_all=True."
                rec_w = rec.weight
                rec_b = getattr(rec, "bias", None)

            thresh_p = None
            th = getattr(neu, "threshold", None)
            if isinstance(th, nn.Parameter) and th.requires_grad:
                thresh_p = th

            specs.append(_LayerSpec(
                synapse=syn, neuron=neu,
                n_in=syn.in_features, n_out=syn.out_features,
                reset=reset,
                rec_weight=rec_w, rec_bias=rec_b,
                thresh_param=thresh_p,
            ))
        return specs

    def _find_output_linear(self) -> nn.Linear:
        """Find the last nn.Linear not already used as a spiking synapse."""
        used = {id(s.synapse) for s in self.layers}
        last = None
        for _, m in self.network.named_modules():
            if isinstance(m, nn.Linear) and id(m) not in used:
                last = m
        if last is None:
            raise ValueError("output_readout='logits' requires a non-spiking Linear.")
        return last

    # -----------------------------------------------------------------------
    # Forward hooks
    # -----------------------------------------------------------------------

    def _install_hooks(self):
        for idx, spec in enumerate(self.layers):
            self._hooks.append(
                spec.synapse.register_forward_hook(self._syn_hook(idx))
            )
            self._hooks.append(
                spec.neuron.register_forward_hook(self._neu_hook(idx))
            )
        if self.output_synapse is not None:
            self._hooks.append(
                self.output_synapse.register_forward_hook(self._out_hook())
            )

    def _syn_hook(self, idx):
        def hook(_, inp, out):
            if inp and torch.is_tensor(inp[0]):
                self._h_input[idx] = inp[0].detach()
        return hook

    def _neu_hook(self, idx):
        def hook(_, _inp, out):
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                self._h_spk[idx] = out[0].detach()
                self._h_mem[idx] = out[1].detach()
        return hook

    def _out_hook(self):
        def hook(_, inp, out):
            if inp and torch.is_tensor(inp[0]):
                self._h_out_in = inp[0].detach()
            if torch.is_tensor(out):
                self._h_out_logits = out.detach()
        return hook

    def _clear_hooks(self):
        for i in range(len(self.layers)):
            self._h_input[i] = self._h_spk[i] = self._h_mem[i] = None
        self._h_out_in = self._h_out_logits = None

    # -----------------------------------------------------------------------
    # Feedback matrices  B_l  (fixed random, Eq. 11)
    # -----------------------------------------------------------------------

    def _ensure_feedback(self, target_dim: int, device, dtype):
        """
        Create random DRTP matrices B_l for hidden layers if not already done and instantiate as buffers.
         - target_dim: dimension of y*_t (number of classes)
         - device, dtype: match the network parameters
        The matrix sizes are (target_dim, n_out) for each hidden layer, where n_out is the layer's number of neurons.
        """
        # @TODO: do we actually need this flag?
        if self._fb_ready:
            return
        # Create a random number generator for reproducible feedback matrices
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.feedback_seed)
        # Scale feedback by 1/sqrt(fan_out) to keep magnitudes reasonable
        std = self.feedback_scale / math.sqrt(max(target_dim, 1))

        for idx in range(self._n_hidden_fb):
            n_out = self.layers[idx].n_out
            B = torch.empty(target_dim, n_out, dtype=dtype)
            B.normal_(mean=0.0, std=std, generator=gen)
            name = f"_fb_{idx}"
            self.register_buffer(name, B.to(device), persistent=True)
            self._fb_names.append(name)

        self._fb_ready = True

    def _get_fb(self, idx: int) -> torch.Tensor:
        return getattr(self, self._fb_names[idx])

    # -----------------------------------------------------------------------
    # Pseudo-derivative  ψ(x)
    # -----------------------------------------------------------------------

    def _psi(self, x: torch.Tensor) -> torch.Tensor:
        if self.pseudo == "tanh":
            return 1.0 - torch.tanh(x).pow(2)
        # fast_sigmoid: 1 / (100|x| + 1)^2
        return 1.0 / (100.0 * x.abs() + 1.0).pow(2)

    # -----------------------------------------------------------------------
    # Output loss / learning signal
    # -----------------------------------------------------------------------

    def _loss_value(self, y_out, y_star):
        if self.output_loss == "ce":
            return -(y_star * F.log_softmax(y_out, dim=1)).sum(1).mean()
        return F.mse_loss(y_out, y_star)

    def _output_signal(self, y_out, y_star):
        """L_t^K = dE_t / dy_t^K  (analytic)."""
        if self.output_loss == "ce":
            return torch.softmax(y_out, dim=1) - y_star   # (B, C)
        return y_out - y_star  # MSE gradient

    # -----------------------------------------------------------------------
    # Target preparation
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_targets(target, n_classes, T, device, dtype):
        """Convert target to one-hot [T, B, C]."""
        if target.dim() == 1:
            y = torch.zeros(target.size(0), n_classes, device=device, dtype=dtype)
            y.scatter_(1, target.unsqueeze(1), 1.0)
            return y.unsqueeze(0).expand(T, -1, -1)
        if target.dim() == 2:
            return target.to(device=device, dtype=dtype).unsqueeze(0).expand(T, -1, -1)
        if target.dim() == 3:
            if target.size(0) == T:
                return target.to(device=device, dtype=dtype)
            if target.size(1) == T:
                return target.transpose(0, 1).to(device=device, dtype=dtype)
        raise ValueError(f"Unsupported target shape {target.shape}")

    # -----------------------------------------------------------------------
    # Accumulate or apply gradient
    # -----------------------------------------------------------------------

    def _apply_grad(self, param: nn.Parameter, grad: torch.Tensor):
        if self.grad_clip > 0:
            grad = grad.clamp(-self.grad_clip, self.grad_clip)
        if self.use_optimizer:
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.add_(grad)
        else:
            param.data.sub_(self.lr * grad)

    # -----------------------------------------------------------------------
    # Expand scalar/vector to (B, n_out)
    # -----------------------------------------------------------------------

    @staticmethod
    def _expand(val, B, n, device, dtype):
        if isinstance(val, torch.Tensor):
            t = val.detach().to(device=device, dtype=dtype)
        else:
            t = torch.tensor(float(val), device=device, dtype=dtype)
        if t.numel() == 1:
            return t.view(1, 1).expand(B, n)
        return t.flatten().unsqueeze(0).expand(B, n)

    # -----------------------------------------------------------------------
    # Core: train one temporal sample
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        One OSTTP training step on a temporal batch.

        Args:
            data:   (T, B, n_input)
            target: (B,) class indices  or  (B, C)  or  (T, B, C)

        Returns:
            loss:   scalar (time-averaged)
            pred:   (B, 1) argmax predictions
        """
        assert data.dim() >= 3, f"Expected (T, B, ...), got {data.shape}"

        T, B = data.shape[0], data.shape[1]
        device, dtype = data.device, data.dtype
        n_layers = len(self.layers)

        # Turn targets into one-hot [T, B, C]
        y_star = self._make_targets(target, self.output_size, T, device, dtype)
        # Generate random matrices for DRTP
        self._ensure_feedback(self.output_size, device, dtype)

        # Reset network state
        self.network.reset()
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # --- Allocate eligibility states per layer ---
        # WEIGHT TRACES
        eps_w = []   # (B, n_out, n_in)
        e_w = []
        dW = []      # accumulated weight gradient

        # RECURRENT WEIGHTS TRACES (RLeaky only)
        eps_h = []   # (B, n_out, n_out) for RLeaky
        e_h = []
        dH = []

        # BIAS TRACES   
        eps_b = []   # (B, n_out) for trainable threshold
        e_b = []
        db = []

        prev_spk = []
        prev_mem = []

        # Initialization for all data structures
        for spec in self.layers:
            n_i, n_o = spec.n_in, spec.n_out
            eps_w.append(torch.zeros(B, n_o, n_i, device=device, dtype=dtype))
            e_w.append(torch.zeros(B, n_o, n_i, device=device, dtype=dtype))
            dW.append(torch.zeros(n_o, n_i, device=device, dtype=dtype))

            if spec.rec_weight is not None and spec.rec_weight.requires_grad:
                eps_h.append(torch.zeros(B, n_o, n_o, device=device, dtype=dtype))
                e_h.append(torch.zeros(B, n_o, n_o, device=device, dtype=dtype))
                dH.append(torch.zeros(n_o, n_o, device=device, dtype=dtype))
            else:
                eps_h.append(None)
                e_h.append(None)
                dH.append(None)

            if spec.thresh_param is not None:
                eps_b.append(torch.zeros(B, n_o, device=device, dtype=dtype))
                e_b.append(torch.zeros(B, n_o, device=device, dtype=dtype))
                db.append(torch.zeros(n_o, device=device, dtype=dtype))
            else:
                eps_b.append(None)
                e_b.append(None)
                db.append(None)

            prev_spk.append(torch.zeros(B, n_o, device=device, dtype=dtype))
            prev_mem.append(torch.zeros(B, n_o, device=device, dtype=dtype))

        # Output readout accumulator
        dW_out = None
        db_out = None
        if self.output_synapse is not None:
            dW_out = torch.zeros_like(self.output_synapse.weight)
            if self.output_synapse.bias is not None:
                db_out = torch.zeros_like(self.output_synapse.bias)

        readout_sum = torch.zeros(B, self.output_size, device=device, dtype=dtype)
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        # ===================================================================
        # Time loop
        # ===================================================================
        for t in range(T):
            # Reset hook storage
            self._clear_hooks()
            # FORWARD PASS
            self.network(data[t])

            # --- Collect per-layer tensors from hooks ---
            x = [self._h_input[i] for i in range(n_layers)]
            spk = [self._h_spk[i] for i in range(n_layers)]
            mem = [self._h_mem[i] for i in range(n_layers)]

            if any(v is None for v in x + spk + mem):
                raise RuntimeError(
                    "Hooks failed to capture all layer tensors. "
                    "Check that your network uses Linear→Leaky/RLeaky pairs."
                )

            # --- Output readout ---
            if self.output_readout == "mem":
                y_out = mem[-1]
            else:
                if self._h_out_logits is not None:
                    y_out = self._h_out_logits
                else:
                    raise RuntimeError("output_readout='logits' but hook missed.")

            # --- Learning signals ---
            y_star_t = y_star[t]
            # Calculate the output learning signal used for the output layer only
            l_out = self._output_signal(y_out, y_star_t)     # (B, C)
            # Calculate the loss
            total_loss += self._loss_value(y_out, y_star_t)
            readout_sum += y_out

            fb_idx = 0  # counter for hidden feedback matrices

            # ---------------------------------------------------------------
            # Per-layer eligibility update
            # ---------------------------------------------------------------
            for i, spec in enumerate(self.layers):
                x_t = x[i]                 # (B, n_in)  presynaptic input
                s_t = mem[i]               # (B, n_out) current membrane
                y_prev = prev_spk[i]       # (B, n_out) previous spike
                s_prev = prev_mem[i]       # (B, n_out) previous membrane

                # Extract data and create the data structures
                beta = self._expand(
                    getattr(spec.neuron, "beta", 1.0), B, spec.n_out, device, dtype
                )
                thresh = self._expand(
                    getattr(spec.neuron, "threshold", 1.0), B, spec.n_out, device, dtype
                )
                
                # Calcola il gradiente della funzione di attivazione (pseudo-derivata) rispetto alla membrana s_t
                psi = self._psi(s_t - thresh)              # (B, n_out)
                # Extract recurrent weights
                H = spec.rec_weight.detach() if spec.rec_weight is not None else None

                # ----- Jacobians (see docstring for equations) -----
                # How much past membrane potential influences current:
                # j_s  = ds_t/ds_{t-1}         (B, n_out)  diagonal
                # How much past output spike of the neuron influence current membrane potential:
                # J_y  = ds_t/dy_{t-1}         diag or full matrix
                # How much bias/threshold influences current membrane potential:
                # ds_db = ∂s_t/∂b              (B, n_out)

                if spec.reset == "zero":
                    # s_t = β(1-y_{t-1})s_{t-1} + I_t [+ y_{t-1}@H]
                    j_s = beta * (1.0 - y_prev)
                    diag_y = -beta * s_prev         # diagonal of ds/dy from reset
                    ds_db = torch.zeros_like(y_prev)  # b doesn't appear in s

                elif spec.reset == "subtract":
                    # s_t = βs_{t-1} - y_{t-1}·b + I_t [+ y_{t-1}@H]
                    j_s = beta
                    diag_y = -thresh                # ds/dy from -y·b term
                    ds_db = -y_prev                 # ∂s/∂b = -y_{t-1}

                else:  # "none"
                    # s_t = βs_{t-1} + I_t [+ y_{t-1}@H]
                    j_s = beta
                    diag_y = torch.zeros_like(beta)  # no dependence on y
                    ds_db = torch.zeros_like(y_prev)

                # ----- Combine J_y · e_prev for each parameter -----
                # For Leaky (no recurrence): J_y is diagonal → element-wise
                # For RLeaky: J_y = H + diag(diag_y) → full matrix multiply

                # If we have no recurrent weights we consider only the weigths and biases
                # We multiply the jacobian for the corresponding traces
                if H is None:
                    # Leaky: J_y is diagonal = diag_y
                    Jy_ew = diag_y.unsqueeze(-1) * e_w[i]   # (B, n_out, n_in)
                    Jy_eh = None
                    Jy_eb = diag_y * e_b[i] if e_b[i] is not None else None
                # If we have recurrent weights we consider also the contribution of the previous spikes to the current state
                else:
                    # RLeaky: J_y = H + diag(diag_y)         (n_out, n_out)
                    # Expand H to batch: (B, n_out, n_out)
                    J_y = H.unsqueeze(0).expand(B, -1, -1) + torch.diag_embed(diag_y)
                    Jy_ew = torch.bmm(J_y, e_w[i])
                    Jy_eh = torch.bmm(J_y, e_h[i]) if e_h[i] is not None else None
                    Jy_eb = (
                        torch.bmm(J_y, e_b[i].unsqueeze(-1)).squeeze(-1)
                        if e_b[i] is not None else None
                    )

                # ----- Eligibility recursion (Eq. 8-9) -----

                # --- W (input weights) ---
                # ∂s_t/∂W[j→i] = x_t[j] for neuron i
                ds_dW = x_t.unsqueeze(1).expand(-1, spec.n_out, -1)   # (B, n_out, n_in)

                eps_w[i] = j_s.unsqueeze(-1) * eps_w[i] + ds_dW + Jy_ew

                # Output layer with "mem" readout: dy/ds = 1 (no spike nonlinearity)
                is_output_mem = (
                    self.output_readout == "mem" and i == n_layers - 1
                )
                dy_ds = torch.ones_like(psi) if is_output_mem else psi
                dy_db = torch.zeros_like(psi) if is_output_mem else -psi

                e_w[i] = dy_ds.unsqueeze(-1) * eps_w[i]               # (B, n_out, n_in)

                # --- H (recurrent weights, RLeaky only) ---
                if eps_h[i] is not None:
                    # ∂s_t/∂H[j→i] = y_{t-1}[j] for neuron i
                    ds_dH = y_prev.unsqueeze(1).expand(-1, spec.n_out, -1)
                    eps_h[i] = j_s.unsqueeze(-1) * eps_h[i] + ds_dH + Jy_eh
                    e_h[i] = dy_ds.unsqueeze(-1) * eps_h[i]

                # --- b (threshold) ---
                if eps_b[i] is not None:
                    Jy_eb_val = Jy_eb if Jy_eb is not None else diag_y * e_b[i]
                    eps_b[i] = j_s * eps_b[i] + ds_db + Jy_eb_val
                    e_b[i] = dy_ds * eps_b[i] + dy_db

                # ----- Learning signal (Eq. 11) -----
                if is_output_mem:
                    # Last spiking layer IS the output → use analytic gradient
                    L = l_out                                # (B, C)
                else:
                    # Hidden layer → random projection of target
                    L = torch.matmul(y_star_t, self._get_fb(fb_idx))  # (B, n_out)
                    fb_idx += 1

                # ----- Accumulate Δθ (Eq. 6) -----
                dW[i] += (L.unsqueeze(-1) * e_w[i]).sum(0)   # (n_out, n_in)
                if dH[i] is not None:
                    dH[i] += (L.unsqueeze(-1) * e_h[i]).sum(0)
                if db[i] is not None:
                    db[i] += (L * e_b[i]).sum(0)

            # --- Non-spiking output layer ---
            if dW_out is not None and self.output_synapse is not None:
                out_in = self._h_out_in
                if out_in is None:
                    raise RuntimeError("Hook missed output linear input.")
                # e_out = dy/ds · x  (identity activation → dy/ds = 1)
                e_out = out_in.unsqueeze(1).expand(-1, self.output_size, -1)
                dW_out += (l_out.unsqueeze(-1) * e_out).sum(0)
                if db_out is not None:
                    db_out += (l_out).sum(0)

            # Save state for next timestep
            for i in range(n_layers):
                prev_spk[i] = spk[i]
                prev_mem[i] = mem[i]

        # ===================================================================
        # Apply accumulated gradients
        # ===================================================================
        denom = float(max(B, 1))

        # For each layer
        for i, spec in enumerate(self.layers):
            self._apply_grad(spec.synapse.weight, dW[i] / denom)
            if dH[i] is not None and spec.rec_weight is not None:
                self._apply_grad(spec.rec_weight, dH[i] / denom)
            if db[i] is not None and spec.thresh_param is not None:
                grad_b = db[i] / denom
                # Reduce to match parameter shape (scalar or per-neuron)
                if spec.thresh_param.numel() == 1:
                    grad_b = grad_b.sum().view_as(spec.thresh_param)
                else:
                    grad_b = grad_b.view_as(spec.thresh_param)
                self._apply_grad(spec.thresh_param, grad_b)

        if dW_out is not None and self.output_synapse is not None:
            self._apply_grad(self.output_synapse.weight, dW_out / denom)
        if db_out is not None and self.output_synapse is not None:
            self._apply_grad(self.output_synapse.bias, db_out / denom)

        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        loss = total_loss / float(max(T, 1))
        pred = readout_sum.argmax(dim=1, keepdim=True)
        return loss.detach(), pred

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def reset(self):
        self.network.reset()
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def checkpoint_state(self) -> dict:
        mats = [getattr(self, n).cpu() for n in self._fb_names] if self._fb_ready else []
        return {"feedback": mats}

    def load_checkpoint_state(self, state: dict):
        fb = state.get("feedback", [])
        if not fb:
            return
        dev = self.layers[0].synapse.weight.device
        dt = self.layers[0].synapse.weight.dtype
        self._ensure_feedback(fb[0].shape[0], dev, dt)
        for name, mat in zip(self._fb_names, fb):
            getattr(self, name).copy_(mat.to(device=dev, dtype=dt))

    def to(self, device):
        super().to(device)
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self