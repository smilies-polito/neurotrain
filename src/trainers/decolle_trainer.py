"""
DECOLLE (Deep Continuous Local Learning) Trainer — unified version.

This version is annotated to highlight:
  - what matches the paper
  - what is an implementation choice
  - what explicitly diverges from the paper

Tag legend:
  [PAPER]            -> concept directly present in the paper
  [PAPER Eq.7/8/9]   -> tied to the main DECOLLE equations
  [PAPER Sec.X]      -> tied to a specific paper section
  [DIVERGENCE]       -> intentional or unavoidable mismatch vs. paper
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import snntorch as snn
except ImportError:
    snn = None


# ═══════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════

def _expand_param(value, n_layers: int, name: str) -> List[float]:
    """
    Expand scalar or sequence to a per-layer list of floats.

    [IMPLEMENTATION] Convenience utility only.
    """
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != n_layers:
            raise ValueError(f"{name}: expected length {n_layers}, got {len(value)}")
        return [float(v) for v in value]
    return [float(value)] * n_layers


def _to_float(x) -> float:
    """
    Extract a plain Python float from a scalar tensor or number.

    [IMPLEMENTATION] Utility only.
    """
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


# ═══════════════════════════════════════════════════════════════════════════
# _LayerInfo — lightweight descriptor for one (synapse, neuron) pair
# ═══════════════════════════════════════════════════════════════════════════

class _LayerInfo:
    """
    Metadata for one DECOLLE layer (synapse → neuron pair).

    [PAPER] DECOLLE is defined layerwise: each layer has its own local readout,
    local error, and local update.
    [IMPLEMENTATION] The paper does not define a Python metadata container.
    """

    __slots__ = (
        "synapse", "neuron", "kind", "is_recurrent",
        "beta", "threshold", "pool", "out_spatial",
    )

    def __init__(self, synapse, neuron, pool=None):
        self.synapse = synapse
        self.neuron = neuron
        self.pool = pool

        if isinstance(synapse, nn.Conv2d):
            self.kind = "conv"
        elif isinstance(synapse, nn.Linear):
            self.kind = "linear"
        elif hasattr(synapse, "conv") and isinstance(synapse.conv, nn.Conv2d):
            self.kind = "conv"
        elif hasattr(synapse, "fc") and isinstance(synapse.fc, nn.Linear):
            self.kind = "linear"
        else:
            raise TypeError(f"Unsupported synapse type: {type(synapse)}")

        self.is_recurrent = (snn is not None and isinstance(neuron, snn.RLeaky))

        self.beta = _to_float(neuron.beta)
        self.threshold = _to_float(neuron.threshold)

        self.out_spatial = None

    @property
    def n_output_features(self) -> int:
        syn = self.synapse
        if hasattr(syn, "out_channels"):
            return syn.out_channels
        if hasattr(syn, "out_features"):
            return syn.out_features
        if hasattr(syn, "conv") and isinstance(syn.conv, nn.Conv2d):
            return syn.conv.out_channels
        if hasattr(syn, "fc") and isinstance(syn.fc, nn.Linear):
            return syn.fc.out_features
        raise AttributeError(f"Cannot infer output features for synapse type {type(syn)}")

    @property
    def n_input_features(self) -> int:
        syn = self.synapse
        if hasattr(syn, "in_channels"):
            return syn.in_channels
        if hasattr(syn, "in_features"):
            return syn.in_features
        if hasattr(syn, "conv") and isinstance(syn.conv, nn.Conv2d):
            return syn.conv.in_channels
        if hasattr(syn, "fc") and isinstance(syn.fc, nn.Linear):
            return syn.fc.in_features
        raise AttributeError(f"Cannot infer input features for synapse type {type(syn)}")


# ═══════════════════════════════════════════════════════════════════════════
# DECOLLETrainer — main class
# ═══════════════════════════════════════════════════════════════════════════

class DECOLLETrainer(nn.Module):
    """
    Unified DECOLLE trainer for FCSNN / ConvSNN / RSNN.

    [PAPER] DECOLLE is defined as an external local-learning procedure:
    local readout, local loss, local error, local update.
    [IMPLEMENTATION] This trainer keeps DECOLLE logic outside the network.

    [DIVERGENCE — Optimizer] The paper (Sec. 2.3.4) uses AdaMax with
    β₁=0, β₂=0.95 and lr=1e-9. This implementation applies vanilla SGD
    (direct weight.data.add_) with no adaptive moment estimation. This
    significantly changes the optimization dynamics and may require
    different learning rates to reproduce paper results.

    [DIVERGENCE — Dropout] The paper (Table 2, Sec. 3.3) applies dropout
    (p=0.5) after every spiking layer, kept active even during testing.
    This implementation does not include dropout at any point.

    [DIVERGENCE — Refractory mechanism] The paper (Eq. 1, Eq. 4) includes
    an explicit refractory variable R with dynamics R[t+1] = γR[t] + (1-γ)S[t]
    and its contribution U = V - ρR + b. This implementation does not manage
    R explicitly; it is delegated entirely to the neuron model (e.g. snntorch).
    Whether R is present depends on the neuron module's own configuration.
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float = 1e-4,
        batch_size: int = 64,
        g_scale: float = 0.5,
        burn_in: int = 0,
        surrogate: str = "sigmoid",
        surrogate_scale: float = 5.0,
        delta: float = 0.5,
        h_with_noise: bool = False,
        omega_std: float = 0.5,
        lambda_u_upper: float = 0.0,
        lambda_u_lower: float = 0.0,
        lr_scale_per_layer: bool = False,
    ):
        super().__init__()

        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.burn_in = burn_in
        self.surrogate_kind = surrogate
        self.surrogate_scale = surrogate_scale
        self.lambda_u_upper = lambda_u_upper
        self.lambda_u_lower = lambda_u_lower
        self.lr_scale_per_layer = lr_scale_per_layer
        self.g_scale = g_scale
        self.h_with_noise = h_with_noise
        self.omega_std = omega_std

        self.n_classes: int = int(network.n_classes)

        self.layer_info: List[_LayerInfo] = self._detect_layers(network)
        self.n_layers = len(self.layer_info)

        self.delta = _expand_param(delta, self.n_layers, "delta")

        self.G = nn.ParameterList()
        self.H = nn.ParameterList()
        self._build_readout_matrices()

        self._traces_ready = False
        self._trace_bs: Optional[int] = None
        self.P: List[Optional[torch.Tensor]] = []
        self.P_rec: List[Optional[torch.Tensor]] = []
        self.S_prev: List[Optional[torch.Tensor]] = []

        self._diag: dict = {}

    def _get_syn_weight(self, syn):
        if hasattr(syn, "weight"):
            return syn.weight
        if hasattr(syn, "conv") and isinstance(syn.conv, nn.Conv2d):
            return syn.conv.weight
        if hasattr(syn, "fc") and isinstance(syn.fc, nn.Linear):
            return syn.fc.weight
        raise AttributeError(f"Cannot get weight from synapse type {type(syn)}")

    def _get_syn_bias(self, syn):
        if hasattr(syn, "bias"):
            return syn.bias
        if hasattr(syn, "conv") and isinstance(syn.conv, nn.Conv2d):
            return syn.conv.bias
        if hasattr(syn, "fc") and isinstance(syn.fc, nn.Linear):
            return syn.fc.bias
        return None

    def _get_conv_stride(self, syn):
        if hasattr(syn, "stride"):
            return syn.stride
        if hasattr(syn, "conv") and isinstance(syn.conv, nn.Conv2d):
            return syn.conv.stride
        raise AttributeError(f"Cannot get stride from conv synapse type {type(syn)}")

    def _get_conv_padding(self, syn):
        if hasattr(syn, "padding"):
            return syn.padding
        if hasattr(syn, "conv") and isinstance(syn.conv, nn.Conv2d):
            return syn.conv.padding
        raise AttributeError(f"Cannot get padding from conv synapse type {type(syn)}")

    def _get_conv_dilation(self, syn):
        if hasattr(syn, "dilation"):
            return syn.dilation
        if hasattr(syn, "conv") and isinstance(syn.conv, nn.Conv2d):
            return syn.conv.dilation
        return (1, 1)

    # ══════════════════════════════════════════════════════════════════════
    # Layer detection
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_layers(network: nn.Module) -> List[_LayerInfo]:
        """
        Walk network.layers and pair each trainable synapse with the following
        spiking neuron, optionally allowing a pooling module in between.

        [PAPER] DECOLLE is defined layerwise, with one local learning rule per
        spiking layer. In the DvsGesture architecture, the spiking nonlinearity
        is applied after pooling.
        [IMPLEMENTATION] This parser assumes a flat ModuleList with pattern:
            [synapse, optional pool, neuron, ...]
        """
        raw = list(network.layers)
        infos: List[_LayerInfo] = []
        i = 0

        POOL_TYPES = (
            nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
            nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
            nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
        )

        while i < len(raw):
            module = raw[i]

            is_synapse = (
                isinstance(module, (nn.Linear, nn.Conv2d))
                or (hasattr(module, "conv") and isinstance(module.conv, nn.Conv2d))
                or (hasattr(module, "fc") and isinstance(module.fc, nn.Linear))
            )

            if is_synapse:
                synapse = module
                pool = None
                i += 1

                if i < len(raw) and isinstance(raw[i], POOL_TYPES):
                    pool = raw[i]
                    i += 1

                if i >= len(raw):
                    raise ValueError("Network ends with a synapse but no neuron.")

                neuron = raw[i]

                if snn is not None and not isinstance(neuron, (snn.Leaky, snn.RLeaky)):
                    raise ValueError(
                        f"Expected snn.Leaky/RLeaky after synapse"
                        f"{' (+ optional pool)' if pool is not None else ''}, "
                        f"got {type(neuron)}"
                    )

                infos.append(_LayerInfo(synapse, neuron, pool))
                i += 1
            else:
                i += 1

        if not infos:
            raise ValueError("No (synapse, neuron) pairs found in network.layers.")
        return infos

    # ══════════════════════════════════════════════════════════════════════
    # Readout matrices G and H
    # ══════════════════════════════════════════════════════════════════════

    def _build_readout_matrices(self) -> None:
        """
        Create fixed random readout (G) and feedback (H) matrices.

        [PAPER Sec.2.3] Each layer has a fixed random readout G^l.
        [PAPER Sec.2.3.2] Feedback H may be G^T times sign-concordant noise.
        """
        for idx, info in enumerate(self.layer_info):
            n_post = info.n_output_features

            is_output = (idx == self.n_layers - 1) and (n_post == self.n_classes)

            if is_output:
                g = torch.eye(self.n_classes)
                h = g.clone()
            else:
                stdv = self.g_scale / math.sqrt(n_post)
                g = torch.empty(self.n_classes, n_post).uniform_(-stdv, stdv)

                if self.h_with_noise:
                    omega = torch.normal(
                        mean=torch.ones(n_post, self.n_classes),
                        std=self.omega_std,
                    )
                    omega = torch.clamp(omega, min=0)
                    h = g.t() * omega
                else:
                    h = g.t().clone()

            self.G.append(nn.Parameter(g, requires_grad=False))
            self.H.append(nn.Parameter(h, requires_grad=False))

    # ══════════════════════════════════════════════════════════════════════
    # Trace management
    # ══════════════════════════════════════════════════════════════════════

    def _ensure_traces(self, batch_size: int, device: torch.device) -> None:
        """
        Allocate trace buffers lazily.

        [PAPER] DECOLLE carries local state forward in time and avoids BPTT memory.
        [DIVERGENCE — Single trace] This trainer stores only one feedforward trace P
        and, optionally, P_rec, not the explicit P/Q pair of Eq. (4). The paper
        defines a two-stage trace:
            P[t+1] = α·P[t] + (1-α)·Q[t]
            Q[t+1] = β·Q[t] + (1-β)·S^{l-1}[t]
        which produces an alpha-function (difference of two exponentials) kernel.
        The single-trace approximation here (P ← β·P + S) collapses this into a
        single exponential, changing the temporal shape of the eligibility trace.
        """
        if self._traces_ready and self._trace_bs == batch_size:
            return

        self.P, self.P_rec, self.S_prev = [], [], []

        for info in self.layer_info:
            if info.kind == "conv":
                self.P.append(None)
            else:
                self.P.append(torch.zeros(batch_size, info.n_input_features, device=device))

            if info.is_recurrent:
                n_out = info.n_output_features
                self.P_rec.append(torch.zeros(batch_size, n_out, device=device))
                self.S_prev.append(torch.zeros(batch_size, n_out, device=device))
            else:
                self.P_rec.append(None)
                self.S_prev.append(None)

        self._traces_ready = True
        self._trace_bs = batch_size

    def _reset_traces(self) -> None:
        self._traces_ready = False
        self._trace_bs = None
        self.P, self.P_rec, self.S_prev = [], [], []

    # ══════════════════════════════════════════════════════════════════════
    # Surrogate gradient
    # ══════════════════════════════════════════════════════════════════════

    def _surrogate_grad(self, u_centered: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Compute σ'(U - threshold).

        [PAPER Eq.7/Eq.8] The update uses σ'(U_i^l).
        [DIVERGENCE — Surrogate function] The paper (Sec. 2.3.4) uses a piecewise
        linear surrogate whose derivative is the boxcar function:
            σ'(x) = 1 if x ∈ [-0.5, 0.5], else 0
        This implementation defaults to a sigmoid surrogate:
            σ'(x) = k · sigmoid(kx) · (1 - sigmoid(kx))
        The boxcar is available via surrogate="boxcar" with configurable delta.
        The paper notes that the surrogate shape is less critical in DECOLLE than
        in BPTT (footnote 1, Sec. 2.3.4), so this is unlikely to be a major issue.
        """
        if self.surrogate_kind == "sigmoid":
            sig = torch.sigmoid(self.surrogate_scale * u_centered)
            return self.surrogate_scale * sig * (1.0 - sig)
        else:
            d = self.delta[layer_idx]
            return ((u_centered >= -d) & (u_centered <= d)).float()

    # ══════════════════════════════════════════════════════════════════════
    # Weight and bias update primitives
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _update_linear(weight: torch.Tensor, mod: torch.Tensor,
                       p: torch.Tensor, lr: float, bs: int) -> float:
        """
        [DIVERGENCE — Optimizer] Applies vanilla SGD: W -= lr * dW.
        The paper (Sec. 2.3.4) uses AdaMax (β₁=0, β₂=0.95, lr=1e-9),
        which provides per-parameter adaptive scaling. This difference
        may require substantial learning rate tuning to match paper results.
        """
        dw = torch.einsum("bi,bj->ij", mod, p) / bs
        weight.data.add_(-lr * dw)
        return lr * dw.norm().item()

    @staticmethod
    def _update_conv(weight: torch.Tensor, mod_map: torch.Tensor,
                     p_map: torch.Tensor, lr: float, bs: int,
                     stride: Tuple[int, ...], padding: Tuple[int, ...],
                     dilation: Tuple[int, ...] = (1, 1)) -> float:
        """
        [DIVERGENCE — Optimizer] Same vanilla SGD note as _update_linear.
        """
        C_out, C_in, kH, kW = weight.shape

        col = F.unfold(
            p_map,
            kernel_size=(kH, kW),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        mod_flat = mod_map.flatten(2)

        if mod_flat.shape[-1] != col.shape[-1]:
            raise RuntimeError(
                "Conv DECOLLE update spatial mismatch: "
                f"mod positions={mod_flat.shape[-1]} vs unfold positions={col.shape[-1]}. "
                f"mod_map shape={tuple(mod_map.shape)}, "
                f"p_map shape={tuple(p_map.shape)}, "
                f"weight shape={tuple(weight.shape)}, "
                f"stride={stride}, padding={padding}, dilation={dilation}"
            )

        dw_flat = torch.einsum("bcl,bkl->ck", mod_flat, col) / bs
        dw = dw_flat.view(C_out, C_in, kH, kW)

        weight.data.add_(-lr * dw)
        return lr * dw.norm().item()

    @staticmethod
    def _update_linear_bias(bias: torch.Tensor, mod: torch.Tensor,
                            lr: float, bs: int) -> float:
        db = mod.sum(dim=0) / bs
        bias.data.add_(-lr * db)
        return lr * db.norm().item()

    @staticmethod
    def _update_conv_bias(bias: torch.Tensor, mod_map: torch.Tensor,
                          lr: float, bs: int) -> float:
        db = mod_map.sum(dim=(0, 2, 3)) / bs
        bias.data.add_(-lr * db)
        return lr * db.norm().item()

    # ══════════════════════════════════════════════════════════════════════
    # Local readout helper
    # ══════════════════════════════════════════════════════════════════════

    def _local_readout(self, spk_k: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Compute the local classifier output Y^l from the spike/state tensor of layer l.

        [PAPER Sec.2.3] Each layer has a fixed random readout Y^l = G^l S^l.
        [DIVERGENCE — GAP for conv layers] For conv feature maps, this implementation
        uses Global Average Pooling over spatial dimensions before applying G^l.
        The paper's DvsGesture architecture (Table 2) uses dense (fully connected)
        local classifiers that operate on the full flattened spatial feature map.
        GAP reduces the information available to the readout and changes the
        effective gradient signal propagated back to the conv weights.
        """
        if spk_k.dim() == 4:
            spk_flat = spk_k.mean(dim=(2, 3))
        else:
            spk_flat = spk_k

        return torch.matmul(spk_flat, self.G[layer_idx].t())

    # ══════════════════════════════════════════════════════════════════════
    # Main training loop
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on one batch of temporal spike data.

        [PAPER] DECOLLE performs updates at every simulation timestep using only
        local variables available at the same layer and current time.
        [DIVERGENCE] This implementation applies the closed-form update manually
        under no_grad, rather than constructing local AD subgraphs.
        [DIVERGENCE — Single trace] This implementation uses a reduced single-trace
        approximation instead of the explicit P/Q dynamics in Eq. (4).
        See _ensure_traces docstring for details.
        """
        T = data.shape[0]
        B = data.shape[1]
        device = data.device

        tgt = torch.zeros(B, self.n_classes, device=device)
        tgt.scatter_(1, target.unsqueeze(1), 1.0)

        self.network.reset()
        self._reset_traces()
        self._ensure_traces(B, device)

        # IMPORTANT:
        # Prediction is based on the last local DECOLLE readout, not on raw
        # spk_list[-1]. This is required for conv networks, where spk_list[-1]
        # is typically a feature map (B,C,H,W), not a class tensor.
        local_accum = torch.zeros(B, self.n_classes, device=device)

        total_loss = torch.tensor(0.0, device=device)
        n_loss = 0

        diag_spike = [0.0] * self.n_layers
        diag_surr = [0.0] * self.n_layers
        diag_dw = [0.0] * self.n_layers
        diag_db = [0.0] * self.n_layers
        diag_p = [0.0] * self.n_layers

        for t in range(T):
            x_t = data[t]
            spk_list, mem_list = self.network(x_t)

            do_update = (t >= self.burn_in)

            for k, info in enumerate(self.layer_info):
                spk_k = spk_list[k]
                mem_k = mem_list[k]

                # (a) Pre-synaptic activity
                if k == 0:
                    s_pre = x_t.reshape(B, -1) if info.kind == "linear" else x_t
                else:
                    prev_spk = spk_list[k - 1]
                    if info.kind == "linear" and prev_spk.dim() > 2:
                        s_pre = prev_spk.flatten(1)
                    else:
                        s_pre = prev_spk

                # (b) Reduced trace update
                # [DIVERGENCE — Trace dynamics] The paper (Eq. 4) defines:
                #     P[t+1] = α·P[t] + (1-α)·Q[t]
                #     Q[t+1] = β·Q[t] + (1-β)·S^{l-1}[t]
                # This code uses a single-pole approximation:
                #     P[t+1] = β·P[t] + S^{l-1}[t]
                # Two differences:
                #   1) The double-exponential (alpha-function) kernel is collapsed
                #      into a single exponential, losing the rise-then-decay shape.
                #   2) The (1-β) scaling factor on the input spike is missing.
                #      This changes the effective amplitude of the trace by a factor
                #      of ~1/(1-β), which can be partially compensated by the
                #      learning rate but alters the relative weighting of recent
                #      vs. older spikes.
                beta = info.beta
                if info.kind == "conv":
                    if self.P[k] is None:
                        self.P[k] = torch.zeros_like(s_pre)
                    self.P[k] = beta * self.P[k] + s_pre
                else:
                    self.P[k] = beta * self.P[k] + s_pre

                p_k = self.P[k]

                # (c) Optional recurrent trace
                if info.is_recurrent:
                    self.P_rec[k] = beta * self.P_rec[k] + self.S_prev[k]
                    self.S_prev[k] = spk_k.clone()

                diag_spike[k] += spk_k.mean().item() / T
                diag_p[k] += p_k.mean().item() / T

                # (d) Surrogate gradient
                # [DIVERGENCE — Surrogate default] See _surrogate_grad docstring.
                u_centered = mem_k - info.threshold
                g_k = self._surrogate_grad(u_centered, k)
                diag_surr[k] += g_k.mean().item() / T

                # Direct-output case only if the last detected layer is already class-sized
                is_output = (k == self.n_layers - 1) and (info.n_output_features == self.n_classes)

                # (e) Membrane regularization
                if mem_k.dim() == 4:
                    u_mean = mem_k.mean(dim=(0, 2, 3), keepdim=True)
                else:
                    u_mean = mem_k.mean(dim=0, keepdim=True)

                reg_upper_term = F.relu(u_mean + 0.01)
                reg_lower_term = F.relu(0.1 - u_mean)

                reg_upper = self.lambda_u_upper * reg_upper_term.mean()
                reg_lower = self.lambda_u_lower * reg_lower_term.mean()

                total_loss += reg_upper + reg_lower

                reg_grad_upper = self.lambda_u_upper * (u_mean > -0.01).float()
                reg_grad_lower = -self.lambda_u_lower * (u_mean < 0.1).float()
                reg_u_grad = (reg_grad_upper + reg_grad_lower).expand_as(mem_k)

                # (f) Local supervised modulation
                # [DIVERGENCE — Loss function] The paper (Sec. 2.3.4) states that
                # all experiments use smooth L1 loss uniformly across all layers.
                # This implementation uses:
                #   - Cross-entropy for the output layer (when n_output == n_classes)
                #   - MSE (0.5 * ||Y - Ŷ||²) for hidden layers
                # This mismatch affects gradient magnitudes and the shape of the
                # error surface, particularly at the output layer.
                if is_output:
                    logits = mem_k
                    total_loss += F.cross_entropy(logits, target)
                    n_loss += 1

                    # Accumulate output logits for prediction
                    local_accum += logits

                    if not do_update:
                        continue

                    supervised_mod = torch.softmax(logits, dim=1) - tgt
                    mod = supervised_mod + reg_u_grad

                else:
                    y_k = self._local_readout(spk_k, k)

                    # Use the last local readout as classifier output for prediction.
                    if k == self.n_layers - 1:
                        local_accum += y_k

                    delta_y = y_k - tgt

                    # Hidden local loss: MSE style
                    # [DIVERGENCE — Loss function] See note above.
                    total_loss += 0.5 * delta_y.pow(2).mean()
                    n_loss += 1

                    if not do_update:
                        continue

                    dldy = delta_y / float(self.n_classes)
                    err_k = torch.matmul(dldy, self.H[k].t())

                    if info.kind == "conv" and spk_k.dim() == 4:
                        err_spatial = err_k.unsqueeze(-1).unsqueeze(-1)
                        supervised_mod = err_spatial * g_k
                    else:
                        supervised_mod = err_k * g_k

                    mod = supervised_mod + reg_u_grad

                # (f-i) Effective learning rate
                layer_lr = self.lr * (10 ** k) if self.lr_scale_per_layer else self.lr

                # (f-ii) Weight and bias update
                # [DIVERGENCE — Optimizer] Vanilla SGD applied here.
                # See class docstring and _update_linear docstring for details.
                syn = info.synapse
                weight = self._get_syn_weight(syn)
                bias = self._get_syn_bias(syn)

                if info.kind == "conv":
                    stride = self._get_conv_stride(syn)
                    padding = self._get_conv_padding(syn)
                    dilation = self._get_conv_dilation(syn)

                    kH, kW = weight.shape[2], weight.shape[3]

                    col = F.unfold(
                        p_k,
                        kernel_size=(kH, kW),
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    )
                    expected_positions = col.shape[-1]
                    got_positions = mod.flatten(2).shape[-1]

                    mod_for_conv = mod

                    if got_positions != expected_positions:
                        H_in, W_in = p_k.shape[2], p_k.shape[3]

                        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
                        pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
                        dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)

                        H_out = (H_in + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
                        W_out = (W_in + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

                        if H_out * W_out != expected_positions:
                            raise RuntimeError(
                                f"Internal conv shape mismatch: expected_positions={expected_positions}, "
                                f"but computed H_out={H_out}, W_out={W_out}"
                            )

                        scale = expected_positions / float(got_positions)
                        mod_for_conv = F.interpolate(
                            mod, size=(H_out, W_out), mode="nearest"
                        ) / scale

                    dw_norm = self._update_conv(
                        weight, mod_for_conv, p_k, layer_lr, B,
                        stride=stride, padding=padding, dilation=dilation,
                    )
                    diag_dw[k] += dw_norm / T

                    if bias is not None:
                        db_norm = self._update_conv_bias(
                            bias, mod_for_conv, layer_lr, B
                        )
                        diag_db[k] += db_norm / T

                else:
                    dw_norm = self._update_linear(
                        weight, mod, p_k, layer_lr, B,
                    )
                    diag_dw[k] += dw_norm / T

                    if bias is not None:
                        db_norm = self._update_linear_bias(
                            bias, mod, layer_lr, B
                        )
                        diag_db[k] += db_norm / T

                # (g) Recurrent weight update
                if info.is_recurrent and self.P_rec[k] is not None:
                    rec_weight = info.neuron.recurrent.weight
                    mod_flat = mod.flatten(1) if mod.dim() > 2 else mod
                    self._update_linear(
                        rec_weight, mod_flat, self.P_rec[k], layer_lr, B,
                    )

        loss = total_loss / max(n_loss, 1)
        pred = local_accum.argmax(dim=1, keepdim=True)

        self._diag = {
            "spike_rates": diag_spike,
            "surr_grad_mean": diag_surr,
            "weight_update_norm": diag_dw,
            "bias_update_norm": diag_db,
            "p_trace_mean": diag_p,
        }

        return loss, pred

    # ══════════════════════════════════════════════════════════════════════
    # Inference
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Inference only.

        [PAPER-compatible] Forward dynamics only, no learning update.
        [IMPLEMENTATION] Prediction uses the accumulated local readout of the
        last detected DECOLLE layer, which is appropriate for conv networks
        where the final spiking layer is a feature map rather than class logits.
        """
        T, B = data.shape[0], data.shape[1]
        self.network.reset()

        local_accum = torch.zeros(B, self.n_classes, device=data.device)

        for t in range(T):
            spk_list, mem_list = self.network(data[t])

            last_idx = self.n_layers - 1
            last_info = self.layer_info[last_idx]
            spk_k = spk_list[last_idx]
            mem_k = mem_list[last_idx]

            is_output = (last_idx == self.n_layers - 1) and (last_info.n_output_features == self.n_classes)

            if is_output:
                local_accum += mem_k
            else:
                local_accum += self._local_readout(spk_k, last_idx)

        return local_accum.argmax(dim=1, keepdim=True)

    # ══════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════

    def get_diagnostics(self) -> dict:
        return dict(self._diag) if self._diag else {}

    def reset(self) -> None:
        self.network.reset()
        self._reset_traces()