"""
DECOLLE (Deep Continuous Local Learning) Trainer — unified version.

This version is annotated line by line to highlight:
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

# snnTorch is the neuron library used by the wrapped networks.
# [IMPLEMENTATION] The paper is not tied to snnTorch specifically.
# [DIVERGENCE] The paper derives DECOLLE for a leaky current-based I&F model
# with explicit membrane, synaptic, and refractory states; here the concrete
# neuron implementation is whatever snnTorch exposes in Leaky / RLeaky.
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
    [PAPER] The paper allows layerwise quantities (e.g. separate local readouts),
    but this helper itself is not from the paper.
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
    local error, and local update. This class is an implementation vehicle for
    that layer-local organization.
    [IMPLEMENTATION] The paper does not define a Python metadata container.
    """

    __slots__ = (
        "synapse", "neuron", "kind", "is_recurrent",
        "beta", "threshold", "pool", "out_spatial",
    )

    def __init__(self, synapse, neuron, pool=None):
        # Reference to the trainable feedforward synapse.
        # [PAPER] This corresponds to W^l in Eq. (7)/(8).
        self.synapse = synapse

        # Reference to the neuron object producing spikes/membrane states.
        # [PAPER] This corresponds to the spiking neuron in layer l.
        self.neuron = neuron

        # Optional pooling module between synapse and neuron.
        # [DIVERGENCE] Pooling is handled by the external wrapped network;
        # the paper's conv setup places spiking nonlinearity after pooling
        # in the DvsGesture architecture table.
        self.pool = pool

        # Classify synapse type.
        # [IMPLEMENTATION] Needed because Linear and Conv require different
        # trace shapes and different update formulas in code.
        if isinstance(synapse, nn.Conv2d):
            self.kind = "conv"
        elif isinstance(synapse, nn.Linear):
            self.kind = "linear"
        else:
            raise TypeError(f"Unsupported synapse type: {type(synapse)}")

        # Detect recurrence.
        # [DIVERGENCE] The paper focuses on feedforward SNN layers with internal
        # neural state recurrence over time; this trainer also supports explicit
        # recurrent weights through snn.RLeaky.
        self.is_recurrent = (snn is not None and isinstance(neuron, snn.RLeaky))

        # Cache neuron decay and threshold.
        # [PAPER] Threshold and decay are central to the membrane dynamics and
        # to the surrogate-gated update.
        # [DIVERGENCE] Here beta is taken from snnTorch's neuron parameter, not
        # from the paper's separate α, β, γ dynamics for P, Q, R.
        self.beta = _to_float(neuron.beta)
        self.threshold = _to_float(neuron.threshold)

        # Reserved field, currently unused.
        self.out_spatial = None

    @property
    def n_output_features(self) -> int:
        # Number of post-synaptic outputs.
        # [PAPER] Needed to size local random readouts G^l.
        syn = self.synapse
        return syn.out_channels if hasattr(syn, "out_channels") else syn.out_features

    @property
    def n_input_features(self) -> int:
        # Number of pre-synaptic inputs.
        # [PAPER] Needed to size the eligibility trace P^l_j in Eq. (7)/(8).
        syn = self.synapse
        return syn.in_channels if hasattr(syn, "in_channels") else syn.in_features


# ═══════════════════════════════════════════════════════════════════════════
# DECOLLETrainer — main class
# ═══════════════════════════════════════════════════════════════════════════

class DECOLLETrainer(nn.Module):
    """
    Unified DECOLLE trainer for FCSNN / ConvSNN / RSNN.

    [PAPER] DECOLLE is defined as an external local-learning procedure: local
    readout, local loss, local error, local update.
    [IMPLEMENTATION] This trainer keeps all DECOLLE logic outside the network.
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

        # Wrapped network.
        # [IMPLEMENTATION] The paper does not prescribe this wrapper API.
        self.network = network

        # Base learning rate η.
        # [PAPER Eq.7/Eq.8] η is explicit in the update rule.
        self.lr = lr

        # Batch size.
        # [PAPER Sec.2.3.4] The paper states DECOLLE is simulated in mini-batches.
        self.batch_size = batch_size

        # Burn-in timesteps before updates.
        # [PAPER Experiments] N-MNIST / DvsGesture use a burn-in period with
        # no update before starting weight changes.
        self.burn_in = burn_in

        # Surrogate gradient family.
        # [PAPER] A surrogate derivative is required.
        # [PAPER Sec.2.3.4] They used a piecewise linear surrogate whose
        # derivative is a boxcar.
        # [DIVERGENCE] This trainer also offers sigmoid.
        self.surrogate_kind = surrogate
        self.surrogate_scale = surrogate_scale

        # Regularization coefficients for Eq. (9).
        # [PAPER Eq.9] Two regularizers are added on U.
        self.lambda_u_upper = lambda_u_upper
        self.lambda_u_lower = lambda_u_lower

        # Optional heuristic learning-rate scaling by depth.
        # [DIVERGENCE] Not in the paper.
        self.lr_scale_per_layer = lr_scale_per_layer

        # Random readout scale.
        # [PAPER Sec.2.3 / Sec.2.3.4] Local readouts use fixed random weights
        # initialized uniformly.
        self.g_scale = g_scale

        # Feedback alignment options.
        # [PAPER Sec.2.3.2] Sign-concordant feedback alignment is discussed explicitly.
        self.h_with_noise = h_with_noise
        self.omega_std = omega_std

        # Number of output classes.
        # [IMPLEMENTATION] Needed to size one-hot targets and local classifiers.
        self.n_classes: int = int(network.n_classes)

        # Detect layerwise synapse→neuron pairs.
        # [PAPER] DECOLLE is layer-local; each layer has its own update.
        self.layer_info: List[_LayerInfo] = self._detect_layers(network)
        self.n_layers = len(self.layer_info)

        # Per-layer surrogate parameter.
        # [IMPLEMENTATION] Helper convenience.
        self.delta = _expand_param(delta, self.n_layers, "delta")

        # Fixed readout matrices G and feedback matrices H.
        # [PAPER Sec.2.3] Fixed random readouts G^l define the local classifiers.
        # [PAPER Sec.2.3.2] H may be sign-concordant noisy feedback.
        self.G = nn.ParameterList()
        self.H = nn.ParameterList()
        self._build_readout_matrices()

        # Trace state.
        # [PAPER] Learning uses local state variables carried forward in time.
        # [DIVERGENCE] This trainer keeps only a single P trace, not paper P and Q.
        self._traces_ready = False
        self._trace_bs: Optional[int] = None
        self.P: List[Optional[torch.Tensor]] = []
        self.P_rec: List[Optional[torch.Tensor]] = []
        self.S_prev: List[Optional[torch.Tensor]] = []

        # Diagnostics cache.
        # [IMPLEMENTATION] Not from the paper.
        self._diag: dict = {}

    # ══════════════════════════════════════════════════════════════════════
    # Layer detection
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_layers(network: nn.Module) -> List[_LayerInfo]:
        """
        Walk network.layers and pair each synapse with its neuron.

        [IMPLEMENTATION] Parser for a flat ModuleList.
        [PAPER] Conceptually corresponds to identifying DECOLLE layers.
        """
        raw = list(network.layers)
        infos: List[_LayerInfo] = []
        i = 0

        while i < len(raw):
            module = raw[i]

            # Detect trainable synapse.
            # [PAPER] Weight-bearing connection W^l.
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                synapse = module
                pool = None
                i += 1

                # Optional pooling.
                # [PAPER DvsGesture/Table 2] Pooling exists in conv architecture.
                # [DIVERGENCE] External parsing depends on the wrapped network.
                if i < len(raw) and isinstance(raw[i], (nn.MaxPool2d, nn.AvgPool2d)):
                    pool = raw[i]
                    i += 1

                if i >= len(raw):
                    raise ValueError("Network ends with a synapse but no neuron.")

                neuron = raw[i]

                # Check spiking neuron type.
                # [PAPER] Each DECOLLE layer ends in a spiking nonlinearity.
                if snn is not None and not isinstance(neuron, (snn.Leaky, snn.RLeaky)):
                    raise ValueError(f"Expected snn.Leaky/RLeaky after synapse, got {type(neuron)}")

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

            # Treat last layer with output dimension = n_classes as direct classifier.
            # [IMPLEMENTATION] This makes the output layer use identity readout.
            # [PAPER] The paper still frames DECOLLE through local readouts;
            # this identity shortcut is a practical specialization.
            is_output = (idx == self.n_layers - 1) and (n_post == self.n_classes)

            if is_output:
                # Identity readout at output layer.
                # [IMPLEMENTATION] Practical convenience.
                g = torch.eye(self.n_classes)
                h = g.clone()
            else:
                # Random fixed readout.
                # [PAPER Sec.2.3.4] Local readout weights G^l were initialized uniformly.
                stdv = self.g_scale / math.sqrt(n_post)
                g = torch.empty(self.n_classes, n_post).uniform_(-stdv, stdv)

                if self.h_with_noise:
                    # Sign-concordant noisy feedback.
                    # [PAPER Sec.2.3.2] H_ij = G^T_ij * ω_ij, with ω ~ N(1, 1/2),
                    # clamped at zero for sign concordance.
                    omega = torch.normal(
                        mean=torch.ones(n_post, self.n_classes),
                        std=self.omega_std,
                    )
                    omega = torch.clamp(omega, min=0)
                    h = g.t() * omega
                else:
                    # Exact transpose.
                    # [PAPER] This corresponds to exact local gradient through the readout.
                    h = g.t().clone()

            # Register as fixed non-trainable tensors.
            # [PAPER] G is fixed.
            self.G.append(nn.Parameter(g, requires_grad=False))
            self.H.append(nn.Parameter(h, requires_grad=False))

    # ══════════════════════════════════════════════════════════════════════
    # Trace management
    # ══════════════════════════════════════════════════════════════════════

    def _ensure_traces(self, batch_size: int, device: torch.device) -> None:
        """
        Allocate trace buffers lazily.

        [PAPER] DECOLLE carries local state forward in time and avoids BPTT memory.
        [DIVERGENCE] The paper uses P and Q traces, plus refractory R in dynamics.
        This trainer stores only one feedforward trace P and, optionally, P_rec.
        """
        if self._traces_ready and self._trace_bs == batch_size:
            return

        self.P, self.P_rec, self.S_prev = [], [], []

        for info in self.layer_info:
            if info.kind == "conv":
                # Conv P trace needs runtime spatial size.
                # [IMPLEMENTATION]
                self.P.append(None)
            else:
                # Linear P trace.
                # [PAPER Eq.7/Eq.8] P_j is the pre-synaptic eligibility trace.
                self.P.append(torch.zeros(batch_size, info.synapse.in_features, device=device))

            if info.is_recurrent:
                # Recurrent trace for explicit recurrent weights.
                # [DIVERGENCE] Extension beyond the core paper derivation.
                n_out = info.n_output_features
                self.P_rec.append(torch.zeros(batch_size, n_out, device=device))
                self.S_prev.append(torch.zeros(batch_size, n_out, device=device))
            else:
                self.P_rec.append(None)
                self.S_prev.append(None)

        self._traces_ready = True
        self._trace_bs = batch_size

    def _reset_traces(self) -> None:
        """
        Reset all trainer trace state.

        [PAPER] Temporal state should not leak across independent samples.
        """
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
        [PAPER Sec.2.3.4] The paper uses a piecewise linear surrogate whose
        derivative is the boxcar.
        [DIVERGENCE] This trainer also offers a sigmoid-derivative surrogate.
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
        Linear weight update:
            dW[i,j] = (1/B) * Σ_b mod[b,i] * P[b,j]

        [PAPER Eq.7/Eq.8] This is exactly the batch-averaged outer-product form
        of the local three-factor rule.
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
        Conv2d weight update via im2col.

        The conv DECOLLE update requires the modulation map and the unfolded
        pre-synaptic trace to refer to the same number of spatial output
        positions. If they do not, raise a detailed error.
        """
        C_out, C_in, kH, kW = weight.shape

        col = F.unfold(
            p_map,
            kernel_size=(kH, kW),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )  # (B, C_in*kH*kW, L)

        mod_flat = mod_map.flatten(2)  # (B, C_out, L)

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
        """
        Linear bias update:
            db[i] = (1/B) * Σ_b mod[b,i]

        [PAPER Sec.2.3.4] Biases were used and trained in all DECOLLE layers.
        [PAPER-compatible] For a bias, ∂U/∂b = 1, so only the post-synaptic
        modulation remains.
        """
        db = mod.sum(dim=0) / bs
        bias.data.add_(-lr * db)
        return lr * db.norm().item()

    @staticmethod
    def _update_conv_bias(bias: torch.Tensor, mod_map: torch.Tensor,
                          lr: float, bs: int) -> float:
        """
        Conv2d bias update:
            db[c] = (1/B) * Σ_{b,h,w} mod[b,c,h,w]

        [PAPER Sec.2.3.4] Biases were trained.
        [PAPER-compatible] Same bias logic as linear, with spatial summation
        because the conv bias is shared over positions.
        """
        db = mod_map.sum(dim=(0, 2, 3)) / bs
        bias.data.add_(-lr * db)
        return lr * db.norm().item()

    # ══════════════════════════════════════════════════════════════════════
    # Main training loop
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on one batch of temporal spike data.

        [PAPER] Updates are made at every simulation time step.
        [PAPER Sec.2.3.4] The paper explicitly states per-timestep updates.
        [DIVERGENCE] This implementation performs the updates manually under
        no_grad instead of using local autodiff subgraphs.
        """
        T = data.shape[0]
        B = data.shape[1]
        device = data.device

        # One-hot targets.
        # [IMPLEMENTATION] Needed for hidden-layer local classifier losses.
        tgt = torch.zeros(B, self.n_classes, device=device)
        tgt.scatter_(1, target.unsqueeze(1), 1.0)

        # Reset neural and trace state.
        # [PAPER] State is temporal but sample-local.
        self.network.reset()
        self._reset_traces()
        self._ensure_traces(B, device)

        # Accumulate output spikes for final prediction.
        # [PAPER Experiments] Classification is often read out by spike counts.
        spk_accum = torch.zeros(B, self.n_classes, device=device)

        # Running loss accumulator.
        # [IMPLEMENTATION] Monitoring only; not used by autograd.
        total_loss = torch.tensor(0.0, device=device)
        n_loss = 0

        # Diagnostics.
        # [IMPLEMENTATION]
        diag_spike = [0.0] * self.n_layers
        diag_surr = [0.0] * self.n_layers
        diag_dw = [0.0] * self.n_layers
        diag_db = [0.0] * self.n_layers
        diag_p = [0.0] * self.n_layers

        # Time loop.
        # [PAPER] Online continual learning over timesteps.
        for t in range(T):
            x_t = data[t]

            # Forward one timestep through the network.
            # [PAPER] Only current-time and local states are needed for the update.
            spk_list, mem_list = self.network(x_t)

            # Accumulate output spikes.
            spk_accum += spk_list[-1]

            # Burn-in gate.
            # [PAPER Experiments] No update during burn-in, but dynamics continue.
            do_update = (t >= self.burn_in)

            # Layer loop.
            # [PAPER] Every layer has its own local readout/loss/update.
            for k, info in enumerate(self.layer_info):
                spk_k = spk_list[k]
                mem_k = mem_list[k]

                # ──────────────────────────────────────────────────────
                # Step (a): Determine pre-synaptic input s_pre
                # ──────────────────────────────────────────────────────
                #
                # [PAPER Eq.7/Eq.8] P_j^l is driven by the pre-synaptic activity.
                if k == 0:
                    s_pre = x_t.reshape(B, -1) if info.kind == "linear" else x_t
                else:
                    prev_spk = spk_list[k - 1]
                    if info.kind == "linear" and prev_spk.dim() > 2:
                        # Conv→linear flattening.
                        # [IMPLEMENTATION]
                        s_pre = prev_spk.flatten(1)
                    else:
                        s_pre = prev_spk

                # ──────────────────────────────────────────────────────
                # Step (b): Update feedforward eligibility trace P
                # ──────────────────────────────────────────────────────
                #
                # [PAPER Eq.3/Eq.4] The original model uses Q and P with separate
                # synaptic and membrane time constants.
                # [DIVERGENCE] Here we keep a single decayed trace:
                #       P[t] = beta * P[t-1] + s_pre[t]
                # rather than:
                #       Q[t+1] = beta Q[t] + (1-beta) S_pre[t]
                #       P[t+1] = alpha P[t] + (1-alpha) Q[t]
                beta = info.beta
                if info.kind == "conv":
                    if self.P[k] is None:
                        self.P[k] = torch.zeros_like(s_pre)
                    self.P[k] = beta * self.P[k] + s_pre
                else:
                    self.P[k] = beta * self.P[k] + s_pre

                p_k = self.P[k]

                # ──────────────────────────────────────────────────────
                # Step (c): Update recurrent trace P_rec (RLeaky only)
                # ──────────────────────────────────────────────────────
                #
                # [DIVERGENCE] Explicit recurrent weight learning is an extension
                # beyond the paper's main feedforward DECOLLE derivation.
                if info.is_recurrent:
                    self.P_rec[k] = beta * self.P_rec[k] + self.S_prev[k]
                    self.S_prev[k] = spk_k.clone()

                # Diagnostics.
                diag_spike[k] += spk_k.mean().item() / T
                diag_p[k] += p_k.mean().item() / T

                # ──────────────────────────────────────────────────────
                # Step (d): Surrogate gradient
                # ──────────────────────────────────────────────────────
                #
                # [PAPER Eq.7/Eq.8] σ'(U_i^l) gates the spike-based part of the update.
                u_centered = mem_k - info.threshold
                g_k = self._surrogate_grad(u_centered, k)
                diag_surr[k] += g_k.mean().item() / T

                # Output-layer detection.
                # [IMPLEMENTATION] Used to switch between output loss and hidden local loss.
                is_output = (k == self.n_layers - 1) and (info.n_output_features == self.n_classes)

                # ══════════════════════════════════════════════════════
                # Step (e): Paper Eq. 9 regularization on membrane U
                # ══════════════════════════════════════════════════════
                #
                # [PAPER Eq.9] L_g = Σ_l L^l + λ1<[U+0.01]_+> + λ2[0.1-<U>]_+
                # [PAPER Sec.2.3.4] These terms prevent sustained firing and silence.
                #
                # IMPORTANT:
                # These regularizers are functions of U directly, not of spikes S.
                # Therefore they contribute directly as dL_reg/dU, without σ'(U).

                if info.kind == "conv" and mem_k.dim() == 4:
                    # Number of non-batch elements per sample.
                    # [IMPLEMENTATION]
                    N_elem = float(mem_k.shape[1] * mem_k.shape[2] * mem_k.shape[3])

                    # Per-sample mean membrane.
                    # [PAPER Eq.9] The lower regularizer is about the layer-average membrane.
                    mem_mean_per_sample = mem_k.flatten(1).mean(dim=1)

                    # Upper regularizer.
                    # [PAPER Eq.9] λ1<[U+0.01]_+>
                    reg_upper = self.lambda_u_upper * F.relu(mem_k + 0.01).mean()

                    # Lower regularizer.
                    # [PAPER Eq.9] λ2[0.1-<U>]_+
                    reg_lower = self.lambda_u_lower * F.relu(0.1 - mem_mean_per_sample).mean()

                    # Gradient of the upper regularizer wrt U.
                    # [PAPER-consistent] Direct dL/dU term.
                    reg_grad_upper = (
                        self.lambda_u_upper
                        * (mem_k + 0.01 > 0).float()
                        / N_elem
                    )

                    # Gradient of the lower regularizer wrt U.
                    # [PAPER-consistent] Direct dL/dU term, broadcast over all units.
                    lower_active = (0.1 - mem_mean_per_sample > 0).float().view(B, 1, 1, 1)
                    reg_grad_lower = (
                        -self.lambda_u_lower
                        * lower_active
                        / N_elem
                    )
                else:
                    # Linear layer case.
                    N_elem = float(mem_k.shape[1])
                    mem_mean_per_sample = mem_k.mean(dim=1)

                    reg_upper = self.lambda_u_upper * F.relu(mem_k + 0.01).mean()
                    reg_lower = self.lambda_u_lower * F.relu(0.1 - mem_mean_per_sample).mean()

                    reg_grad_upper = (
                        self.lambda_u_upper
                        * (mem_k + 0.01 > 0).float()
                        / N_elem
                    )

                    lower_active = (0.1 - mem_mean_per_sample > 0).float().unsqueeze(1)
                    reg_grad_lower = (
                        -self.lambda_u_lower
                        * lower_active
                        / N_elem
                    )

                # Add regularization loss for monitoring.
                # [PAPER Eq.9]
                total_loss += reg_upper + reg_lower

                # Direct membrane-gradient contribution.
                # [PAPER Eq.9-compatible]
                reg_u_grad = reg_grad_upper + reg_grad_lower

                # ══════════════════════════════════════════════════════
                # Step (f): Compute local supervised term + total modulation
                # ══════════════════════════════════════════════════════

                if is_output:
                    # Output layer loss on membrane logits.
                    # [DIVERGENCE] The paper's main derivation emphasizes local readouts
                    # and MSE-like local losses; here the output layer uses CE on logits.
                    logits = mem_k
                    total_loss += F.cross_entropy(logits, target)
                    n_loss += 1

                    if not do_update:
                        continue

                    # CE gradient wrt logits.
                    # [IMPLEMENTATION] This is already dL/dU for the output membrane.
                    supervised_mod = torch.softmax(logits, dim=1) - tgt

                    # Total modulation = supervised term + direct membrane regularizer term.
                    # [PAPER-compatible] Both are local at this layer and current time.
                    mod = supervised_mod + reg_u_grad

                else:
                    # Hidden layer local readout.
                    # [PAPER Sec.2.3] Y^l = G^l S^l
                    if info.kind == "conv" and spk_k.dim() == 4:
                        # [DIVERGENCE] GAP before readout instead of full flatten.
                        # The paper's conv implementation uses dense readouts on flattened maps.
                        spk_pooled = spk_k.mean(dim=(2, 3))
                    else:
                        spk_pooled = spk_k

                    g_mat = self.G[k]
                    h_mat = self.H[k]

                    # Local classifier output.
                    # [PAPER Sec.2.3] Fixed random readout Y^l = G^l S^l
                    y_k = torch.matmul(spk_pooled, g_mat.t())

                    # Readout error.
                    # [PAPER Eq.8] Y^l - Yhat^l
                    delta_y = y_k - tgt

                    # Hidden local loss.
                    # [PAPER Eq.8] Special case shown for MSE.
                    total_loss += 0.5 * delta_y.pow(2).mean()
                    n_loss += 1

                    if not do_update:
                        continue

                    # dL/dY for the local classifier.
                    # [PAPER Eq.8] proportional to Y - Yhat
                    dldy = delta_y / float(self.n_classes)

                    # Backpropagate through the fixed local readout.
                    # [PAPER Sec.2.3.2] This uses H, either exact transpose or sign-concordant variant.
                    err_k = torch.matmul(dldy, h_mat.t())

                    # Spike-based supervised term.
                    # [PAPER Eq.7/Eq.8] error_i * σ'(U_i)
                    if info.kind == "conv" and spk_k.dim() == 4:
                        # Broadcast channelwise classifier error over spatial positions.
                        # [DIVERGENCE] Consequence of using GAP readout.
                        err_spatial = err_k.unsqueeze(-1).unsqueeze(-1)
                        supervised_mod = err_spatial * g_k
                    else:
                        supervised_mod = err_k * g_k

                    # Combine spike-based supervised term with direct membrane regularizer term.
                    mod = supervised_mod + reg_u_grad

                # ──────────────────────────────────────────────────────
                # Step (f-i): Effective learning rate
                # ──────────────────────────────────────────────────────
                #
                # [DIVERGENCE] Optional depth-based LR scaling is not in the paper.
                layer_lr = self.lr * (10 ** k) if self.lr_scale_per_layer else self.lr

                # ══════════════════════════════════════════════════════
                # Step (f-ii): Feedforward weight and bias update
                # ══════════════════════════════════════════════════════

                if info.kind == "conv":
                    syn = info.synapse
                    mod_for_conv = mod

                    if mod.dim() != 4:
                        raise RuntimeError(
                            f"Expected 4D modulation for conv layer, got shape {tuple(mod.shape)}"
                        )

                    # Compute the conv-output spatial grid expected by the weight update.
                    # This is derived directly from p_k and the conv geometry, so it
                    # remains correct even if pooling is not explicitly exposed in
                    # network.layers.
                    kH, kW = syn.weight.shape[2], syn.weight.shape[3]

                    col = F.unfold(
                        p_k,
                        kernel_size=(kH, kW),
                        stride=syn.stride,
                        padding=syn.padding,
                        dilation=syn.dilation,
                    )
                    expected_positions = col.shape[-1]
                    got_positions = mod.flatten(2).shape[-1]

                    if got_positions != expected_positions:
                        H_in, W_in = p_k.shape[2], p_k.shape[3]

                        stride_h, stride_w = (
                            syn.stride if isinstance(syn.stride, tuple)
                            else (syn.stride, syn.stride)
                        )
                        pad_h, pad_w = (
                            syn.padding if isinstance(syn.padding, tuple)
                            else (syn.padding, syn.padding)
                        )
                        dil_h, dil_w = (
                            syn.dilation if isinstance(syn.dilation, tuple)
                            else (syn.dilation, syn.dilation)
                        )

                        H_out = (H_in + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
                        W_out = (W_in + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

                        if H_out * W_out != expected_positions:
                            raise RuntimeError(
                                f"Internal conv shape mismatch: expected_positions={expected_positions}, "
                                f"but computed H_out={H_out}, W_out={W_out}"
                            )

                        # Resample the modulation map to the conv-output grid.
                        # The division keeps the overall scale roughly stable when
                        # broadcasting a pooled modulation back to a finer grid.
                        scale = expected_positions / float(got_positions)
                        mod_for_conv = F.interpolate(
                            mod, size=(H_out, W_out), mode="nearest"
                        ) / scale

                    dw_norm = self._update_conv(
                        syn.weight, mod_for_conv, p_k, layer_lr, B,
                        stride=syn.stride, padding=syn.padding, dilation=syn.dilation,
                    )

                    if syn.bias is not None:
                        db_norm = self._update_conv_bias(
                            syn.bias, mod_for_conv, layer_lr, B
                        )
                        diag_db[k] += db_norm / T

                else:
                    syn = info.synapse

                    dw_norm = self._update_linear(
                        syn.weight, mod, p_k, layer_lr, B,
                    )

                    if syn.bias is not None:
                        db_norm = self._update_linear_bias(
                            syn.bias, mod, layer_lr, B
                        )
                        diag_db[k] += db_norm / T

                diag_dw[k] += dw_norm / T

                # ══════════════════════════════════════════════════════
                # Step (g): Recurrent weight update (RLeaky only)
                # ══════════════════════════════════════════════════════
                #
                # [DIVERGENCE] Extension beyond the core paper formulation.
                if info.is_recurrent and self.P_rec[k] is not None:
                    rec_weight = info.neuron.recurrent.weight
                    mod_flat = mod.flatten(1) if mod.dim() > 2 else mod
                    self._update_linear(
                        rec_weight, mod_flat, self.P_rec[k], layer_lr, B,
                    )

        # Average scalar loss for reporting.
        # [IMPLEMENTATION]
        loss = total_loss / max(n_loss, 1)

        # Final prediction by output spike count.
        # [PAPER Experiments] This matches the spike-count decoding style used in DvsGesture.
        pred = spk_accum.argmax(dim=1, keepdim=True)

        # Save diagnostics.
        # [IMPLEMENTATION]
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
        """
        T, B = data.shape[0], data.shape[1]
        self.network.reset()

        spk_accum = torch.zeros(B, self.n_classes, device=data.device)
        for t in range(T):
            spk_list, _ = self.network(data[t])
            spk_accum += spk_list[-1]

        return spk_accum.argmax(dim=1, keepdim=True)

    # ══════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════

    def get_diagnostics(self) -> dict:
        """
        Return per-layer diagnostics.

        [IMPLEMENTATION] Extra debugging/inspection utility.
        """
        return dict(self._diag) if self._diag else {}

    def reset(self) -> None:
        """
        Reset network and trainer state.

        [PAPER-compatible] Clears temporal state between runs.
        """
        self.network.reset()
        self._reset_traces()