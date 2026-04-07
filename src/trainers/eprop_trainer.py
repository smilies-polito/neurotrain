"""
E-prop (Eligibility Propagation) trainer for snnTorch-based RSNN networks.

Implements the e-prop learning algorithm for recurrent spiking neural networks
(RSNNs) using snnTorch neurons, following:

    G. Bellec et al., "A solution to the learning dilemma for recurrent networks
    of spiking neurons," Nature Communications, vol. 11, no. 3625, 2020.
    DOI: 10.1038/s41467-020-17236-y

    Reference TensorFlow implementation:
    https://github.com/IGITUGraz/eligibility_propagation

This module implements the *classification* variant of e-prop (Eq. 29) for
LIF neurons only (Eqs. 6–7). ALIF eligibility traces (Eqs. 24–25) are
included as commented-out code blocks, marked with [ALIF].

Network interface contract (RSNN):
───────────────────────────────────────────────────────────────────────
  The trainer wraps an RSNN network that exposes:
    - forward(x) → (spk_rec, mem_rec)   single-timestep forward pass
        spk_rec[0]: recurrent spikes z_j^t  [batch, n_rec]  (Eq. 7)
        mem_rec[0]: recurrent membrane v_j^t [batch, n_rec]  (Eq. 6)
        spk_rec[1], mem_rec[1]: spiking output layer (IGNORED by e-prop)
    - input_layers[0].weight             W^in  [n_rec, n_in]
    - recurrent_layers[0].recurrent.weight   W^rec [n_rec, n_rec]
    - fc_out.weight                      W^out [n_out, n_rec]
    - reset(device=...)                  reset all internal states
    - is_recurrent = True
    - in_shape, recurrent_dim, n_classes

  The trainer owns the analog readout (Eq. 11) and does NOT use the
  network's spiking output layer. This keeps the network general-purpose
  while the trainer implements e-prop-specific readout dynamics.
───────────────────────────────────────────────────────────────────────

Equation map (paper → code):
──────────────────────────────────────────────────────────────────────
  Paper Eq.  │ Description                        │ Code variable
─────────────┼────────────────────────────────────┼──────────────────
  Eq. 6      │ LIF membrane dynamics              │ v_t  (from network)
  Eq. 7      │ Spike generation z_j^t = H(v-θ)   │ z_t  (from network)
  Eq. 8–10   │ ALIF threshold adaptation          │ [ALIF] blocks
  Eq. 11     │ Leaky analog readout               │ vo   (in trainer)
  Eq. 12     │ Low-pass filter F_α                │ x_in_bar, z_bar_prev
  Eq. 13–14  │ Eligibility vector recursion       │ ε via x_in_bar, z_bar
  Eq. 22     │ LIF ε_{ji} = F_α(z_i^t)           │ z_bar_prev
  Eq. 23     │ LIF e_{ji} = ψ_j · F_α(z_i)      │ e_in, e_rec
  Eq. 24     │ ALIF slow ε_{ji,a}                │ [ALIF] block
  Eq. 25     │ ALIF e_{ji} with β term            │ [ALIF] block
  Eq. 28     │ Kappa-smoothed traces F_κ(e)       │ trace_in/rec/out
  Eq. 29     │ Classification weight update       │ weight update block
  Eq. 4      │ Learning signal L_j^t              │ L_t
  Methods    │ Pseudo-derivative ψ_j^t            │ _surrogate_gradient()
──────────────────────────────────────────────────────────────────────

Where spikes z_t are used in the e-prop computation:
──────────────────────────────────────────────────────────────────────
  z_t (binary spikes from Eq. 7) feeds into three places:
    1. Analog readout (Eq. 11):  vo = κ·vo + z_t @ W_out^T + b_out
    2. Recurrent eligibility vector (Eq. 22):  z_bar_prev = F_α(z^{t-1})
       → z_t is added to z_bar AFTER computing e_rec at this timestep,
         so e_rec uses z^{t-1} as required by the paper.
    3. Output trace (Eq. 28):  trace_out = F_κ(z_t)
──────────────────────────────────────────────────────────────────────

Known deviations from the paper:
──────────────────────────────────────────────────────────────────────
  D1. No refractory period. The paper (Methods, below Eq. 7) specifies
      z_j^t = 0 for 2–5 ms after a spike, with ψ = 0 during that period.
      snnTorch's RLeaky does NOT implement refractoriness — the neuron
      can fire on consecutive timesteps. The pseudo-derivative naturally
      goes toward zero after a subtract-reset (since |v - v_th| is large),
      but this is not an exact refractory mask.

  D2. Eq. 6 coherence depends on snnTorch RLeaky internals. Verified:
      snnTorch RLeaky with reset_mechanism="subtract", reset_delay=False
      implements v^{t+1} = α·v^t + W_rec·z^t + W_in·x^{t+1} - z^t·v_th,
      which matches Eq. 6.

  D3. update_last / update_every are implementation extensions not in the
      paper. When active, the weight update deviates from the standard
      Eq. 29 which sums over ALL timesteps. Traces are always accumulated
      regardless, so the gradient direction is preserved but magnitude and
      timing differ. Documented inline.

  D4. Output weight gradient (Supp. Note 3). The paper derives
      ΔW^out_{kj} in Supplementary Note 3. Our implementation
      w_out_grad = err^T @ F_κ(z) follows the same structure as
      the TU Graz reference code (dloss_dw_out).
──────────────────────────────────────────────────────────────────────
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class EpropTrainer(BaseTrainer):
    """
    E-prop trainer for RSNN networks, implementing online weight updates.

    Currently supports:
      - LIF neurons (Eqs. 6–7) with eligibility traces from Eq. 22–23
      - Classification via cross-entropy (Eq. 29)
      - Symmetric e-prop (B_{jk} = W^out_{kj})

    Does NOT yet support:
      - ALIF neurons (Eqs. 8–10) with slow eligibility traces (Eqs. 24–25)
      - Random e-prop or adaptive e-prop feedback
      - Reward-based e-prop (Eq. 5, 36–37)
    """

    # Reference values from Bellec et al. 2020 (Methods section).
    # These define the e-prop algorithm's parameters independently of
    # the network's own neuron parameters. The trainer overrides the
    # network's threshold/decay for e-prop computations to ensure
    # algorithmic consistency with the paper.
    PAPER_DEFAULTS = {
        "dt":      1.0,       # ms — simulation timestep
        "tau_m":   20.0,      # ms — membrane time constant (Eq. 6)
        "tau_out": 30.0,      # ms — readout time constant (Eq. 11)
        "v_th":    0.03,      # threshold voltage (Methods; TU Graz code)
        "gamma":   0.3,       # pseudo-derivative dampening (Methods)
        # [ALIF]
        "tau_a":   200.0,     # ms — adaptation time constant (Eq. 10)
        "beta":    0.07,      # adaptation strength (Eq. 8)
    }

    def __init__(
        self,
        network,
        lr: float,
        batch_size: int,
        gamma: float = None,
        dt: float = None,
        tau_mem: float = None,
        tau_out: float = None,
        threshold: float = None,
        lr_layer_norm: tuple = (1.0, 1.0, 1.0),
        quant: bool = False,
        use_optimizer: bool = True,
        optimizer=None,
        update_last: bool = False,
        update_every: int = 1,
        seq_batch_size: int = 1,
        **kwargs,
    ):
        """
        Initialize e-prop trainer.

        All e-prop algorithm parameters default to the paper's reference values
        (PAPER_DEFAULTS) if not explicitly provided. The network is treated as
        trainer-agnostic — its own beta/threshold are used only for forward
        dynamics, while the trainer uses its own α, κ, v_th for eligibility
        traces and pseudo-derivatives.

        Args:
            network:    RSNN network instance.
            lr:         Learning rate η.
            batch_size: Batch size.
            gamma:      Pseudo-derivative dampening γ_pd (default: 0.3, Methods).
            dt:         Simulation timestep δt in ms (default: 1.0, Methods).
            tau_mem:    Membrane time constant τ_m in ms (default: 20.0, Eq. 6).
            tau_out:    Readout time constant τ_out in ms (default: 30.0, Eq. 11).
            threshold:  Firing threshold v_th (default: 0.03, Methods).
            lr_layer_norm: Per-layer learning rate modulation (not in paper).
            use_optimizer: If True, apply e-prop gradients through an optimizer (Adam by default). If False, apply pure online synaptic updates directly to the weight tensors.
            optimizer:  Optional optimizer instance to use when use_optimizer=True.
            update_last:  If True, only update weights at last timestep [D3].
            update_every: Update weights every N timesteps [D3].
        """
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.quant = quant
        self.use_optimizer = use_optimizer
        self.update_last = update_last
        self.update_every = update_every
        self.seq_batch_size = seq_batch_size

        # ── Validate network ──
        if not (hasattr(network, "is_recurrent") and network.is_recurrent):
            raise TypeError(
                "EpropTrainer requires a recurrent RSNN; "
                f"got network={type(network).__name__}."
            )
        if hasattr(network, "hidden_size") and len(network.hidden_size) != 1:
            raise ValueError(
                "EpropTrainer currently supports only single-layer recurrent RSNNs."
            )

        # ── E-prop algorithm parameters ──
        # Default to PAPER_DEFAULTS; user can override any of them.
        # These are the trainer's own parameters, independent of the network.
        P = self.PAPER_DEFAULTS
        _dt      = dt        if dt        is not None else P["dt"]
        _tau_mem = tau_mem    if tau_mem   is not None else P["tau_m"]
        _tau_out = tau_out    if tau_out   is not None else P["tau_out"]

        # α = exp(-δt/τ_m) — membrane decay for eligibility vectors (Eq. 6, 22)
        self.alpha = math.exp(-_dt / _tau_mem)
        # κ = exp(-δt/τ_out) — readout decay for trace smoothing (Eq. 11, 28)
        self.kappa = math.exp(-_dt / _tau_out)
        # v_th — threshold for pseudo-derivative (Methods, below Eq. 9)
        self.threshold = threshold if threshold is not None else P["v_th"]
        # γ_pd — dampening factor for pseudo-derivative (Methods, below Eq. 9)
        self.gamma = gamma if gamma is not None else P["gamma"]

        # [ALIF] Would also need:
        # _tau_a = tau_a if tau_a is not None else P["tau_a"]
        # self.rho = math.exp(-_dt / _tau_a)  # ρ, Eq. 10
        # self.beta = P["beta"]                # β, Eq. 8

        self.lr_layer = lr_layer_norm

        # ── Loss function for monitoring ──
        # [Point 1] The paper uses cross-entropy E = -Σ_{t,k} π*_k log π_k
        # for classification. The e-prop weight update (Eq. 29) uses the
        # gradient (π_k - π*_k), which is implemented directly — the loss
        # function here is only for monitoring/reporting, NOT for computing
        # gradients. We use cross-entropy to match the paper's loss definition.
        self.loss_fn = nn.CrossEntropyLoss()

        # ── Resolve weight references from RSNN ──
        self._w_in = network.input_layers[0].weight          # [n_rec, n_in]

        rlif = network.recurrent_layers[0]
        if hasattr(rlif, "recurrent") and hasattr(rlif.recurrent, "weight"):
            self._w_rec = rlif.recurrent.weight               # [n_rec, n_rec]
        else:
            raise AttributeError(
                f"Cannot find recurrent weight in {type(rlif).__name__}. "
                f"Expected .recurrent.weight (snnTorch RLeaky)."
            )

        self._w_out = network.fc_out.weight                   # [n_out, n_rec]

        # [Point 2] Output bias b_k^out from Eq. 11.
        # If the network's fc_out has a bias, use it; otherwise default to 0.
        if network.fc_out.bias is not None:
            self._b_out = network.fc_out.bias                 # [n_out]
        else:
            self._b_out = None  # No bias — simplification noted in [D] section

        # ── Optimizer / update mode ──
        # If use_optimizer is False, weight updates are applied directly
        # as pure online e-prop, i.e. W ← W - η·grad/batch_size per timestep.
        # If use_optimizer is True, the computed gradients are injected into
        # an optimizer step (Adam by default), which yields a hybrid training
        # recipe rather than the paper's pure online rule.
        self._external_optimizer = optimizer
        self.use_optimizer = use_optimizer
        if use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(network.parameters(), lr=lr)
        else:
            self.optimizer = None

    # ══════════════════════════════════════════════════════════════════════
    #  Single-timestep forward pass for e-prop
    # ══════════════════════════════════════════════════════════════════════

    def _eprop_step(self, x: torch.Tensor, vo: torch.Tensor):
        """
        Run one timestep: recurrent dynamics via network, analog readout in trainer.

        1. Calls network.forward(x) → recurrent spikes z_t and membrane v_t
           (Eqs. 6–7). [Point 4: verified that snnTorch RLeaky with
           reset_mechanism="subtract" and reset_delay=False matches Eq. 6.]
           The network's spiking output layer also executes but is IGNORED.

        2. Computes analog readout (Eq. 11):
              y_k^t = κ · y_k^{t-1} + Σ_j W^out_{kj} · z_j^t + b_k^out

        Args:
            x:  Input [batch, n_in] (flattened)
            vo: Previous analog output membrane [batch, n_out]

        Returns:
            z_t: Recurrent spikes z_j^t ∈ {0,1} [batch, n_rec]  (Eq. 7)
            v_t: Recurrent membrane v_j^t [batch, n_rec]          (Eq. 6)
            vo:  Updated analog readout y_k^t [batch, n_out]      (Eq. 11)
        """
        # Reshape flat input to network's expected in_shape if spatial
        in_shape = self.network.in_shape
        if len(in_shape) > 1:
            x_net = x.view(x.shape[0], *in_shape)
        else:
            x_net = x

        # [Point 4] Recurrent dynamics (Eqs. 6–7)
        # snnTorch RLeaky (reset_mechanism="subtract", reset_delay=False):
        #   v_j^{t+1} = α·v_j^t + W_rec·z^t + W_in·x^{t+1} - z_j^t·v_th
        #   z_j^{t+1} = H(v_j^{t+1} - v_th)
        # This matches Eq. 6–7. Note: NO refractory period [D1].
        spk_rec, mem_rec = self.network.forward(x_net)

        z_t = spk_rec[0]   # spikes z_j^t ∈ {0,1}  [batch, n_rec]
        v_t = mem_rec[0]   # membrane v_j^t          [batch, n_rec]

        # [Point 2] Analog readout (Eq. 11):
        #   y_k^t = κ · y_k^{t-1} + Σ_j W^out_{kj} · z_j^t + b_k^out
        vo_new = self.kappa * vo + z_t @ self._w_out.t()
        if self._b_out is not None:
            vo_new = vo_new + self._b_out

        return z_t, v_t, vo_new

    # ══════════════════════════════════════════════════════════════════════
    #  Pseudo-derivative  (Methods section, between Eqs. 9 and 10)
    # ══════════════════════════════════════════════════════════════════════

    def _surrogate_gradient(self, mem: torch.Tensor) -> torch.Tensor:
        """
        Pseudo-derivative ψ_j^t (Methods, "Gradient descent for RSNNs"):

            ψ_j^t = (γ_pd / v_th) · max(0, 1 - |v_j^t - v_th| / v_th)

        Triangular surrogate centered at v_th, peak value γ_pd/v_th.

        [Point 3] Refractory mask: The paper specifies ψ = 0 during the
        refractory period (2–5 ms after each spike). snnTorch's RLeaky does
        NOT implement refractoriness [D1]. After a subtract-reset, the
        membrane drops below threshold, so |v - v_th| is typically > v_th
        and the max(0, ...) clamp gives ψ ≈ 0 naturally. This is approximate
        — not an exact refractory mask — but functionally similar for the
        subtract-reset mechanism used here.
        """
        return (self.gamma / self.threshold) * torch.clamp(
            1.0 - torch.abs((mem - self.threshold) / self.threshold),
            min=0.0,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Entry point
    # ══════════════════════════════════════════════════════════════════════

    def normalize_sequence(self, data: torch.Tensor, timesteps: int | None = None) -> torch.Tensor:
        """Convert input sequence to trainer layout [T, B, n_in]."""
        if data.dim() == 5:
            if timesteps is not None:
                if data.size(0) == timesteps:
                    return data.flatten(start_dim=2)
                if data.size(1) == timesteps:
                    return data.transpose(0, 1).flatten(start_dim=2)
            if data.size(0) == self.batch_size:
                return data.transpose(0, 1).flatten(start_dim=2)
            return data.flatten(start_dim=2)

        if data.dim() == 4 and data.shape[2:] == self.network.in_shape:
            if timesteps is not None:
                if data.size(0) == timesteps:
                    return data
                if data.size(1) == timesteps:
                    return data.transpose(0, 1)
            if data.size(0) == self.batch_size:
                return data.transpose(0, 1)
            return data

        if data.dim() == 3:
            if timesteps is not None:
                if data.size(0) == timesteps:
                    return data
                if data.size(1) == timesteps:
                    return data.transpose(0, 1)
            if data.size(0) == self.batch_size:
                return data.transpose(0, 1)
            return data

        raise ValueError(
            f"Unexpected data shape {tuple(data.shape)}; expected [T, B, ...] or [B, T, ...]."
        )

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on a single batch using recurrent e-prop with online updates.

        Args:
            data:   Input tensor [T, B, ...] — flattened to [T, B, n_in] if needed
            target: Target class labels [B] (integer indices)

        Returns:
            loss: Scalar cross-entropy loss (for monitoring, Eq. 29's loss)
            pred: Predicted class indices [B, 1]
        """
        data = self.normalize_sequence(data)
        if data.ndim > 3:
            data = data.flatten(start_dim=2)

        num_timesteps, batch_size, _ = data.shape
        device = data.device
        n_out = self.network.n_classes

        # One-hot encode targets for classification (Eq. 29: π*_k)
        tgt_onehot = torch.zeros(batch_size, n_out, device=device)
        tgt_onehot.scatter_(1, target.view(-1, 1), 1.0)
        return self._train_recurrent_online(data, tgt_onehot, target)

    # ══════════════════════════════════════════════════════════════════════
    #  Core e-prop loop
    # ══════════════════════════════════════════════════════════════════════

    def _train_recurrent_online(
        self,
        data: torch.Tensor,
        tgt_onehot: torch.Tensor,
        target_indices: torch.Tensor,
    ):
        """
        Online recurrent e-prop for classification (LIF, Eq. 29).

            ΔW_{ji} = -η Σ_t L_j^t · ē_{ji}^t

        where L_j^t = Σ_k W^out_{kj} · (π_k^t - π*_k^t)  (Eq. 4, symmetric)
        and   ē_{ji}^t = F_κ(ψ_j^t · F_α(z_i^{t-1}))     (Eqs. 22–23, 28)

        [Point 6] When update_last=True or update_every>1, weight updates
        occur only at selected timesteps. This deviates from Eq. 29 which
        sums over all t. Eligibility traces are ALWAYS accumulated to
        preserve temporal information; only the weight modification step
        is gated. This is an implementation extension [D3].
        """
        num_timesteps, batch_size, _ = data.shape
        device = data.device

        n_in = data.shape[2]
        n_rec = self.network.hidden_size[0]
        n_out = self.network.n_classes

        # Reset all neuron states: v_j^0 = 0, z_j^0 = 0
        self.network.reset(device=device)

        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad()

        # ── Eligibility vectors (Eq. 22): pre-synaptic filtered traces ──
        x_in_bar = torch.zeros(batch_size, n_in, device=device)    # F_α(x_i^t)
        z_bar_prev = torch.zeros(batch_size, n_rec, device=device) # F_α(z_i^{t-1})

        # ── Kappa-smoothed eligibility traces (Eq. 28) ──
        trace_in = torch.zeros(batch_size, n_rec, n_in, device=device)   # F_κ(e^in)
        trace_rec = torch.zeros(batch_size, n_rec, n_rec, device=device) # F_κ(e^rec)
        trace_out = torch.zeros(batch_size, n_rec, device=device)        # F_κ(z_j^t)

        # [ALIF] a_t = torch.zeros(batch_size, n_rec, device=device)
        # [ALIF] eps_a = torch.zeros(batch_size, n_rec, n_rec, device=device)
        # [ALIF] eps_a_in = torch.zeros(batch_size, n_rec, n_in, device=device)

        # Analog readout membrane y_k^t (Eq. 11, owned by trainer)
        vo = torch.zeros(batch_size, n_out, device=device)
        vo_sum = None

        for t in range(num_timesteps):

            # ── Forward: z_t (spikes, Eq. 7), v_t (membrane, Eq. 6), vo (Eq. 11) ──
            z_t, v_t, vo = self._eprop_step(data[t], vo)

            vo_sum = vo.clone() if vo_sum is None else (vo_sum + vo)

            # STEP 1: Pseudo-derivative ψ_j^t (Methods, below Eq. 9)
            # [Point 3] No explicit refractory mask — see _surrogate_gradient docstring
            h_t = self._surrogate_gradient(v_t)

            # [ALIF] A_t = self.threshold + self.beta * a_t  (Eq. 8)

            # STEP 2: Eligibility vectors (Eq. 22)
            x_in_bar = self.alpha * x_in_bar + data[t]       # F_α(x_i^t)
            # z_bar_prev holds F_α(z^{t-1}) — use BEFORE updating with z_t

            # STEP 3: Eligibility traces (Eq. 23): e = ψ · ε
            e_in = h_t.unsqueeze(2) * x_in_bar.unsqueeze(1)    # [B, n_rec, n_in]
            e_rec = h_t.unsqueeze(2) * z_bar_prev.unsqueeze(1)  # [B, n_rec, n_rec]

            # [ALIF] Eq. 24–25

            # STEP 4: Update z_bar for next timestep
            z_bar_prev = self.alpha * z_bar_prev + z_t

            # [ALIF] Eq. 10: a_t = ρ · a_t + z_t

            # STEP 5: Kappa-smoothed traces (Eq. 28 / Supp. Note 3)
            trace_in = self.kappa * trace_in + e_in
            trace_rec = self.kappa * trace_rec + e_rec
            # [Point 7] Output trace: F_κ(z_j^t) — matches TU Graz reference
            # (pre_term_w_out = exp_convolve(z, decay=readout_decay))
            trace_out = self.kappa * trace_out + z_t

            # ── Skip weight update if not at update timestep [D3] ──
            do_update = True
            if self.update_last and t < num_timesteps - 1:
                do_update = False
            if not ((t + 1) % self.update_every == 0):
                do_update = False
            if not do_update:
                continue

            # STEP 6: Error signal (Eq. 29)
            # π_k^t = softmax(y_k^t), err = π_k^t - π*_k^t
            yo_t = F.softmax(vo, dim=1)
            err_t = yo_t - tgt_onehot

            # STEP 7: Learning signal (Eq. 4, symmetric e-prop)
            # L_j^t = Σ_k W^out_{kj} · (π_k^t - π*_k^t)
            L_t = err_t @ self._w_out

            # STEP 8: Weight gradients (Eq. 29)
            w_in_grad_t = self.lr_layer[0] * torch.einsum("br,bri->ri", L_t, trace_in)
            w_rec_grad_t = self.lr_layer[1] * torch.einsum("br,brj->rj", L_t, trace_rec)
            # [Point 7] Output weight gradient (Supp. Note 3):
            # ΔW^out_{kj} = Σ_b,t err_k · F_κ(z_j) = err^T @ trace_out
            w_out_grad_t = self.lr_layer[2] * (err_t.t() @ trace_out)

            # STEP 9: Apply updates
            if self.use_optimizer and self.optimizer is not None:
                self._w_in.grad = (
                    w_in_grad_t if self._w_in.grad is None
                    else self._w_in.grad + w_in_grad_t
                )
                self._w_rec.grad = (
                    w_rec_grad_t if self._w_rec.grad is None
                    else self._w_rec.grad + w_rec_grad_t
                )
                self._w_out.grad = (
                    w_out_grad_t if self._w_out.grad is None
                    else self._w_out.grad + w_out_grad_t
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                self._w_in.data -= self.lr * w_in_grad_t / batch_size
                self._w_rec.data -= self.lr * w_rec_grad_t / batch_size
                self._w_out.data -= self.lr * w_out_grad_t / batch_size

        # ── Final prediction and monitoring loss ──
        with torch.no_grad():
            pred = vo_sum.argmax(dim=1, keepdim=True)
            # [Point 1] Cross-entropy loss for monitoring, matching paper's
            # E = -Σ_{t,k} π*_k log π_k. We compute it on the accumulated
            # output vo_sum as a summary statistic. Note: the actual e-prop
            # weight updates do NOT use this loss — they use (π_k - π*_k)
            # directly from Eq. 29.
            loss = self.loss_fn(vo_sum, target_indices)

        return loss.detach(), pred

    def set_optimizer_mode(self, use_optimizer: bool, optimizer=None):
        """Switch between pure online updates and optimizer-based updates."""
        self.use_optimizer = use_optimizer
        self._external_optimizer = optimizer
        if use_optimizer:
            self.optimizer = optimizer or self.optimizer or torch.optim.Adam(
                self.network.parameters(), lr=self.lr
            )
        else:
            self.optimizer = None

    def reset(self, device: torch.device | None = None):
        """Reset all neuron states in the network, optionally moving them to a device."""
        self.network.reset(device=device)

    def to(self, device):
        """Move trainer and network to device, recreating optimizer if needed."""
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self