"""
E-prop (Eligibility Propagation) trainer for snnTorch-based networks.

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

Equation map (paper → code):
──────────────────────────────────────────────────────────────────────
  Paper Eq.  │ Description                        │ Code location
─────────────┼────────────────────────────────────┼──────────────────
  Eq. 6      │ LIF membrane potential dynamics    │ network.step()
  Eq. 7      │ Spike generation via Heaviside     │ network.step()
  Eq. 8–10   │ ALIF threshold adaptation          │ [ALIF] blocks
  Eq. 11     │ Leaky readout neuron               │ network.step() → vo
  Eq. 12     │ Low-pass filter F_α definition     │ x_in_bar, z_bar_prev
  Eq. 13     │ Eligibility trace definition       │ e_in, e_rec
  Eq. 14     │ Eligibility vector recursion        │ ε via x_in_bar, z_bar
  Eq. 22     │ LIF elig. vector = F_α(z_i^t)     │ z_bar_prev
  Eq. 23     │ LIF elig. trace = ψ_j · F_α(z_i)  │ e_in, e_rec
  Eq. 24     │ ALIF slow elig. vector ε_{ji,a}   │ [ALIF] block
  Eq. 25     │ ALIF elig. trace with β term       │ [ALIF] block
  Eq. 28     │ Regression weight update + F_κ     │ trace_in/rec/out
  Eq. 29     │ Classification weight update       │ weight update block
  Eq. 4      │ Learning signal L_j^t              │ L_t computation
  Methods    │ Pseudo-derivative ψ_j^t            │ _surrogate_gradient()
──────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from networks.benchmarking.r_snn import RSNN


class EpropTrainer(BaseTrainer):
    """
    E-prop trainer for recurrent SNNs, implementing online weight updates.

    Currently supports:
      - LIF neurons (Eqs. 6–7) with eligibility traces from Eq. 22–23
      - Classification via cross-entropy (Eq. 29)
      - Symmetric e-prop (B_{jk} = W^out_{kj})

    Does NOT yet support:
      - ALIF neurons (Eqs. 8–10) with slow eligibility traces (Eqs. 24–25)
      - Random e-prop or adaptive e-prop feedback
      - Reward-based e-prop (Eq. 5, 36–37)
    """

    def __init__(
        self,
        network: RSNN,
        lr: float,
        batch_size: int,
        gamma: float = 0.3,
        tau_mem: float = 0.9,
        tau_out: float = 0.9,
        lr_layer_norm: tuple = (1.0, 1.0, 1.0),
        quant: bool = False,
        use_optimizer: bool = False,
        optimizer=None,
        update_last: bool = False,
        update_every: int = 1,
        seq_batch_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        # γ_pd in the paper (Methods, below Eq. 9): dampening factor for
        # the pseudo-derivative. Bellec et al. use γ_pd = 0.3.
        self.gamma = gamma
        self.quant = quant
        self.use_optimizer = use_optimizer
        self.update_last = update_last
        self.update_every = update_every
        self.seq_batch_size = seq_batch_size

        if not (hasattr(network, "is_recurrent") and network.is_recurrent):
            raise TypeError(
                "EpropTrainer requires a recurrent RSNN; "
                f"got network={type(network).__name__}."
            )

        # ── Decay factors from the paper ──
        #
        # Reference values from Bellec et al. 2020 (Methods section):
        #   δt    = 1 ms           time step (used in all simulations)
        #   τ_m   = 20 ms          membrane time constant
        #   τ_out = 30 ms          readout time constant (TU Graz reference code)
        #   v_th  = 0.03           threshold (TU Graz reference code)
        #   γ_pd  = 0.3            pseudo-derivative dampening factor
        #
        # Derived reference values:
        #   α  = exp(-δt/τ_m)   = exp(-1/20)  ≈ 0.9512
        #   κ  = exp(-δt/τ_out) = exp(-1/30)  ≈ 0.9672
        #
        # [ALIF] reference values (Methods, Eqs. 8–10):
        #   τ_a  = 100–2000 ms     adaptation time constant (task-dependent)
        #   ρ    = exp(-δt/τ_a)    ≈ 0.995 (τ_a=200ms) to 0.9995 (τ_a=2000ms)
        #   β    ≈ 0.07            adaptation strength
        #
        PAPER_REF = {
            "dt":      1.0,       # ms
            "tau_m":   20.0,      # ms  →  α ≈ 0.9512
            "tau_out": 30.0,      # ms  →  κ ≈ 0.9672
            "v_th":    0.03,      # arbitrary units
            "gamma":   0.3,       # pseudo-derivative dampening
            "alpha":   0.9512,    # exp(-dt/tau_m), from Eq. 6
            "kappa":   0.9672,    # exp(-dt/tau_out), from Eq. 11

            # [ALIF]
            "tau_a":   200.0,     # ms (task-dependent, 100–2000 ms)
            "beta":    0.07,      # adaptation strength

        }

        # α = exp(-δt/τ_m) — membrane decay, Eq. 6
        # α  = exp(-δt/τ_m)   = exp(-1/20)  ≈ 0.9512
        self.alpha = PAPER_REF["alpha"]
        # κ = exp(-δt/τ_out) — readout decay, Eq. 11
        # κ  = exp(-δt/τ_out) = exp(-1/30)  ≈ 0.9672
        self.kappa = PAPER_REF["kappa"]
        # v_th — firing threshold, Eqs. 7, 9
        self.threshold = PAPER_REF["v_th"]

        # [ALIF] Would also need:
        # self.rho = float(network.rho)    # ρ = exp(-δt/τ_a), Eq. 10
        # self.beta = float(network.beta)  # β, adaptation strength, Eq. 8
        #
        # Reference: ρ = exp(-1/200) ≈ 0.9950, β = 0.07

        # Per-layer learning rate modulation (not from paper, practical addition)
        self.lr_layer = lr_layer_norm
        # Loss function for monitoring (actual gradients are computed analytically)
        self.loss_fn = nn.MSELoss()

        self._external_optimizer = optimizer
        if use_optimizer:
            if optimizer is not None:
                self.optimizer = optimizer
            else:
                self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        else:
            self.optimizer = None

    # ══════════════════════════════════════════════════════════════════════
    #  Pseudo-derivative  (Methods section, between Eqs. 9 and 10)
    # ══════════════════════════════════════════════════════════════════════

    def _surrogate_gradient(self, mem: torch.Tensor) -> torch.Tensor:
        """
        Compute the pseudo-derivative ψ_j^t (Methods, "Gradient descent for RSNNs").

        The paper defines (for LIF, where A_j^t = v_th):

            ψ_j^t = (1/v_th) · γ_pd · max(0, 1 - |v_j^t - v_th| / v_th)

        This is a triangular function centered at v_th with support [0, 2·v_th]
        and peak value γ_pd / v_th at v = v_th.

        The 1/v_th prefactor comes from ∂z_j^t/∂v_j^t: since z = H(v - v_th),
        the surrogate replaces the Dirac delta δ(v - v_th) which has units 1/v_th.

        In the TU Graz reference code:
            v_scaled = (v - thr) / thr        # normalized: 0 at threshold, -1 at rest
            post_term = pseudo_derivative(v_scaled, γ) / thr
        which gives exactly (γ/thr) · max(0, 1 - |v - thr|/thr).

        During the refractory period, ψ should be 0. This is handled implicitly
        because snnTorch clamps the membrane potential after a spike, pushing
        |v - v_th| > v_th and thus ψ = 0 via the max(0, ...) clamp.

        Args:
            mem: Membrane potential v_j^t, shape [batch, n_rec]

        Returns:
            ψ_j^t, shape [batch, n_rec]
        """
        return (self.gamma / self.threshold) * torch.clamp(
            1.0 - torch.abs((mem - self.threshold) / self.threshold),
            min=0.0,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Entry point
    # ══════════════════════════════════════════════════════════════════════

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on a single batch using recurrent e-prop with online updates.

        Args:
            data:   Input tensor [num_timesteps, batch_size, n_in]
            target: Target class labels [batch_size]  (integer indices)

        Returns:
            loss: Scalar loss tensor (for monitoring; not used for learning)
            pred: Predicted class indices [batch_size, 1]
        """

        
        num_timesteps = int(data.shape[0])
        batch_size = int(data.shape[1])
        if data.dim() > 3:
            data = data.flatten(2)
        device = data.device
        n_out = self.network.n_out

        # One-hot encode targets for classification (Eq. 29 uses π*_k)
        tgt_onehot = torch.zeros(batch_size, n_out, device=device)
        tgt_onehot.scatter_(1, target.view(-1, 1), 1.0)
        return self._train_recurrent_online(data, tgt_onehot)

    # ══════════════════════════════════════════════════════════════════════
    #  Core e-prop loop  (online, timestep-by-timestep)
    # ══════════════════════════════════════════════════════════════════════

    def _train_recurrent_online(self, data: torch.Tensor, tgt_onehot: torch.Tensor):
        """
        Online recurrent e-prop for classification (LIF neurons).

        Implements Eq. 29 from the paper:

            ΔW^rec_{ji} = -η Σ_t [ Σ_k B_{jk}(π_k^t - π*_k^t) ] · ē_{ji}^t

        where:
            - π_k^t = softmax_k(y_1^t, ..., y_K^t)    predicted class probs
            - π*_k^t                                    target one-hot
            - B_{jk} = W^out_{kj}                      symmetric feedback
            - ē_{ji}^t = F_κ(e_{ji}^t)                 kappa-smoothed elig. trace
            - e_{ji}^t = ψ_j^t · ε_{ji}^t              eligibility trace (Eq. 23)
            - ε_{ji}^t = F_α(z_i^{t-1})                eligibility vector (Eq. 22)
                 [for input weights: ε_{ji}^t = F_α(x_i^t), see note below Eq. 23]

        The kappa-smoothing (F_κ) arises because the readout neuron (Eq. 11)
        applies a leaky filter to the spikes before computing the loss. As
        derived in Supplementary Note 3, this transforms the eligibility trace
        e_{ji}^t into ē_{ji}^t = F_κ(e_{ji}^t) in the weight update formula.
        """
        if data.dim() > 3:
            data = data.flatten(2)
        num_timesteps = int(data.shape[0])
        batch_size = int(data.shape[1])
        device = data.device

        n_in = self.network.n_in
        n_rec = self.network.n_rec
        n_out = self.network.n_out

        # ── Reset network state ──
        # Sets v_j^0 = 0, z_j^0 = 0, y_k^0 = 0 (initial conditions)
        self.network.reset(device=device)

        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad()

        # ══════════════════════════════════════════════════════════════════
        #  State variables for eligibility computation
        # ══════════════════════════════════════════════════════════════════

        # ── Pre-synaptic filtered traces (= eligibility vectors, Eq. 14/22) ──
        #
        # For LIF neurons, the eligibility vector ε_{ji}^t (Eq. 13) reduces
        # to a simple low-pass filter of the presynaptic activity (Eq. 22):
        #
        #   ε_{ji}^{t+1} = F_α(z_i^t)     for recurrent weights W^rec_{ji}
        #   ε_{ji}^{t+1} = F_α(x_i^{t+1}) for input weights W^in_{ji}
        #
        # The asymmetry in time indices comes from Eq. 6: the membrane update
        # at time t+1 depends on recurrent spikes z^t but input spikes x^{t+1}.
        # (See the note below Eq. 23 in the paper.)

        # F_α(x_i^t): filtered input spikes. Eq. 12 applied to inputs.
        x_in_bar = torch.zeros(batch_size, n_in, device=device)

        # F_α(z_i^{t-1}): filtered PREVIOUS recurrent spikes.
        # CRITICAL: must use z^{t-1}, not z^t. The TU Graz reference
        # implements this via shift_by_one_time_step(z). In our online loop,
        # we achieve this by computing e_rec BEFORE updating z_bar with z_t.
        z_bar_prev = torch.zeros(batch_size, n_rec, device=device)

        # ── Kappa-smoothed eligibility traces (Eq. 28, Supplementary Note 3) ──
        #
        # The readout neuron y_k^t = κ·y_k^{t-1} + Σ_j W^out_{kj} z_j^t (Eq. 11)
        # introduces a low-pass filter between spikes and the loss. As shown in
        # Supplementary Note 3, this means the weight update uses ē_{ji}^t = F_κ(e_{ji}^t)
        # instead of e_{ji}^t directly:
        #
        #   ē_{ji}^t = κ · ē_{ji}^{t-1} + e_{ji}^t     (Eq. 12 with κ)
        #
        # For the output weights, the relevant trace is simply F_κ(z_j^t).

        # F_κ(e^in_{ji}): smoothed input eligibility traces [batch, n_rec, n_in]
        trace_in = torch.zeros(batch_size, n_rec, n_in, device=device)
        # F_κ(e^rec_{ji}): smoothed recurrent eligibility traces [batch, n_rec, n_rec]
        trace_rec = torch.zeros(batch_size, n_rec, n_rec, device=device)
        # F_κ(z_j^t): smoothed spike train for output weight update
        trace_out = torch.zeros(batch_size, n_rec, device=device)

        # [ALIF] Additional state variables for adaptive threshold neurons:
        # # a_j^t: adaptive threshold component, Eq. 10
        # a_t = torch.zeros(batch_size, n_rec, device=device)
        # # ε_{ji,a}^t: slow component of eligibility vector, Eq. 24
        # eps_a = torch.zeros(batch_size, n_rec, n_rec, device=device)  # recurrent
        # eps_a_in = torch.zeros(batch_size, n_rec, n_in, device=device)  # input

        # ── Output membrane potential ──
        # y_k^t from Eq. 11 (before softmax)
        vo = torch.zeros(batch_size, n_out, device=device)
        # Accumulated output for final prediction (not part of e-prop per se)
        vo_sum = None

        # ══════════════════════════════════════════════════════════════════
        #  Main loop over timesteps
        # ══════════════════════════════════════════════════════════════════

        for t in range(num_timesteps):

            # ── Forward pass: Eqs. 6, 7, 11 ──
            # network.step() computes:
            #   v_j^{t+1} = α·v_j^t + Σ_{i≠j} W^rec_{ji}·z_i^t
            #                + Σ_i W^in_{ji}·x_i^{t+1} - z_j^t·v_th    (Eq. 6)
            #   z_j^{t+1} = H(v_j^{t+1} - v_th)                        (Eq. 7)
            #   y_k^{t+1} = κ·y_k^t + Σ_j W^out_{kj}·z_j^{t+1} + b_k  (Eq. 11)
            #
            # Returns z_t = z^{t+1}, v_t = v^{t+1}, vo = y^{t+1} in paper notation.
            # (The +1 offset is because network.step() advances one timestep.)
            z_t, v_t, vo = self.network.step(data[t], vo)

            # ── Shape normalisation ──
            # snnTorch may return [n_rec, batch] instead of [batch, n_rec]
            if z_t.shape == (n_rec, batch_size):
                z_t = z_t.t()
            if v_t.shape == (n_rec, batch_size):
                v_t = v_t.t()
            if vo.shape == (n_out, batch_size):
                vo = vo.t()

            assert z_t.shape == (batch_size, n_rec), \
                f"z_t shape {z_t.shape} != expected {(batch_size, n_rec)}"
            assert v_t.shape == (batch_size, n_rec), \
                f"v_t shape {v_t.shape} != expected {(batch_size, n_rec)}"
            assert vo.shape == (batch_size, n_out), \
                f"vo shape {vo.shape} != expected {(batch_size, n_out)}"

            # Accumulate output for final prediction
            vo_sum = vo.clone() if vo_sum is None else (vo_sum + vo)

            # ══════════════════════════════════════════════════════════════
            #  STEP 1: Pseudo-derivative ψ_j^t
            #  (Methods, "Gradient descent for RSNNs", below Eq. 9)
            #
            #  ψ_j^t = (γ_pd / v_th) · max(0, 1 - |v_j^t - v_th| / v_th)
            #
            #  For LIF: A_j^t = v_th (constant threshold).
            #  For ALIF: replace v_th with A_j^t = v_th + β·a_j^t (Eq. 8).
            # ══════════════════════════════════════════════════════════════

            h_t = self._surrogate_gradient(v_t)  # [batch, n_rec]

            # [ALIF] With adaptive threshold, ψ would use the dynamic threshold:
            # A_t = self.threshold + self.beta * a_t             # Eq. 8
            # h_t = (self.gamma / self.threshold) * torch.clamp(
            #     1.0 - torch.abs((v_t - A_t) / self.threshold),
            #     min=0.0,
            # )

            # ══════════════════════════════════════════════════════════════
            #  STEP 2: Pre-synaptic filtered traces (eligibility vectors)
            #  (Eqs. 12, 14, 22)
            #
            #  General recursion (Eq. 14):
            #    ε_{ji}^t = (∂h_j^t / ∂h_j^{t-1}) · ε_{ji}^{t-1}
            #               + ∂h_j^t / ∂W_{ji}
            #
            #  For LIF (Eq. 22), this simplifies to:
            #    ε_{ji}^{t+1} = α · ε_{ji}^t + z_i^t       (recurrent)
            #    ε_{ji}^{t+1} = α · ε_{ji}^t + x_i^{t+1}   (input)
            #
            #  because ∂v^{t+1}/∂v^t = α  and  ∂v^t/∂W^rec_{ji} = z_i^{t-1},
            #  ∂v^t/∂W^in_{ji} = x_i^t  (see derivation below Eq. 23).
            # ══════════════════════════════════════════════════════════════

            # Input eligibility vector: F_α(x_i^t)
            # Uses current input x^t (not x^{t-1}), per Eq. 6 convention
            # where W^in appears with x^{t+1} in the update for v^{t+1}.
            x_in_bar = self.alpha * x_in_bar + data[t]  # [batch, n_in]

            # Recurrent eligibility vector: F_α(z_i^{t-1})
            # z_bar_prev currently holds F_α accumulated up to z^{t-1}.
            # We use it NOW, BEFORE updating with z_t.
            # This matches the paper: ε_{ji}^{t+1} = F_α(z_i^t), meaning
            # at the current computation step, the trace uses the spike
            # from the PREVIOUS timestep (Eq. 22, and note below Eq. 23:
            # "one needs to replace the network spikes z_i^{t-1}").
            # z_bar_prev is ready to use as-is here.  [batch, n_rec]

            # ══════════════════════════════════════════════════════════════
            #  STEP 3: Eligibility traces e_{ji}^t  (Eq. 23)
            #
            #  For LIF:
            #    e_{ji}^{t+1} = ψ_j^{t+1} · ε_{ji}^{t+1}
            #                 = ψ_j^{t+1} · F_α(z_i^t)        (Eq. 23)
            #
            #  This is the product of the post-synaptic term (ψ) and the
            #  pre-synaptic term (filtered spike train). It captures how
            #  much the spike of neuron j at time t+1 depends on the
            #  weight W_{ji}, locally and in a forward manner.
            # ══════════════════════════════════════════════════════════════

            # e^in_{ji}^t = ψ_j^t · F_α(x_i^t)
            # Shape: [batch, n_rec, n_in] = [batch, n_rec, 1] * [batch, 1, n_in]
            e_in = h_t.unsqueeze(2) * x_in_bar.unsqueeze(1)

            # e^rec_{ji}^t = ψ_j^t · F_α(z_i^{t-1})
            # Shape: [batch, n_rec, n_rec] = [batch, n_rec, 1] * [batch, 1, n_rec]
            e_rec = h_t.unsqueeze(2) * z_bar_prev.unsqueeze(1)

            # [ALIF] For ALIF neurons, the eligibility trace has an additional
            # slow component from the adaptive threshold (Eq. 25):
            #
            #   e_{ji}^t = ψ_j^t · ( F_α(z_i^{t-1}) - β · ε_{ji,a}^t )
            #
            # where ε_{ji,a}^t is the slow eligibility vector (Eq. 24):
            #
            #   ε_{ji,a}^{t+1} = ψ_j^t · F_α(z_i^{t-1}) + (ρ - ψ_j^t · β) · ε_{ji,a}^t
            #
            # The term (ρ - ψ·β) causes the slow trace to decay with the
            # adaptation time constant τ_a (via ρ), creating the "highways
            # into the future" discussed in the paper (p. 6).
            #
            # For recurrent weights:
            # eps_a = h_t.unsqueeze(2) * z_bar_prev.unsqueeze(1) \
            #       + (self.rho - h_t.unsqueeze(2) * self.beta) * eps_a     # Eq. 24
            # e_rec = h_t.unsqueeze(2) * (z_bar_prev.unsqueeze(1)
            #                             - self.beta * eps_a)              # Eq. 25
            #
            # For input weights:
            # eps_a_in = h_t.unsqueeze(2) * x_in_bar.unsqueeze(1) \
            #          + (self.rho - h_t.unsqueeze(2) * self.beta) * eps_a_in
            # e_in = h_t.unsqueeze(2) * (x_in_bar.unsqueeze(1)
            #                            - self.beta * eps_a_in)

            # ══════════════════════════════════════════════════════════════
            #  STEP 4: Update recurrent pre-synaptic trace for next timestep
            #
            #  After computing e_rec with z_bar_prev = F_α(z^{t-1}),
            #  we now incorporate z_t so that at the NEXT iteration,
            #  z_bar_prev will correctly hold F_α(z^t) = F_α(z^{(t+1)-1}).
            # ══════════════════════════════════════════════════════════════

            z_bar_prev = self.alpha * z_bar_prev + z_t  # [batch, n_rec]

            # [ALIF] Update adaptive threshold for next timestep (Eq. 10):
            # a_t = self.rho * a_t + z_t
            # This feeds back into the dynamic threshold A_j^t (Eq. 8)
            # and the pseudo-derivative computation at the next timestep.

            # ══════════════════════════════════════════════════════════════
            #  STEP 5: Kappa-smoothed eligibility traces ē_{ji}^t
            #  (Eq. 28 / Supplementary Note 3)
            #
            #  The readout y_k^t = κ·y_k^{t-1} + Σ_j W^out_{kj}·z_j^t
            #  (Eq. 11) introduces a temporal filter between the spikes
            #  and the loss. For regression (Eq. 28), this requires
            #  convolving the eligibility traces with the same filter:
            #
            #    ē_{ji}^t = F_κ(e_{ji}^t) = κ · ē_{ji}^{t-1} + e_{ji}^t
            #
            #  For classification (Eq. 29) with cross-entropy loss, the
            #  same smoothing applies (derived in Supplementary Note 3).
            #
            #  For the output weights W^out_{kj}, the gradient involves
            #  F_κ(z_j^t) rather than F_κ(e_{ji}^t), since the output
            #  weights don't go through the recurrent eligibility traces.
            # ══════════════════════════════════════════════════════════════

            # ē^in_{ji} = F_κ(e^in_{ji})
            trace_in = self.kappa * trace_in + e_in      # [batch, n_rec, n_in]
            # ē^rec_{ji} = F_κ(e^rec_{ji})
            trace_rec = self.kappa * trace_rec + e_rec    # [batch, n_rec, n_rec]
            # F_κ(z_j^t) for output weight gradient
            trace_out = self.kappa * trace_out + z_t      # [batch, n_rec]

            # ══════════════════════════════════════════════════════════════
            #  Skip weight update if not at an update timestep
            #
            #  Traces are ALWAYS updated (steps 1–5 above) to preserve
            #  temporal information. Only the weight modification is gated.
            # ══════════════════════════════════════════════════════════════

            do_update = True
            if self.update_last and t < num_timesteps - 1:
                do_update = False
            if not ((t + 1) % self.update_every == 0):
                do_update = False
            if not do_update:
                continue

            # ══════════════════════════════════════════════════════════════
            #  STEP 6: Output probabilities and error signal
            #
            #  For classification (Eq. 29):
            #    π_k^t = softmax_k(y_1^t, ..., y_K^t)
            #    err_k^t = π_k^t - π*_k^t
            #
            #  where π*_k^t is the one-hot target vector.
            # ══════════════════════════════════════════════════════════════

            # π_k^t: predicted class probabilities
            yo_t = F.softmax(vo, dim=1)   # [batch, n_out]
            # Error: (π_k^t - π*_k^t), appears in Eq. 29
            err_t = yo_t - tgt_onehot     # [batch, n_out]

            # ══════════════════════════════════════════════════════════════
            #  STEP 7: Learning signal L_j^t  (Eq. 4)
            #
            #  L_j^t = Σ_k B_{jk} · (π_k^t - π*_k^t)
            #
            #  In *symmetric* e-prop, B_{jk} = W^out_{kj} (the transpose
            #  of the output weight from neuron j to output k). This would
            #  be exact if the network had no recurrent connections.
            #
            #  In *random* e-prop, B_{jk} are random fixed weights.
            #  In *adaptive* e-prop, B_{jk} co-evolve with W^out_{kj}.
            #  (See paper p.4 and Supplementary Note 2.)
            #
            #  Implementation: err_t @ W_out gives:
            #    [batch, n_out] @ [n_out, n_rec] = [batch, n_rec]
            #  which computes Σ_k err_k · W^out_{kj} = Σ_k B_{jk} · err_k
            # ══════════════════════════════════════════════════════════════

            L_t = err_t @ self.network.w_out  # [batch, n_rec]

            # ══════════════════════════════════════════════════════════════
            #  STEP 8: Weight gradients  (Eq. 29)
            #
            #  The full weight update for classification is:
            #
            #    ΔW^rec_{ji} = -η Σ_t L_j^t · ē^rec_{ji}^t     (Eq. 29)
            #    ΔW^in_{ji}  = -η Σ_t L_j^t · ē^in_{ji}^t      (same form)
            #    ΔW^out_{kj} = -η Σ_t err_k^t · F_κ(z_j^t)     (Supp. Note 3)
            #
            #  Here we compute the per-timestep contribution (before
            #  summation over t), averaged over the batch dimension.
            #
            #  The einsum "br,bri->ri" computes:
            #    grad[r,i] = Σ_b L_t[b,r] · trace[b,r,i]
            #  which is Σ_b L_j^t · ē_{ji}^t summed over the batch.
            # ══════════════════════════════════════════════════════════════

            # ΔW^in: gradient for input weights [n_rec, n_in]
            w_in_grad_t = self.lr_layer[0] * torch.einsum(
                "br,bri->ri", L_t, trace_in
            )
            # ΔW^rec: gradient for recurrent weights [n_rec, n_rec]
            w_rec_grad_t = self.lr_layer[1] * torch.einsum(
                "br,brj->rj", L_t, trace_rec
            )
            # ΔW^out: gradient for output weights [n_out, n_rec]
            # = Σ_b err_k^t · F_κ(z_j^t) = err^T @ trace_out
            w_out_grad_t = self.lr_layer[2] * (err_t.t() @ trace_out)

            # ══════════════════════════════════════════════════════════════
            #  STEP 9: Apply weight updates
            #
            #  Two modes:
            #  (a) With optimizer: accumulate .grad, then optimizer.step()
            #  (b) Manual SGD: W ← W - η · ΔW / batch_size
            #
            #  Note: the paper uses ΔW_{ji} = -η Σ_t L_j^t · ē_{ji}^t
            #  (Eq. 27), so the update is W ← W + ΔW = W - η·(L·ē).
            #  Since we compute grad = L·ē (positive), the manual update
            #  subtracts: W.data -= lr * grad / batch_size.
            # ══════════════════════════════════════════════════════════════

            if self.use_optimizer and self.optimizer is not None:
                self.network.w_in.grad = (
                    w_in_grad_t
                    if self.network.w_in.grad is None
                    else self.network.w_in.grad + w_in_grad_t
                )
                self.network.w_rec.grad = (
                    w_rec_grad_t
                    if self.network.w_rec.grad is None
                    else self.network.w_rec.grad + w_rec_grad_t
                )
                self.network.w_out.grad = (
                    w_out_grad_t
                    if self.network.w_out.grad is None
                    else self.network.w_out.grad + w_out_grad_t
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                self.network.w_in.data -= self.lr * w_in_grad_t / batch_size
                self.network.w_rec.data -= self.lr * w_rec_grad_t / batch_size
                self.network.w_out.data -= self.lr * w_out_grad_t / batch_size

        # ── Final prediction (not part of e-prop, just for evaluation) ──
        with torch.no_grad():
            pred = vo_sum.argmax(dim=1, keepdim=True)
            loss = self.loss_fn(vo_sum, tgt_onehot)

        return loss.detach(), pred

    def reset(self):
        """Reset all LIF neuron states in the network."""
        self.network.reset()

    def to(self, device):
        """Move trainer and network to device, recreating optimizer if needed."""
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self
