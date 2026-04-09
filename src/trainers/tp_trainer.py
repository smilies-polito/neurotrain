"""
Trace Propagation (TP) Trainer.

Supports FCSNN, RSNN, and generic CNN architectures (e.g. VGG9) in a single unified class.

Architecture detection (duck-typing):
  FCSNN        → hasattr(network, 'synapses')
  RSNN         → hasattr(network, 'input_layers')
  VGG9 (CNN)   → hasattr(network, 'VGG9_CFG')

Design note — student path via network.forward():
  The TP algorithm requires two forward paths through the same weight matrices:
    • Student path (green): input x_t propagated through W_l → produces s_l^t
    • Target path  (purple): one-hot c* projected via S → propagated through W_l → s̃_l^t
  The STUDENT path is computed by calling network(data[t]) at each timestep — the
  network's own snntorch dynamics (beta, reset mechanism, surrogate) are adopted.
  The TARGET path is emulated by the trainer using the network's weight matrices
  directly, with the trainer's own alpha/vth/ArcTan-surrogate dynamics.

  TBPTT is naturally prevented: torch.autograd.grad(..., retain_graph=False) frees
  the computation graph after each timestep's gradient computation.  snntorch's
  internal state (lif.mem) still holds the correct VALUE for the next forward pass,
  but its grad_fn is gone — so the next timestep's gradient only flows through the
  current-step computation, not through time.

Design note — target path dynamics:
  The target path uses the trainer's own ArcTan surrogate (_spike) applied to:
    ṽ_l^t = α·ṽ_l^{t-1} + W_l·s̃_{l-1}^t - V_th·s̃_l^{t-1}   (Eq 1)
  This gives the trainer independent control over the target path regardless of
  the network's snntorch configuration.

Design note — output layer (paper Sec 3.1):
  During training the TP algorithm uses a pure Leaky Integrator at the output:
    mem_out^t = mem_out^{t-1} + W_out · s_{L-1}^t
  No alpha decay, no threshold, no spike.  The trainer always applies this rule.

  For evaluation to be CONSISTENT with training, configure the network with:
    out_integrator=True  (parameter available in FCSNN and RSNN)
  which sets beta=1.0 and threshold=1e9 on the output neuron so it never fires.
  VGG9 networks already have a correct Leaky Integrator head (leak=1.0, no fire).

  During eval, use mem_rec[-1] at the FINAL timestep only — do NOT accumulate it
  over timesteps.  The integrator already accumulates internally (mem += W·spk each
  step); summing over timesteps gives a triangular sum, not a pure integral.

Design note — batch size constraint (paper Sec 3.2):
  TP requires batch_size >= 2.  The contrastive loss (Eq 13-15) computes a [B,B]
  inter-sample similarity matrix z_l = ε_l @ ε̃_l^T.  With B=1 this is a 1×1
  scalar, log_softmax([x]) = 0 always, and all gradients vanish — learning is
  impossible.  This is a fundamental algorithmic requirement, not a limitation of
  this implementation.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class TPTrainer(BaseTrainer):
    """
    Unified Trace Propagation trainer for FCSNN, RSNN, and VGG9-based CNNs.

    Args:
        network:                  The SNN.  Its forward() provides the student path.
        lr:                       Learning rate for SGD.
        batch_size:               Must be >= 2.  The contrastive loss computes a
                                  [B,B] similarity matrix; with B=1 log_softmax
                                  collapses to 0 and all gradients vanish
                                  (paper Sec 3.2, fundamental algorithmic requirement).
        alpha:                    TARGET-PATH membrane decay (α in paper Eq 1).
                                  The student path uses the network's own snntorch beta.
        beta:                     Eligibility trace decay (β in paper Eq 11-12).
        vth:                      TARGET-PATH spike threshold (V_th in paper Eq 2).
                                  The student path uses the network's own threshold.
        surrogate_scale:          ArcTan surrogate gradient scale for the TARGET path.
                                  The student path uses the network's own surrogate.
        train_target_propagator:  Whether to train the projection matrix S.
        use_optimizer:            If True, create an internal SGD optimizer.
        optimizer:                External optimizer (overrides use_optimizer).
        input_shape:              Required for CNN networks (e.g. (2, 128, 128)).
                                  Used to infer the S projection matrix size.
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        alpha: float = 0.98,
        beta: float = 0.98,
        vth: float = 1.0,
        surrogate_scale: float = 1.0,
        train_target_propagator: bool = True,
        use_optimizer: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        input_shape: Optional[Tuple] = None,
        **kwargs,
    ):
        super().__init__()

        if batch_size < 2:
            raise ValueError(
                "TP requires batch_size >= 2: the contrastive loss (Eq 14) computes a "
                "[B,B] inter-sample similarity matrix; with B=1 log_softmax collapses "
                "to 0 and all gradients vanish.  This is a fundamental algorithmic "
                "constraint (paper Sec 3.2)."
            )

        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = alpha           # α: target-path membrane decay (Eq 1)
        self.beta = beta             # β: trace decay (Eq 11, 12)
        self.vth = vth               # V_th: target-path spike threshold (Eq 2)
        self.surrogate_scale = surrogate_scale
        self.train_target_propagator = train_target_propagator

        # Extract learnable weight blocks and output layer from the network
        self._extract_layers()

        # Initialize projection matrix S (Alg 1 line 14: for l=1, W_l = S)
        # S projects the one-hot target c* ∈ R^C to the first hidden layer's space
        self.S = self._init_S(input_shape)

        # Optimizer: includes all network params + S.
        # network.parameters() already covers recurrent weights for RSNN.
        if optimizer is not None:
            self.optimizer = optimizer
        elif use_optimizer:
            params = list(self.network.parameters())
            if self.train_target_propagator:
                params.append(self.S.weight)
            self.optimizer = torch.optim.Adam(params, lr=lr)
        else:
            self.optimizer = None

    # -------------------------------------------------------------------------
    # Layer extraction
    # -------------------------------------------------------------------------

    def _extract_layers(self):
        """
        Build self.blocks, self.output_layer, self.recurrent_weights.

        blocks: list of (weight_layer, pool_fn_or_None) for each hidden block.
          - weight_layer is nn.Linear for FC/Rec, WSConv2d for CNN.
          - pool_fn is None for FC/Rec, MaxPool/AdaptiveAvgPool for CNN.
            For VGG9: pool_fn is stored for the TARGET path (the student path
            already has pooling applied by network.forward()).
        output_layer: the final nn.Linear (used as pure integrator during training).
        recurrent_weights: list of nn.Linear recurrent modules (RSNN only; empty otherwise).
          Needed so the trainer can explicitly request their gradient via autograd.grad().
        """
        if hasattr(self.network, 'synapses'):          # FCSNN
            self.blocks = [(w, None) for w in self.network.synapses[:-1]]
            self.output_layer = self.network.synapses[-1]
            self.recurrent_weights: List[nn.Linear] = []

        elif hasattr(self.network, 'input_layers'):    # RSNN
            # Recurrent weights (rlif.recurrent) are used inside network.forward().
            # _detach_network_hidden() detaches rlif.spk before each timestep so
            # the gradient for recurrent.weight uses the current-step spike only (no TBPTT).
            self.blocks = [(w, None) for w in self.network.input_layers]
            self.output_layer = self.network.fc_out
            self.recurrent_weights = [
                rlif.recurrent for rlif in self.network.recurrent_layers
            ]

        elif hasattr(self.network, 'VGG9_CFG'):        # VGG9 CNN architecture
            # pool_fn stored for the TARGET path; network.forward() already applies
            # pool before returning spk_list[l] (student path).
            self.blocks = []
            for i, (_, _, _) in enumerate(self.network.VGG9_CFG, start=1):
                conv = getattr(self.network, f'conv{i}')
                pool = getattr(self.network, f'pool{i}', None)
                self.blocks.append((conv, pool))
            self.output_layer = self.network.head.fc
            self.recurrent_weights = []

        else:
            raise ValueError(
                "TPTrainer supports FCSNN (synapses), RSNN (input_layers), "
                "or VGG9-based CNNs (VGG9_CFG). Unknown network type."
            )

        self.n_blocks = len(self.blocks)
        self.n_classes = self.output_layer.out_features

    def _init_S(self, input_shape: Optional[Tuple]) -> nn.Linear:
        """
        Initialize projection matrix S ∈ R^{C × H_1}.
        S projects the one-hot target c* from C classes to the first hidden
        layer's feature space (Alg 1 lines 13-14, Fig 2).
        """
        layer, pool_fn = self.blocks[0]

        if isinstance(layer, nn.Linear):
            s_out_size = layer.out_features
        else:
            if input_shape is None:
                raise ValueError(
                    "TPTrainer requires input_shape=(C, H, W) for CNN networks "
                    "(e.g. input_shape=(2, 128, 128) for DVSGesture)."
                )
            with torch.no_grad():
                dev = next(self.network.parameters()).device
                dummy = torch.zeros(1, *input_shape, device=dev)
                out = layer(dummy)                            # (1, C, H, W)
                if pool_fn is not None:
                    out = pool_fn((out >= self.vth).float())  # apply pool shape
            s_out_size = out[0].numel()

        S = nn.Linear(self.n_classes, s_out_size, bias=False)
        nn.init.kaiming_normal_(S.weight)
        return S

    # -------------------------------------------------------------------------
    # TBPTT prevention — detach network hidden states between timesteps
    # -------------------------------------------------------------------------

    def _detach_network_hidden(self):
        """
        Detach snntorch's internal membrane/spike tensors to prevent TBPTT.

        After torch.autograd.grad(..., retain_graph=False), the computation graph
        is freed but snntorch's stored hidden tensors (lif.mem, rlif.spk, etc.)
        still carry grad_fn pointers into that freed graph.  On the next forward,
        snntorch builds a new graph that includes those stale tensors → backward
        tries to traverse the freed graph → RuntimeError.

        Detaching replaces each hidden tensor with a fresh leaf (same value, no
        grad_fn) so the next timestep's graph is self-contained.

        This is training logic and deliberately lives in the trainer, not in the
        network (networks are pure inference modules).
        """
        if hasattr(self.network, 'synapses'):          # FCSNN
            for neu in self.network.neurons:
                neu.mem = neu.mem.detach()

        elif hasattr(self.network, 'input_layers'):    # RSNN
            for rlif in self.network.recurrent_layers:
                rlif.mem = rlif.mem.detach()
                rlif.spk = rlif.spk.detach()
            self.network.lif_out.mem = self.network.lif_out.mem.detach()

        elif hasattr(self.network, 'VGG9_CFG'):        # VGG9 CNN architecture
            for i in range(1, self.network._num_blocks + 1):
                attr = f'mem{i}'
                setattr(self.network, attr, getattr(self.network, attr).detach())
            self.network.head.mem = self.network.head.mem.detach()

    # -------------------------------------------------------------------------
    # Spike function with ArcTan surrogate gradient — used for the TARGET path
    # -------------------------------------------------------------------------

    class _SpikeFunction(torch.autograd.Function):
        """
        Forward: hard threshold Θ(v - V_th).
        Backward: ArcTan surrogate  scale / (1 + (π·(v - V_th))²).

        Used for the TARGET path only.  The student path uses the network's
        own surrogate gradient (applied inside network.forward()).
        (Eq 2, surrogate type "1" from reference)
        """
        @staticmethod
        def forward(ctx, v, vth, scale):
            ctx.save_for_backward(v)
            ctx.vth = vth
            ctx.scale = scale
            return (v >= vth).float()

        @staticmethod
        def backward(ctx, grad_output):
            v, = ctx.saved_tensors
            surrogate = ctx.scale / (1.0 + (torch.pi * (v - ctx.vth)).pow(2))
            return grad_output * surrogate, None, None

    def _spike(self, v: torch.Tensor) -> torch.Tensor:
        """Θ(v - V_th) with ArcTan surrogate gradient (target path, Eq 2)."""
        return self._SpikeFunction.apply(v, self.vth, self.surrogate_scale)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Algorithm 1: Trace Propagation.

        Args:
            data:   [T, B, *features]  (time-major, rate or event coded)
            target: [B]                (class indices)

        Returns:
            (loss, pred)  — scalar loss and [B] predicted indices.

        Student path: network.forward() is called at each timestep.
          - spk_rec[l] for l in range(n_blocks) gives the hidden student spikes.
          - _detach_network_hidden() is called before each forward to prevent TBPTT.
          - For VGG9: spk_rec[l] from network.forward() already includes pooling.

        Target path: emulated by the trainer.
          - Uses alpha/vth/ArcTan-surrogate (trainer-controlled, Eq 1).
          - Starts from c* projected via S at layer 0 (Alg 1 line 13-14).
        """
        T = data.size(0)
        B = data.size(1)
        device = data.device

        # One-hot target c* ∈ R^{B×C} (Alg 1 notation)
        c_star = F.one_hot(target, self.n_classes).float()  # [B, C]

        # ------------------------------------------------------------------
        # Reset network and initialize trainer state
        # ------------------------------------------------------------------
        self.network.reset()

        # Target path state (trainer-managed)
        v_t = [None] * self.n_blocks       # target membranes     ṽ_l^t  (pre-pool)
        s_t_raw = [None] * self.n_blocks   # target spikes pre-pool s̃_l^t (for reset term)
        s_t = [None] * self.n_blocks       # target spikes post-pool (input to next layer)
        eps_t = [None] * self.n_blocks     # target traces         ε̃_l^t  (Eq 12)

        # Student eligibility traces — built from network.forward() spikes
        eps_s = [None] * self.n_blocks  # student traces    ε_l^t  (Eq 11)

        # Input-level traces (Alg 1 lines 5-6)
        # x_flat is used ONLY for eps_in; network.forward() handles its own flattening.
        eps_in = torch.zeros(B, data[0].flatten(1).size(1), device=device)
        eps_in_target = torch.zeros(B, self.n_classes, device=device)

        # Output integrator (Sec 3.1): pure accumulator, no decay/threshold
        mem_out = torch.zeros(B, self.n_classes, device=device)

        if self.optimizer:
            self.optimizer.zero_grad()

        # ==================================================================
        # Time loop — Algorithm 1 line 2
        # ==================================================================
        for t in range(T):
            x_t = data[t]                   # [B, *features]
            x_flat = x_t.flatten(1)         # [B, F] — for input trace only

            if self.optimizer:
                self.optimizer.zero_grad()

            # --- Alg 1 lines 5-6: input-level trace update ---
            eps_in = self.beta * eps_in.detach() + x_flat
            eps_in_target = self.beta * eps_in_target.detach() + c_star

            # ----------------------------------------------------------
            # Student path: network forward (Alg 1 lines 7-11)
            # Detach hidden states first to prevent TBPTT through freed graphs.
            # ----------------------------------------------------------
            self._detach_network_hidden()
            spk_rec, mem_rec = self.network(x_t)
            # spk_rec[l] for l in range(n_blocks) = hidden student spikes at layer l.
            # For VGG9: pooling is already applied inside network.forward().

            # ----------------------------------------------------------
            # Target path (Alg 1 lines 12-20) — trainer-emulated
            # ----------------------------------------------------------
            cur_t = c_star  # target path starts with one-hot (Alg 1 line 13)

            for l, (layer, pool_fn) in enumerate(self.blocks):
                # --- Student trace (Eq 11) ---
                spk_s = spk_rec[l]
                eps_prev_s = eps_s[l] if eps_s[l] is not None else torch.zeros_like(spk_s)
                eps_s[l] = self.beta * eps_prev_s.detach() + spk_s

                # --- Target path (Eq 1, Eq 2, Eq 12) ---
                if l == 0:
                    # Alg 1 line 14: W_l = S, s̃_0^t = c*
                    linear_t = self.S(cur_t)   # [B, s_out_size]
                    # For CNN (VGG9): S outputs flat; reshape to match block-0 spatial output.
                    # block 0 has no pooling in VGG9_CFG so spk_rec[0] has the conv output shape.
                    if not isinstance(layer, nn.Linear):
                        linear_t = linear_t.view(B, *spk_rec[0].shape[1:])
                else:
                    # Alg 1 lines 16-19: same W_l as student
                    linear_t = layer(cur_t)

                v_prev_t = v_t[l] if v_t[l] is not None else torch.zeros_like(linear_t)
                # Reset uses the pre-pool spike (same shape as v_t[l]).
                s_prev_raw = s_t_raw[l] if s_t_raw[l] is not None else torch.zeros_like(linear_t)

                v_t[l] = (
                    self.alpha * v_prev_t.detach()
                    + linear_t
                    - self.vth * s_prev_raw.detach()
                )

                spk_t_raw = self._spike(v_t[l])          # pre-pool (for reset term)
                s_t_raw[l] = spk_t_raw
                if pool_fn is not None:
                    spk_t = pool_fn(spk_t_raw)            # post-pool (input to next layer)
                else:
                    spk_t = spk_t_raw
                s_t[l] = spk_t

                # Eq 12: ε̃_l^t = β·ε̃_l^{t-1} + s̃_l^t (use post-pool for trace)
                eps_prev_t = eps_t[l] if eps_t[l] is not None else torch.zeros_like(spk_t)
                eps_t[l] = self.beta * eps_prev_t.detach() + spk_t

                cur_t = spk_t

            # --- Output layer (pure integrator, Sec 3.1) ---
            # mem_out^t = mem_out^{t-1} + W_out · s_{L-1}^t
            # Uses the last hidden block's student spike from network.forward().
            # .flatten(1) is needed for VGG9 where the last conv output is spatial.
            mem_out = mem_out + self.output_layer(spk_rec[self.n_blocks - 1].flatten(1))

            # ==============================================================
            # Contrastive loss + gradient accumulation — Alg 1 lines 22-27
            # ==============================================================
            for l, (layer, _) in enumerate(self.blocks):
                # Eq 14: z_l^t[b,b'] = ε_l^t[b] · ε̃_l^t[b']^T
                h1 = eps_s[l].flatten(1)    # [B, H]  student trace
                t1 = eps_t[l].flatten(1)    # [B, H]  target trace
                z_l = h1 @ t1.t()           # [B, B]  similarity matrix

                # Eq 15: y_l^t soft target (softmax of negative distances)
                t0 = eps_in_target.flatten(1) if l == 0 else eps_t[l - 1].flatten(1)
                dist = (
                    (t0.unsqueeze(1) - t0.unsqueeze(0)).pow(2).sum(-1) + 1e-9
                ).sqrt()                    # [B, B]
                y_l = F.softmax(-dist, dim=1).detach()   # [B, B]

                # Eq 13: E_l^t = -Σ y_l * log softmax(z_l)
                loss_l = torch.sum(
                    -y_l * F.log_softmax(z_l, dim=1), dim=1
                ).mean()

                # Eq 18: ∂E_l/∂W_l via autograd.
                # Student gradient: loss_l → eps_s[l] → spk_rec[l] → network.forward() → layer.weight
                # Target gradient (l>0): loss_l → eps_t[l] → spk_t → v_t[l] → layer(cur_t) → layer.weight
                # Target gradient (l=0): loss_l → eps_t[0] → spk_t → v_t[0] → S(c*) → S.weight
                # Recurrent gradient (RSNN): loss_l → eps_s[l] → spk_rec[l] → rlif(cur) → rlif.recurrent.weight
                #   rlif.spk (prev spike) is detached → gradient uses current-step value only (no TBPTT)
                grad_params = [layer.weight]
                if l == 0 and self.train_target_propagator:
                    grad_params.append(self.S.weight)
                if self.recurrent_weights and l < len(self.recurrent_weights):
                    grad_params.append(self.recurrent_weights[l].weight)

                grads = torch.autograd.grad(
                    loss_l, grad_params, retain_graph=False
                )

                k = 0
                if layer.weight.grad is None:
                    layer.weight.grad = grads[k]
                else:
                    layer.weight.grad += grads[k]
                k += 1

                if l == 0 and self.train_target_propagator:
                    if self.S.weight.grad is None:
                        self.S.weight.grad = grads[k]
                    else:
                        self.S.weight.grad += grads[k]
                    k += 1

                if self.recurrent_weights and l < len(self.recurrent_weights):
                    rw = self.recurrent_weights[l]
                    if rw.weight.grad is None:
                        rw.weight.grad = grads[k]
                    else:
                        rw.weight.grad += grads[k]

            # ==============================================================
            # Output layer: delta rule (Sec 3.1)
            # ΔW_out = (softmax(mem_out^t) - c*) ⊗ ε_{L-1}^t  / B
            # ==============================================================
            err = F.softmax(mem_out, dim=1) - c_star                 # [B, C]
            grad_out = err.t() @ eps_s[-1].flatten(1) / B            # [C, H_{L-1}]

            if self.output_layer.weight.grad is None:
                self.output_layer.weight.grad = grad_out
            else:
                self.output_layer.weight.grad += grad_out

            # --- Alg 1 line 28: W^{t+1} = W^t - η·ΔW ---
            if self.optimizer:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # ==================================================================
        # End of time loop — compute final loss/pred for logging
        # ==================================================================
        with torch.no_grad():
            final_loss = F.cross_entropy(mem_out, target)
            pred = mem_out.argmax(dim=1)

        return final_loss, pred

    def reset(self):
        """Reset network state (snntorch hidden state) between sequences."""
        self.network.reset()

    def to(self, device):
        super().to(device)
        self.S = self.S.to(device)
        return self
