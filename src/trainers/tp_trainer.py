"""
Trace Propagation (TP) Trainer.

Supports FCSNN, RSNN, and generic CNN architectures (e.g. ConvSNN, VGG9) in a single unified class.

Architecture detection (duck-typing):
  FCSNN        → hasattr(network, 'synapses')
  RSNN         → hasattr(network, 'input_layers')
  ConvSNN      → hasattr(network, 'conv1') + hasattr(network, 'fc') + hasattr(network, 'lif_out')
                 pool_before_spike=True  (conv → pool → LIF ordering)
  VGG9 (CNN)   → hasattr(network, 'conv1') + hasattr(network, 'head')
                 pool_before_spike=False (conv → LIF → pool ordering)

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

Design note — contrastive loss for CNN layers (tp_cnn.py reference):
  For conv layers the traces have shape [B, C, H, W].  The reference
  implementation computes one [B,B] similarity matrix per spatial position
  and averages (spatial-mean form):
    z_l = mean_s ( ε_l[:,:,s] @ ε̃_l[:,:,s]^T )
  Magnitude ≈ C (channels), not C*H*W — keeping softmax responsive to
  inter-sample structure.  Flattening everything to [B, C*H*W] makes the
  diagonal dwarf all off-diagonal entries and kills the gradient signal.

Design note — batch size constraint (paper Sec 3.2):
  TP requires batch_size >= 2.  The contrastive loss (Eq 13-15) computes a [B,B]
  inter-sample similarity matrix z_l = ε_l @ ε̃_l^T.  With B=1 this is a 1×1
  scalar, log_softmax([x]) = 0 always, and all gradients vanish — learning is
  impossible.  This is a fundamental algorithmic requirement, not a limitation of
  this implementation.
"""

from typing import List, Optional, Tuple

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks._components import LeakyIntegrator
from trainers.base_trainer import BaseTrainer

# -----------------------------------------------------------------------------
# Supported layer types for the CNN probe.
# To add support for a new layer kind, extend the appropriate set and add a
# probe rule in `_probe_pool_placement` / `_find_output_layer`.
# -----------------------------------------------------------------------------
_SUPPORTED_WEIGHT_TYPES = (nn.Conv2d, nn.Linear)
_SUPPORTED_POOL_TYPES = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)
_SUPPORTED_SPIKE_TYPES = (snn.Leaky,)


class TPTrainer(BaseTrainer):
    """
    Unified Trace Propagation trainer for FCSNN, RSNN, ConvSNN, and VGG9-based CNNs.

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
        self._input_shape = tuple(input_shape) if input_shape is not None else None

        # Extract learnable weight blocks and output layer from the network.
        # For CNN networks, layout (block list + pool placement) is discovered
        # by tracing a dummy forward — no hand-written branches per architecture.
        self._extract_layers()

        # Initialize projection matrix S (Alg 1 line 14: for l=1, W_l = S)
        # S projects the one-hot target c* ∈ R^C to the first hidden layer's space
        self.S = self._init_S()

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
        Build self.blocks, self.output_layer, self.recurrent_weights, self.pool_before_spike.

        Strategy:
          - FC and recurrent networks expose their layer order explicitly
            (synapses / input_layers + recurrent_layers) → use it directly.
          - CNN networks are discovered by walking conv{i}/pool{i} attributes
            in numerical order and probing pool placement (before/after spike)
            via forward-hook call ordering on a single dummy forward.
            See _probe_pool_placement and _find_output_layer.

        blocks: list of (weight_layer, pool_fn_or_None) for each hidden block.
          - weight_layer is nn.Linear for FC/Rec, nn.Conv2d for CNN.
          - pool_fn is None for FC/Rec; MaxPool/AvgPool/AdaptiveAvgPool for CNN.
        output_layer: the final nn.Linear (used as a pure integrator during training).
        recurrent_weights: list of nn.Linear recurrent modules (RSNN only; empty otherwise).
        pool_before_spike: True for ConvSNN-style (conv → pool → LIF);
          False for VGG9-style (conv → LIF → pool). Single uniform value across blocks.
        """
        if hasattr(self.network, 'synapses'):          # FCSNN
            self.blocks = [(w, None) for w in self.network.synapses[:-1]]
            self.output_layer = self.network.synapses[-1]
            self.recurrent_weights: List[nn.Linear] = []
            self.pool_before_spike = False
            self._validate_integrator_output()

        elif hasattr(self.network, 'input_layers'):    # RSNN
            # Recurrent weights (rlif.recurrent) are used inside network.forward().
            # _detach_network_hidden() detaches rlif.spk before each timestep so
            # the gradient for recurrent.weight uses the current-step spike only (no TBPTT).
            self.blocks = [(w, None) for w in self.network.input_layers]
            self.output_layer = self.network.fc_out
            self.recurrent_weights = [
                rlif.recurrent for rlif in self.network.recurrent_layers
            ]
            self.pool_before_spike = False
            self._validate_integrator_output()

        elif hasattr(self.network, 'conv1'):           # CNN family — probe
            self.recurrent_weights = []
            self._probe_cnn()

        else:
            raise ValueError(
                "TPTrainer: unsupported network. Expected one of:\n"
                "  - FCSNN: exposes `synapses` (ModuleList of nn.Linear).\n"
                "  - RSNN: exposes `input_layers`, `recurrent_layers`, `fc_out`.\n"
                "  - CNN: exposes `conv1`, `conv2`, ... and (optionally) "
                "`pool1`, `pool2`, ... in forward order, plus an integrator-style "
                "output (LeakyIntegrator head, or `lif_out` with out_integrator=True)."
            )

        self.n_blocks = len(self.blocks)
        self.n_classes = self.output_layer.out_features

    # -------------------------------------------------------------------------
    # CNN probe — discover blocks, output, and pool placement from a vanilla net.
    # -------------------------------------------------------------------------

    def _probe_cnn(self):
        """
        Discover the CNN block list, output layer, and pool placement by tracing
        a dummy forward through `network`. No config required.
        """
        # 1. Block list: walk conv{i}/pool{i} attributes in numerical order.
        self.blocks = []
        i = 1
        while hasattr(self.network, f'conv{i}'):
            conv = getattr(self.network, f'conv{i}')
            if not isinstance(conv, _SUPPORTED_WEIGHT_TYPES):
                raise ValueError(
                    f"TPTrainer: conv{i} is of type {type(conv).__name__}; "
                    f"expected one of {[t.__name__ for t in _SUPPORTED_WEIGHT_TYPES]}."
                )
            pool = getattr(self.network, f'pool{i}', None)
            if pool is not None and not isinstance(pool, _SUPPORTED_POOL_TYPES):
                raise ValueError(
                    f"TPTrainer: pool{i} is of type {type(pool).__name__}; "
                    f"expected one of {[t.__name__ for t in _SUPPORTED_POOL_TYPES]}."
                )
            self.blocks.append((conv, pool))
            i += 1
        if not self.blocks:
            raise ValueError(
                "TPTrainer: no `conv{i}` attributes found on network. "
                "CNN networks must expose conv1, conv2, ... in forward order."
            )

        # 2. Output layer.
        self.output_layer = self._find_output_layer()

        # 3. Pool placement (uniform single bool).
        self.pool_before_spike = self._probe_pool_placement()

        # 4. Output must be integrator-compatible (TP Sec 3.1).
        self._validate_integrator_output()

    def _find_output_layer(self) -> nn.Linear:
        """
        Locate the trainable output linear layer for TP. Reject network shapes
        that TP cannot model.
        """
        net = self.network
        # VGG9 TP-style: head is a LeakyIntegrator wrapping nn.Linear.
        head = getattr(net, 'head', None)
        if head is not None:
            if isinstance(head, LeakyIntegrator):
                return head.fc
            if isinstance(head, nn.Linear):
                # OTTT-style: plain Linear preceded by global pool. TP requires
                # an integrator-style output; reject explicitly.
                raise ValueError(
                    "TPTrainer: network exposes a plain nn.Linear `head` "
                    "(e.g. OTTT-style VGG9 with head_type='global_linear'). "
                    "TP requires an integrator-style output (paper Sec 3.1). "
                    "Use head_type='leaky_integrator' instead."
                )
            raise ValueError(
                f"TPTrainer: unsupported `head` module type {type(head).__name__}. "
                "Supported: LeakyIntegrator (with .fc)."
            )
        # ConvSNN-style: separate fc + lif_out modules.
        if hasattr(net, 'fc') and hasattr(net, 'lif_out'):
            if not isinstance(net.fc, nn.Linear):
                raise ValueError(
                    f"TPTrainer: network.fc is {type(net.fc).__name__}; expected nn.Linear."
                )
            return net.fc
        # RSNN-style: fc_out (handled in _extract_layers, but kept here for completeness).
        if hasattr(net, 'fc_out') and isinstance(net.fc_out, nn.Linear):
            return net.fc_out
        raise ValueError(
            "TPTrainer: cannot identify output layer. Supported patterns: "
            "VGG9-style `head` of type LeakyIntegrator, ConvSNN-style "
            "`fc`+`lif_out`, or RSNN-style `fc_out`."
        )

    def _probe_pool_placement(self) -> bool:
        """
        Determine whether pools fire BEFORE or AFTER the spike on a per-block
        basis by inspecting forward-hook call ordering during a single dummy
        forward. Returns one uniform bool (raises on mixed placement).

        - pool_before_spike=True  → conv → pool → lif  (e.g. ConvSNN)
        - pool_before_spike=False → conv → lif → pool  (e.g. VGG9 TP-style)
        """
        # If no block has a pool, the value is irrelevant — pick False.
        if not any(p is not None for _, p in self.blocks):
            return False

        block_convs = {conv: idx for idx, (conv, _) in enumerate(self.blocks)}
        block_pools = {p for _, p in self.blocks if p is not None}

        events: List[Tuple[str, nn.Module]] = []

        def make_hook(kind):
            def hook(_mod, _inp, _out):
                events.append((kind, _mod))
            return hook

        handles = []
        try:
            for m in self.network.modules():
                if m in block_convs:
                    handles.append(m.register_forward_hook(make_hook('conv')))
                elif m in block_pools:
                    handles.append(m.register_forward_hook(make_hook('pool')))
                elif isinstance(m, _SUPPORTED_SPIKE_TYPES):
                    handles.append(m.register_forward_hook(make_hook('lif')))

            input_shape = self._resolve_input_shape()
            dev = next(self.network.parameters()).device
            dummy = torch.zeros(1, *input_shape, device=dev)
            self.network.reset()
            with torch.no_grad():
                self.network(dummy)
            self.network.reset()
        finally:
            for h in handles:
                h.remove()

        # Walk events: for each conv that belongs to a block, scan forward until
        # the next conv-in-block (or end), recording first 'pool' and first 'lif'.
        decisions: List[Optional[bool]] = []
        n = len(events)
        i = 0
        while i < n:
            kind, mod = events[i]
            if kind != 'conv' or mod not in block_convs:
                i += 1
                continue
            block_idx = block_convs[mod]
            # Skip blocks without a pool (decision irrelevant).
            if self.blocks[block_idx][1] is None:
                i += 1
                continue
            pool_pos: Optional[int] = None
            lif_pos: Optional[int] = None
            j = i + 1
            while j < n and not (events[j][0] == 'conv' and events[j][1] in block_convs):
                if events[j][0] == 'pool' and events[j][1] is self.blocks[block_idx][1] and pool_pos is None:
                    pool_pos = j
                elif events[j][0] == 'lif' and lif_pos is None:
                    lif_pos = j
                j += 1
            if pool_pos is not None and lif_pos is not None:
                decisions.append(pool_pos < lif_pos)
            elif pool_pos is not None and lif_pos is None:
                # Pool fired but no LIF before next block — unusual; treat as before-spike.
                decisions.append(True)
            i = j if j > i else i + 1

        valid = [d for d in decisions if d is not None]
        if not valid:
            return False
        if any(valid) and not all(valid):
            raise ValueError(
                "TPTrainer: mixed pool placement detected — some blocks pool "
                "before spike (conv→pool→LIF) and others after (conv→LIF→pool). "
                "TP target-path replay requires a single uniform placement."
            )
        return valid[0]

    def _validate_integrator_output(self):
        """
        Verify the network's output is integrator-style (no spike, no decay)
        as required by TP Sec 3.1. Accepts:
          - LeakyIntegrator head (always integrator-style by construction).
          - snn.Leaky `lif_out` with threshold ≥ 1e6 (never fires) and beta ≈ 1.
          - Networks that explicitly set self.out_integrator = True.
        """
        net = self.network
        # 1. LeakyIntegrator head — always OK.
        if isinstance(getattr(net, 'head', None), LeakyIntegrator):
            return
        # 2. Explicit flag wins (mirrors FCSNN/RSNN/ConvSNN out_integrator pattern).
        if bool(getattr(net, 'out_integrator', False)):
            return
        # 3. Inspect lif_out (or final neuron in `neurons`) directly.
        out_neuron = None
        if hasattr(net, 'lif_out') and isinstance(net.lif_out, _SUPPORTED_SPIKE_TYPES):
            out_neuron = net.lif_out
        elif hasattr(net, 'neurons') and len(net.neurons) > 0:
            cand = net.neurons[-1]
            if isinstance(cand, _SUPPORTED_SPIKE_TYPES):
                out_neuron = cand
        if out_neuron is not None:
            thr = float(getattr(out_neuron, 'threshold', 1.0))
            beta = getattr(out_neuron, 'beta', 1.0)
            if torch.is_tensor(beta):
                beta = float(beta.detach().mean().item())
            else:
                beta = float(beta)
            if thr >= 1e6 and beta >= 0.999:
                return
            raise ValueError(
                "TPTrainer requires an integrator-style output (TP Sec 3.1) — "
                f"the output neuron has threshold={thr:g}, beta={beta:g}. "
                "Construct the network with out_integrator=True (sets beta=1.0, "
                "threshold=1e9) so it accumulates without firing."
            )
        # 4. No recognizable output neuron and no flag → reject.
        raise ValueError(
            "TPTrainer: cannot verify the network output is integrator-style. "
            "Supported markers: LeakyIntegrator head, network.out_integrator=True, "
            "or snn.Leaky lif_out with threshold≥1e6 and beta≈1."
        )

    def _resolve_input_shape(self) -> Tuple[int, ...]:
        """Resolve the dummy-forward input shape from constructor arg or net attrs."""
        if self._input_shape is not None:
            return tuple(self._input_shape)
        for attr in ('input_shape', 'in_shape'):
            s = getattr(self.network, attr, None)
            if s is not None:
                return tuple(s)
        raise ValueError(
            "TPTrainer: probing requires input_shape=(C, H, W). Pass it via "
            "TPTrainer(..., input_shape=(C,H,W)) or expose network.input_shape "
            "/ network.in_shape."
        )

    def _init_S(self) -> nn.Linear:
        """
        Initialize projection matrix S ∈ R^{C × H_1}.
        S projects the one-hot target c* from C classes to the first hidden
        layer's feature space (Alg 1 lines 13-14, Fig 2).
        """
        layer, pool_fn = self.blocks[0]

        if isinstance(layer, nn.Linear):
            s_out_size = layer.out_features
        else:
            input_shape = self._resolve_input_shape()
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

        elif hasattr(self.network, 'conv1'):           # ConvSNN or VGG9
            # init_hidden=True: snntorch stores membrane inside lif{i}.mem.
            for i in range(1, self.n_blocks + 1):
                lif = getattr(self.network, f'lif{i}')
                if lif.mem is not None:
                    lif.mem = lif.mem.detach()
            if hasattr(self.network, 'head') and hasattr(self.network.head, 'mem'):
                if self.network.head.mem is not None:
                    self.network.head.mem = self.network.head.mem.detach()
            elif hasattr(self.network, 'lif_out') and self.network.lif_out.mem is not None:
                self.network.lif_out.mem = self.network.lif_out.mem.detach()

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
            # Pooling is already applied inside network.forward() for all CNN architectures.

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
                    # For CNN: S outputs flat; reshape to match block-0 student spike shape.
                    # The reshape already aligns spatial dimensions — pool_fn is NOT applied
                    # here regardless of pool_before_spike, preventing double-pooling.
                    if not isinstance(layer, nn.Linear):
                        linear_t = linear_t.view(B, *spk_rec[0].shape[1:])
                else:
                    # Alg 1 lines 16-19: same W_l as student.
                    # pool_before_spike=True (ConvSNN): pool is part of the effective weight
                    #   (conv → pool → LIF), so apply pool_fn to conv output before v_t.
                    # pool_before_spike=False (VGG9): pool is applied to spikes after threshold,
                    #   so compute v_t from raw conv output and pool spk_t_raw below.
                    if pool_fn is not None and self.pool_before_spike:
                        linear_t = pool_fn(layer(cur_t))  # pool before membrane (ConvSNN-style)
                    else:
                        linear_t = layer(cur_t)           # pool after spike or no pool

                v_prev_t = v_t[l] if v_t[l] is not None else torch.zeros_like(linear_t)
                s_prev_raw = s_t_raw[l] if s_t_raw[l] is not None else torch.zeros_like(linear_t)

                v_t[l] = (
                    self.alpha * v_prev_t.detach()
                    + linear_t
                    - self.vth * s_prev_raw.detach()
                )

                spk_t_raw = self._spike(v_t[l])
                s_t_raw[l] = spk_t_raw
                # Pool after spike only for VGG9-style (l>0); l=0 is handled by the reshape above.
                if pool_fn is not None and not self.pool_before_spike and l > 0:
                    spk_t = pool_fn(spk_t_raw)            # post-spike pool (VGG9-style)
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
                # For conv layers (dim>2) use the spatial-mean form from the
                # paper's reference implementation (tp_cnn.py:loss): compute a
                # separate [B,B] matrix per spatial position and average.
                # This keeps per-position magnitudes O(C) rather than O(C*H*W),
                # preventing softmax from collapsing onto the self-similarity
                # diagonal and killing inter-sample gradient signal.
                if eps_s[l].dim() > 2:
                    # [B, C, H, W] → [H*W, B, C] and [H*W, C, B]
                    h1 = eps_s[l].flatten(2).permute(2, 0, 1)   # [S, B, C]
                    t1 = eps_t[l].flatten(2).permute(2, 1, 0)   # [S, C, B]
                    z_l = (h1 @ t1).mean(0)                      # [B, B]
                else:
                    h1 = eps_s[l]                                # [B, H]
                    t1 = eps_t[l]                                # [B, H]
                    z_l = h1 @ t1.t()                            # [B, B]

                # Eq 15: y_l^t soft target (softmax of negative distances)
                # For spatial tensors (conv layers) use spatial-mean distance
                # (same as z_l): compute pairwise distance per spatial position
                # and average.
                # Memory-efficient form: ||a-b||² = ||a||² + ||b||² - 2<a,b>
                # avoids materializing the [S,B,B,C] intermediate from the
                # naive expansion (t0.unsqueeze(2)-t0.unsqueeze(1)).pow(2).sum(-1),
                # which can reach 4+ GB at B=128.  Peak memory is now [S,B,B].
                # .clamp(min=0) guards against tiny negatives from FP cancellation.
                t0_raw = eps_in_target if l == 0 else eps_t[l - 1]
                if t0_raw.dim() > 2:
                    t0 = t0_raw.flatten(2).permute(2, 0, 1)        # [S, B, C]
                    norms = t0.pow(2).sum(-1)                       # [S, B]
                    cross = t0 @ t0.transpose(-1, -2)               # [S, B, B]
                    dist = (norms.unsqueeze(-1) + norms.unsqueeze(-2) - 2 * cross).clamp(min=0).sqrt().mean(0)  # [B, B]
                else:
                    t0 = t0_raw                                     # [B, F]
                    norms = t0.pow(2).sum(-1)                       # [B]
                    cross = t0 @ t0.t()                             # [B, B]
                    dist = (norms.unsqueeze(1) + norms.unsqueeze(0) - 2 * cross).clamp(min=0).sqrt()           # [B, B]
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
