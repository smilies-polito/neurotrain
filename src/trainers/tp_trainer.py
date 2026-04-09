"""
Trace Propagation (TP) Trainer.

Implements Algorithm 1 from:
    Pes et al. (2026) - "Traces propagation: memory-efficient and scalable
    forward-only learning in spiking neural networks"

Supports FCSNN, RSNN, and DVSGEST_VGG9 in a single unified class.

Architecture detection (duck-typing):
  FCSNN        → hasattr(network, 'synapses')
  RSNN         → hasattr(network, 'input_layers')
  DVSGEST_VGG9 → hasattr(network, 'VGG9_CFG')

Design note — network/trainer separation:
  The trainer maintains its OWN membrane potentials, spikes, and eligibility
  traces for both the student (green) and target (purple) paths.
  The network is used as a weight container only; its snntorch state is
  never touched during train_sample().

Design note — output layer:
  Paper Sec 3.1: all architectures use a simple integrator as output
  (no threshold, no spike, leak=1.0): mem_out^t = mem_out^{t-1} + W_out*s_{L-1}^t.
  The trainer always does this, regardless of the network's last snntorch layer.

Design note — evaluation with snntorch networks:
  During eval the test loop calls network(data[t]) and accumulates spk_rec[-1].
  For VGG9: spk_rec[-1] IS the membrane (LeakyIntegrator returns mem, no spikes) — exact.
  For FCSNN/RSNN: spk_rec[-1] is spike count. Since neurons fire in proportion to
  their input, argmax(spike_sum) ≈ argmax(integrated membrane). This is standard
  SNN practice and works reliably in practice.
  Using snn.Leaky with a very high threshold as a leaky integrator would require
  different thresholds per layer (hidden must fire, output must not), which
  FCSNN/RSNN don't support directly. Hence we use the spike-count approximation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class TPTrainer(BaseTrainer):
    """
    Unified Trace Propagation trainer for FCSNN, RSNN, and DVSGEST_VGG9.

    Args:
        network:                  The SNN (weight container during training).
        lr:                       Learning rate for SGD.
        batch_size:               Must be >= 2 (contrastive loss requires pairs).
        alpha:                    Membrane decay (α in paper Eq 1).
        beta:                     Eligibility trace decay (β in paper Eq 11-12).
        vth:                      Spike threshold (V_th in paper).
        surrogate_scale:          ArcTan surrogate gradient scale (paper uses 1.0).
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
            raise ValueError("TP requires batch_size >= 2 for contrastive loss.")

        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = alpha           # α: membrane decay (Eq 1)
        self.beta = beta             # β: trace decay (Eq 11, 12)
        self.vth = vth               # V_th: spike threshold (Eq 2)
        self.surrogate_scale = surrogate_scale
        self.train_target_propagator = train_target_propagator

        # Extract learnable weight blocks and output layer from the network
        self._extract_layers()

        # Initialize projection matrix S (Alg 1 line 14: for l=1, W_l = S)
        # S projects the one-hot target c* ∈ R^C to the first hidden layer's space
        self.S = self._init_S(input_shape)

        # Optimizer: includes all network params + S
        if optimizer is not None:
            self.optimizer = optimizer
        elif use_optimizer:
            params = list(self.network.parameters())
            if self.train_target_propagator:
                params.append(self.S.weight)
            if self.recurrent_weights:
                params.extend(w.weight for w in self.recurrent_weights)
            self.optimizer = torch.optim.SGD(params, lr=lr)
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
        output_layer: the final nn.Linear (pure integrator during training).
        recurrent_weights: list of nn.Linear (RSNN only; empty otherwise).
        """
        if hasattr(self.network, 'synapses'):          # FCSNN
            # synapses = [W_1, W_2, ..., W_L]
            # hidden blocks: all but the last synapse
            self.blocks = [(w, None) for w in self.network.synapses[:-1]]
            self.output_layer = self.network.synapses[-1]
            self.recurrent_weights = []

        elif hasattr(self.network, 'input_layers'):    # RSNN
            # input_layers = [W_1, ..., W_{L-1}]  (hidden linear layers)
            # recurrent_layers = [RLeaky_1, ..., RLeaky_{L-1}]
            #   each RLeaky carries .recurrent (nn.Linear) for R_l (Eq 1)
            self.blocks = [(w, None) for w in self.network.input_layers]
            self.output_layer = self.network.fc_out
            self.recurrent_weights = [
                rlif.recurrent for rlif in self.network.recurrent_layers
            ]

        elif hasattr(self.network, 'VGG9_CFG'):        # DVSGEST_VGG9
            # Each VGG block: (WSConv2d, optional pool)
            self.blocks = []
            for i, (_, _, _) in enumerate(self.network.VGG9_CFG, start=1):
                conv = getattr(self.network, f'conv{i}')
                pool = getattr(self.network, f'pool{i}', None)
                self.blocks.append((conv, pool))
            # Output: fc inside the LeakyIntegrator head
            self.output_layer = self.network.head.fc
            self.recurrent_weights = []

        else:
            raise ValueError(
                "TPTrainer supports FCSNN (synapses), RSNN (input_layers), "
                "or DVSGEST_VGG9 (VGG9_CFG). Unknown network type."
            )

        self.n_blocks = len(self.blocks)
        # Derive n_classes from the output layer's out_features
        self.n_classes = self.output_layer.out_features

    def _init_S(self, input_shape: Optional[Tuple]) -> nn.Linear:
        """
        Initialize projection matrix S ∈ R^{C × H_1}.
        S projects the one-hot target c* from C classes to the first hidden
        layer's feature space (Alg 1 lines 13-14, Fig 2).
        """
        layer, pool_fn = self.blocks[0]

        if isinstance(layer, nn.Linear):
            # FC / Recurrent: output size is known directly
            s_out_size = layer.out_features
        else:
            # CNN: output size depends on spatial dimensions of first conv block
            if input_shape is None:
                raise ValueError(
                    "TPTrainer requires input_shape=(C, H, W) for CNN networks "
                    "(e.g. input_shape=(2, 128, 128) for DVSGesture)."
                )
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                out = layer(dummy)                            # (1, C, H, W)
                if pool_fn is not None:
                    out = pool_fn((out >= self.vth).float())  # apply pool shape
            s_out_size = out[0].numel()

        S = nn.Linear(self.n_classes, s_out_size, bias=False)
        nn.init.kaiming_normal_(S.weight)
        return S

    # -------------------------------------------------------------------------
    # Spike function with ArcTan surrogate gradient
    # -------------------------------------------------------------------------

    class _SpikeFunction(torch.autograd.Function):
        """
        Forward: hard threshold Θ(v - V_th).
        Backward: ArcTan surrogate  scale / (1 + (π·(v - V_th))²).
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
        """Θ(v - V_th) with ArcTan surrogate gradient (Eq 2)."""
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
        """
        T = data.size(0)
        B = data.size(1)
        device = data.device

        # One-hot target c* ∈ R^{B×C} (Alg 1 notation)
        c_star = F.one_hot(target, self.n_classes).float()  # [B, C]

        # ------------------------------------------------------------------
        # Initialize state (zero at start of each sequence)
        # ------------------------------------------------------------------
        # Membrane potentials v_l^t (Eq 1) and target ṽ_l^t
        # Shapes: 2D (B, H) for FC/Rec; 4D (B, C, H, W) for CNN.
        # We init as zeros and let the first forward call define the shape.

        v_s = [None] * self.n_blocks   # student membranes
        v_t = [None] * self.n_blocks   # target membranes
        s_s = [None] * self.n_blocks   # student spikes  s_l^t  (Eq 2)
        s_t = [None] * self.n_blocks   # target spikes   s̃_l^t
        eps_s = [None] * self.n_blocks  # student traces ε_l^t  (Eq 11)
        eps_t = [None] * self.n_blocks  # target traces  ε̃_l^t  (Eq 12)

        # Input-level traces (Alg 1 lines 5-6)
        # eps_in:        ε_0^t = β*ε_0^{t-1} + x_t
        # eps_in_target: ε̃_0^t = β*ε̃_0^{t-1} + c*
        eps_in = torch.zeros(B, data[0].flatten(1).size(1), device=device)
        eps_in_target = torch.zeros(B, self.n_classes, device=device)

        # Output integrator mem_out (Sec 3.1): pure leak=1, no threshold
        mem_out = torch.zeros(B, self.n_classes, device=device)

        if self.optimizer:
            self.optimizer.zero_grad()

        # ==================================================================
        # Time loop — Algorithm 1 line 2
        # ==================================================================
        for t in range(T):
            x_t = data[t]                        # [B, *features]
            x_flat = x_t.flatten(1)              # [B, F] for input trace

            if self.optimizer:
                self.optimizer.zero_grad()

            # --- Alg 1 lines 5-6: input-level trace update ---
            # ε_0^t = β·ε_0^{t-1} + x_t
            eps_in = self.beta * eps_in.detach() + x_flat
            # ε̃_0^t = β·ε̃_0^{t-1} + c*
            eps_in_target = self.beta * eps_in_target.detach() + c_star

            # ----------------------------------------------------------
            # Forward loop — Alg 1 lines 7-20
            # ----------------------------------------------------------
            # FC/Rec: first layer is nn.Linear → needs flat [B, F] input
            # CNN: first layer is Conv2d → needs spatial [B, C, H, W] input
            cur_s = x_flat if isinstance(self.blocks[0][0], nn.Linear) else x_t
            cur_t = c_star   # target path always starts with one-hot (Alg 1 line 13)

            for l, (layer, pool_fn) in enumerate(self.blocks):
                rec_w = self.recurrent_weights[l] if self.recurrent_weights else None

                # ---- Student path (green) ----
                # Eq 1: v_l^t = α·v_l^{t-1} + W_l·s_{l-1}^t + [R_l·s_l^{t-1}] - V_th·s_l^{t-1}
                linear_s = layer(cur_s)
                if rec_w is not None:
                    # recurrent term: R_l · s_l^{t-1}  (Eq 1, only RSNN)
                    s_prev = s_s[l] if s_s[l] is not None else torch.zeros_like(linear_s)
                    linear_s = linear_s + rec_w(s_prev.detach())

                v_prev_s = v_s[l] if v_s[l] is not None else torch.zeros_like(linear_s)
                s_prev_s = s_s[l] if s_s[l] is not None else torch.zeros_like(linear_s)

                v_s[l] = (
                    self.alpha * v_prev_s.detach()
                    + linear_s
                    - self.vth * s_prev_s.detach()          # soft reset
                )

                spk_s = self._spike(v_s[l])                  # Eq 2: s_l^t = Θ(v_l^t - V_th)
                if pool_fn is not None:
                    spk_s = pool_fn(spk_s)
                s_s[l] = spk_s

                # Eq 11: ε_l^t = β·ε_l^{t-1} + s_l^t
                eps_prev_s = eps_s[l] if eps_s[l] is not None else torch.zeros_like(spk_s)
                eps_s[l] = self.beta * eps_prev_s.detach() + spk_s

                # ---- Target path (purple) ----
                # Alg 1 lines 12-15: for l=0, use S instead of W_l and c* as input
                if l == 0:
                    # Alg 1 line 14: W_l = S, s̃_0^t = c*
                    linear_t = self.S(cur_t)   # [B, H1] for FC or [B, C1*H1*W1] for CNN
                    # For CNN: S outputs flat vector; reshape to match spatial linear_s
                    if not isinstance(layer, nn.Linear):
                        linear_t = linear_t.view_as(linear_s)
                else:
                    # Alg 1 lines 16-19: same W_l as student
                    linear_t = layer(cur_t)

                v_prev_t = v_t[l] if v_t[l] is not None else torch.zeros_like(linear_t)
                s_prev_t = s_t[l] if s_t[l] is not None else torch.zeros_like(linear_t)

                v_t[l] = (
                    self.alpha * v_prev_t.detach()
                    + linear_t
                    - self.vth * s_prev_t.detach()
                )

                spk_t = self._spike(v_t[l])
                if pool_fn is not None:
                    spk_t = pool_fn(spk_t)
                s_t[l] = spk_t

                # Eq 12: ε̃_l^t = β·ε̃_l^{t-1} + s̃_l^t
                eps_prev_t = eps_t[l] if eps_t[l] is not None else torch.zeros_like(spk_t)
                eps_t[l] = self.beta * eps_prev_t.detach() + spk_t

                cur_s = spk_s    # next layer input
                cur_t = spk_t

            # --- Output layer (pure integrator, Sec 3.1) ---
            # mem_out^t = mem_out^{t-1} + W_out · s_{L-1}^t
            # No alpha, no threshold, no reset — this is the key fix vs previous code
            mem_out = mem_out + self.output_layer(cur_s)

            # ==============================================================
            # Contrastive loss + gradient accumulation — Alg 1 lines 22-27
            # ==============================================================
            for l, (layer, _) in enumerate(self.blocks):
                # Eq 14: z_l^t[b,b'] = ε_l^t[b] · ε̃_l^t[b']^T
                h1 = eps_s[l].flatten(1)    # [B, H]  student trace
                t1 = eps_t[l].flatten(1)    # [B, H]  target trace
                z_l = h1 @ t1.t()           # [B, B]  similarity matrix

                # Eq 15: y_l^t[b,b'] = softmax(-dist(ε̃_{l-1}^t[b], ε̃_{l-1}^t[b']))
                t0 = eps_in_target.flatten(1) if l == 0 else eps_t[l - 1].flatten(1)  # [B, F]
                # Euclidean distance (reference tp_mlp.py line 226)
                dist = (
                    (t0.unsqueeze(1) - t0.unsqueeze(0)).pow(2).sum(-1) + 1e-9
                ).sqrt()                    # [B, B]
                y_l = F.softmax(-dist, dim=1).detach()   # [B, B]  soft target

                # Eq 13: E_l^t = -Σ y_l * log softmax(z_l)
                loss_l = torch.sum(
                    -y_l * F.log_softmax(z_l, dim=1), dim=1
                ).mean()

                # Eq 18: ∂E_l/∂W_l via autograd
                # For l>0: both student and target paths go through layer.weight,
                # so autograd gives ΔW^{in} + ΔW^{trg} in one call (Eq 18).
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
