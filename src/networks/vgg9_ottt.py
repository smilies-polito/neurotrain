"""
Spiking VGG-9 faithfully replicating OTTT-SNN (Xiao et al., NeurIPS 2022).

Source: temp_repos/OTTT-SNN/models/spiking_vgg.py, cfg 'A'
        temp_repos/OTTT-SNN/spikingjelly_codes/reference_codes/spiking_vggws_ottt.py
        (light_classifier=True, fc_hw=1 — the default)

Architecture (DVSGesture config):
  8 conv blocks [64,128,256,256,512,512,512,512], AvgPool after blocks 2,4,6.
  Each block: ScaledWSConv2d(bias=True) -> LIF -> Scale(2.74).
  Classifier: AdaptiveAvgPool2d(1,1) -> Linear(512, C) — no LIF in head.

Neuron defaults (from OnlineLIFNode, tau=2):
  beta = 1-1/tau = 0.5, threshold = 1.0, subtract reset, sigmoid surrogate (slope=4).

Init: kaiming_normal for conv, normal_(0, 0.01) for linear, bias = 0.

forward() processes a single timestep (B, C, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


# ---------------------------------------------------------------------------
# Weight Standardization Conv2d  (identical to OTTT-SNN ScaledWSConv2d)
# ---------------------------------------------------------------------------

class ScaledWSConv2d(nn.Conv2d):
    """
    Conv2d layer with Weight Standardization and learnable gain.
    
    Inherits from nn.Conv2d, so it can be used as a drop-in replacement.
    Overrides the forward() method to apply weight standardization before convolution.
    """

    def __init__(self, *args, gain=True, gain_init=1.8, eps=1e-4, **kwargs):
        """
        Initialize ScaledWSConv2d.

        Args:
            *args: Positional arguments passed to nn.Conv2d (in_channels, out_channels, kernel_size, ...)
            gain (bool): If True, create a learnable gain parameter per output channel.
                         If False, no gain is applied (fixed normalization).
            gain_init (float): Initial value for the learnable gain. Default 1.8 compensates for
                               WS signal suppression (WS normalizes output to ~1/sqrt(fan_in),
                               so gain_init=1.8 restores signal magnitude to a level where LIF
                               neurons can realistically reach threshold). Matches TP's fixed scale.
            eps (float): Small constant added to variance for numerical stability.
                         Prevents division by zero if variance is very small.
            **kwargs: Keyword arguments passed to nn.Conv2d (stride, padding, dilation, groups, bias, etc.)
        """
        super().__init__(*args, **kwargs)
        if gain:
            self.gain = nn.Parameter(
                torch.full((self.out_channels, 1, 1, 1), gain_init)
            )
        else:
            self.gain = None
        self.eps = eps

    def forward(self, x):
        """
        Forward pass with weight standardization.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output of convolution with shape (batch_size, out_channels, out_height, out_width)
        """
        # Calculate fan-in: number of input connections per output filter
        # self.weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        # fan_in = in_channels * kernel_h * kernel_w
        # This is the total number of inputs feeding into each output channel
        fan_in = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        # Compute mean and variance of weights for each output channel
        # Computed across input channels [1] and spatial dimensions [2, 3]
        # Result shape: (out_channels, 1, 1, 1)
        # keepdim=True maintains dimensions for broadcasting
        mean = self.weight.mean(dim=[1, 2, 3], keepdim=True)
        var = self.weight.var(dim=[1, 2, 3], keepdim=True)
        # Standardize weights using Weight Standardization formula:
        #   w_std = (w - mean) / sqrt(var * fan_in + eps)
        # 
        # Why multiply variance by fan_in?
        #   - Larger fan_in means more inputs, so we expect more variance
        #   - Scaling by fan_in normalizes for network depth (Xavier initialization idea)
        #   - This prevents gradients from exploding/vanishing through layers
        # 
        # Why add eps?
        #   - Prevents division by zero if variance is exactly 0
        #   - Improves numerical stability during backprop
        w = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        # Optionally multiply by learnable gain parameter
        # Shape: (out_channels, 1, 1, 1) broadcasts across all weight dimensions
        # Gain allows the network to learn the optimal scaling after standardization
        if self.gain is not None:
            w = w * self.gain
        # Perform convolution using the standardized weights
        # F.conv2d applies: output = conv(input, standardized_weights, bias, ...)
        # We pass:
        #   - x: input tensor
        #   - w: standardized weight tensor (computed above, NOT self.weight)
        #   - self.bias: original bias (not standardized)
        #   - self.stride, self.padding, self.dilation, self.groups: from nn.Conv2d init
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class Scale(nn.Module):
    """
    Multiply every activation by a fixed constant.

    In OTTT-SNN, a Scale(2.74) layer is inserted after each LIF neuron when
    weight standardization is active. The value 2.74 is derived empirically:
    WS keeps weights at unit variance, so spike rates tend to be low. Multiplying
    by ~2.74 compensates for this and restores the effective signal magnitude
    entering the next conv layer, making training more stable.

    This layer has no learnable parameters — the scale is fixed at construction.
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Constant multiplier applied to every element of the input.
        """
        super().__init__()
        self.scale = scale

    def forward(self, x):
        # Element-wise multiplication — shape is unchanged.
        return x * self.scale


# ---------------------------------------------------------------------------
# OTTT VGG-9
# ---------------------------------------------------------------------------

class OTTT_VGG9(nn.Module):
    """
    Spiking VGG-9 replicating OTTT-SNN's online_spiking_vgg11_ws (light classifier).

    Architecture overview
    ---------------------
    8 convolutional blocks with channel progression [64, 128, 256, 256, 512, 512, 512, 512].
    Each block follows the pattern:
        ScaledWSConv2d  →  LIF neuron  →  Scale(2.74)  →  (AvgPool2d at blocks 2, 4, 6)

    After all conv blocks:
        AdaptiveAvgPool2d(1, 1)  →  flatten  →  Linear(512, num_classes)

    The classifier head outputs raw logits (no spiking neuron), so the trainer
    accumulates them over timesteps and takes the argmax at evaluation time.

    Time dimension
    --------------
    forward() processes ONE timestep at a time — shape (B, C, H, W).
    The caller (trainer or test loop) is responsible for iterating over timesteps
    and resetting membrane potentials between sequences via reset().

    Why VGG-9?
    ----------
    VGG-N names count total layers with learnable weights.
    This network has 8 conv layers + 1 FC layer = 9 weight layers → VGG-9.
    """

    def __init__(
        self,
        in_channels=2,
        num_classes=11,
        beta=0.5,
        threshold=1.0,
        spike_grad=None,
        ws_scale=2.74,
        verbose=False,
        debug=False,
    ):
        """
        Args:
            in_channels (int): Input channels. DVSGesture uses 2 (ON/OFF polarity events).
            num_classes (int): Output classes. DVSGesture has 11 gesture categories.
            beta (float): LIF membrane decay factor. Derived from OTTT tau=2: beta = 1 - 1/tau = 0.5.
                          At each timestep: mem = beta * mem + input_current.
                          beta=0.5 means the membrane forgets half its potential every step.
            threshold (float): Spike threshold. Neuron fires when mem >= threshold, then
                               mem is reduced by threshold (soft/subtract reset).
            spike_grad (callable): Surrogate gradient function for the non-differentiable
                                   spike operation. OTTT uses sigmoid with slope=4 (default).
            ws_scale (float): Constant applied by Scale layers after each LIF. Default 2.74
                              matches the OTTT-SNN paper.
            verbose (bool): If True, print a config summary and per-layer weight stats after init.
        """
        super().__init__()
        self._verbose = verbose
        self._debug   = debug
        # Accumulators used by debug instrumentation — never printed mid-run.
        # Call debug_summary(reset=True) once per epoch to see a compact table.
        self._dbg_fwd_calls = 0   # total forward() calls (= timesteps seen)
        # per-layer lists, appended on every forward call:
        #   _dbg_spike_rates[i]  — spike rate of block i+1
        #   _dbg_in_stds[i]      — std of the pre-LIF input  (hypothesis 3: attenuation)
        #   _dbg_gain_vals[i]    — mean gain of conv i+1     (hypothesis 4: gain collapse)
        #   _dbg_grad_norms[i]   — grad norms at Scale output (hypothesis 1: explosion)
        self._dbg_spike_rates: list[list[float]] = [[] for _ in range(8)]
        self._dbg_in_stds:     list[list[float]] = [[] for _ in range(8)]
        self._dbg_gain_vals:   list[list[float]] = [[] for _ in range(8)]
        self._dbg_grad_norms:  list[list[float]] = [[] for _ in range(8)]

        # Default surrogate: sigmoid with slope=4, matching OTTT's OnlineLIFNode.
        # The surrogate gradient approximates d(spike)/d(membrane) during backprop.
        # Slope=4 gives a relatively wide gradient — less vanishing than steeper surrogates.
        # @TODO: remove the gradient from here since it's training related stuff
        if spike_grad is None:
            spike_grad = surrogate.sigmoid(slope=4)

        self.ws_scale = ws_scale

        # ===== ARCHITECTURE DEFINITION =====
        # Channel progression doubles from 64 to 128, then expands to 256 and 512.
        # This matches the standard VGG feature extractor design:
        #   deeper layers → more channels → more abstract features.
        channels = [64, 128, 256, 256, 512, 512, 512, 512]

        # AvgPool2d(2, 2) is applied after blocks 2, 4, and 6.
        # This halves the spatial resolution three times:
        #   128×128 → 64×64 → 32×32 → 16×16
        # Block 8 is NOT followed by a regular pool — AdaptiveAvgPool handles it.
        pool_after = {2, 4, 6}

        # ===== LIF NEURON CONFIGURATION =====
        # All 8 LIF neurons share the same hyperparameters (beta, threshold, surrogate).
        # init_hidden=False: we manage membrane state manually (stored as self.mem{i}).
        # learn_beta=False, learn_threshold=False: fixed dynamics, matching the paper.
        lif_kwargs = dict(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            reset_mechanism='subtract',  # soft reset: mem -= threshold * spike (not hard reset to 0)
            init_hidden=False,
            learn_beta=False,
            learn_threshold=False,
        )

        # ===== BUILD 8 CONV BLOCKS DYNAMICALLY =====
        # We use setattr to register layers as named attributes (conv1…conv8, lif1…lif8, etc.)
        # so they are visible to PyTorch's parameter/module tracking.
        # The alternative (nn.ModuleList) would work too, but named attributes are more explicit.
        prev_ch = in_channels
        for i, ch in enumerate(channels, start=1):
            # Weight-standardized convolution, 3×3 kernel, same padding, with bias.
            setattr(self, f'conv{i}', ScaledWSConv2d(prev_ch, ch, 3, padding=1, bias=True))
            # setattr(self, f'conv{i}', nn.Conv2d(prev_ch, ch, 3, padding=1, bias=True))
            # LIF neuron: integrates conv output, fires if membrane >= threshold.
            setattr(self, f'lif{i}', snn.Leaky(**lif_kwargs))
            # Post-spike scaling: compensates for low spike rates under WS.
            setattr(self, f'scale{i}', Scale(ws_scale))
            # Spatial downsampling at blocks 2, 4, 6. AvgPool is used (not MaxPool)
            # because spikes are binary (0/1) — averaging preserves rate information
            # better than max-selecting a single spike.
            if i in pool_after:
                setattr(self, f'pool{i}', nn.AvgPool2d(2, 2))
            prev_ch = ch

        # ===== CLASSIFIER HEAD =====
        # AdaptiveAvgPool2d(1, 1) collapses the spatial dimensions to a single value
        # per channel, regardless of the input spatial size. After block 8 the feature
        # map is 16×16; AdaptiveAvgPool reduces it to 1×1 → 512 features.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Final linear layer maps 512 features to class scores (raw logits, no LIF).
        self.fc = nn.Linear(512, num_classes)

        # ===== BOOKKEEPING =====
        # Store the config so forward() and init_states() can iterate without
        # repeating the hard-coded channel list.
        self._channels = channels
        self._pool_after = pool_after
        self._num_blocks = len(channels)

        # ===== MEMBRANE STATE INITIALISATION =====
        # Each LIF neuron needs a persistent membrane potential that carries over
        # between timesteps within the same sequence.
        # We store them as self.mem1 … self.mem8, initially scalar zeros.
        # init_states() properly initialises them to zero tensors of the right shape.
        for i in range(1, self._num_blocks + 1):
            setattr(self, f'mem{i}', torch.zeros(1))

        self._initialize_weights()
        if self._verbose:
            self._print_config()

    def forward(self, x):
        """
        Process a single timestep through all 8 conv blocks and the classifier head.

        Args:
            x (torch.Tensor): Single frame of shape (B, C, H, W).
                              For DVSGesture: (B, 2, 128, 128).

        Returns:
            spk_list (list[Tensor]): Spike (or scaled spike) output of each block,
                                     plus the FC logits as the last element.
                                     Length = 9 (8 conv blocks + 1 classifier).
            mem_list (list[Tensor]): Membrane potential after each LIF neuron,
                                     plus the FC logits repeated as the last element.
                                     Same length as spk_list.

        Why return both spikes and membranes?
            - Trainers that implement local learning rules (STLLR, TP) need access to
              both spike rates and membrane potentials to compute their update signals.
            - BPTT only needs the final output, but having a consistent interface
              means the same network class works with all trainers.
        """
        spk_list = []
        mem_list = []

        if self._debug:
            self._dbg_fwd_calls += 1

        for i in range(1, self._num_blocks + 1):
            # ===== STEP 1: Convolution with weight standardization =====
            # Input x at block 1: (B, 2, 128, 128) — raw DVS frame
            # Input x at later blocks: scaled spikes from the previous block
            x = getattr(self, f'conv{i}')(x)

            # ===== STEP 2: LIF neuron — integrate and fire =====
            # lif(input_current, membrane_potential) → (spike, new_membrane)
            # spike: binary tensor (0.0 or 1.0), same shape as x
            # new_membrane: updated membrane potential after the step
            lif = getattr(self, f'lif{i}')
            mem = getattr(self, f'mem{i}')

            # --- DEBUG hypothesis 3: accumulate pre-LIF input std (signal attenuation) ---
            if self._debug:
                self._dbg_in_stds[i - 1].append(x.std().item())

            spk, mem = lif(x, mem)
            # Persist the updated membrane for the next timestep call.
            setattr(self, f'mem{i}', mem)

            # --- DEBUG hypotheses 2 & 4: accumulate spike rate and gain value ---
            if self._debug:
                self._dbg_spike_rates[i - 1].append(spk.mean().item())
                gain_val = (getattr(self, f'conv{i}').gain.mean().item()
                            if getattr(self, f'conv{i}').gain is not None else float('nan'))
                self._dbg_gain_vals[i - 1].append(gain_val)

            # ===== STEP 3: Scale — amplify sparse binary spikes =====
            # Raw spikes are 0/1, so the expected value equals the firing rate.
            # Multiplying by 2.74 rescales this to a range more similar to
            # typical continuous activations, making the next conv layer easier to train.
            spk = getattr(self, f'scale{i}')(spk)

            # --- DEBUG hypothesis 1: accumulate grad norm at Scale output (gradient explosion) ---
            if self._debug and spk.requires_grad:
                _buf = self._dbg_grad_norms[i - 1]
                def _grad_hook(grad, _b=_buf):
                    _b.append(grad.norm().item())
                spk.register_hook(_grad_hook)

            # ===== STEP 4: Spatial pooling (blocks 2, 4, 6 only) =====
            # Reduces spatial resolution: 128→64→32→16 over the three pool layers.
            if i in self._pool_after:
                spk = getattr(self, f'pool{i}')(spk)

            spk_list.append(spk)
            mem_list.append(mem)
            # Pass spikes (not membrane potentials) to the next block.
            # This is the standard SNN forward pass: each layer receives spikes, not voltages.
            x = spk

        # ===== CLASSIFIER HEAD =====
        # Collapse spatial dims: (B, 512, 16, 16) → (B, 512, 1, 1) → (B, 512)
        out = self.global_pool(x).flatten(1)
        # Linear projection to class scores: (B, 512) → (B, num_classes)
        # No LIF here — the trainer accumulates these logits over all timesteps
        # and takes the argmax after summing.
        out = self.fc(out)
        spk_list.append(out)
        mem_list.append(out)

        return spk_list, mem_list

    def init_states(self):
        """
        Reset all membrane potentials to zero.

        Must be called at the start of each new sequence (before the first timestep).
        Without this, membrane state from a previous sample would bleed into the next,
        causing incorrect integration and degraded accuracy.

        snn.Leaky.init_leaky() returns a properly-shaped zero tensor on the correct device,
        handling GPU placement automatically.
        """
        for i in range(1, self._num_blocks + 1):
            lif = getattr(self, f'lif{i}')
            setattr(self, f'mem{i}', lif.init_leaky())

    def reset(self):
        """
        Alias for init_states(), required by the trainer interface.

        All trainers call network.reset() between samples. Having this alias
        means OTTT_VGG9 is a drop-in for any trainer that follows the interface,
        without the trainer needing to know the internal method name.
        """
        self.init_states()

    def debug_summary(self, label: str = "", reset: bool = True) -> None:
        """
        Print a compact per-layer table of the stats accumulated since the last call
        (or since init).  Call once per epoch — not per timestep.

        Columns:
          spike_rate — mean firing rate (hypothesis 2: silence onset layer)
          in_std     — mean std of pre-LIF input (hypothesis 3: signal attenuation)
          gain       — mean learnable gain value (hypothesis 4: gain collapse)
          grad_mean  — mean Scale-output grad norm (hypothesis 1: gradient explosion)
          grad_max   — max  Scale-output grad norm

        Args:
            label: optional prefix, e.g. "epoch=1 train" or "epoch=1 eval".
            reset: if True, clear all accumulators after printing.
        """
        if not self._debug:
            return

        import math

        def _finite(lst):
            return [x for x in lst if x == x and not math.isinf(x)]

        def _mean(lst):
            f = _finite(lst)
            return sum(f) / len(f) if f else float('nan')

        def _max(lst):
            f = _finite(lst)
            return max(f) if f else float('nan')

        def _first(lst):
            f = _finite(lst)
            return f[0] if f else float('nan')

        def _nan_pct(lst):
            if not lst:
                return float('nan')
            return 100.0 * (len(lst) - len(_finite(lst))) / len(lst)

        hdr = f"[OTTT_VGG9 debug] {label}  fwd_calls={self._dbg_fwd_calls}"
        print(hdr)
        print(f"  {'Layer':<6} {'spike_rate':>10} {'in_std_0':>9} {'in_std':>8} "
              f"{'gain_0':>7} {'gain':>7} {'grad_mean':>10} {'grad_max':>10} {'nan%':>6}")
        for i in range(self._num_blocks):
            print(
                f"  L{i+1:<5} "
                f"{_mean(self._dbg_spike_rates[i]):>10.4f} "
                f"{_first(self._dbg_in_stds[i]):>9.4f} "
                f"{_mean(self._dbg_in_stds[i]):>8.4f} "
                f"{_first(self._dbg_gain_vals[i]):>7.4f} "
                f"{_mean(self._dbg_gain_vals[i]):>7.4f} "
                f"{_mean(self._dbg_grad_norms[i]):>10.3e} "
                f"{_max( self._dbg_grad_norms[i]):>10.3e} "
                f"{_nan_pct(self._dbg_in_stds[i]):>5.1f}%"
            )

        if reset:
            self._dbg_fwd_calls = 0
            self._dbg_spike_rates = [[] for _ in range(self._num_blocks)]
            self._dbg_in_stds     = [[] for _ in range(self._num_blocks)]
            self._dbg_gain_vals   = [[] for _ in range(self._num_blocks)]
            self._dbg_grad_norms  = [[] for _ in range(self._num_blocks)]

    def _print_config(self):
        """Print a one-time config summary with per-layer weight statistics."""
        import math
        conv_type = type(getattr(self, 'conv1')).__name__
        n_params = sum(p.numel() for p in self.parameters())

        print(f"\n{'='*60}")
        print(f"  OTTT_VGG9")
        print(f"{'='*60}")
        print(f"  {'Conv Type':<25} {conv_type}")
        print(f"  {'Channels':<25} {self._channels}")
        print(f"  {'Pooling After':<25} {sorted(self._pool_after)}")
        print(f"  {'Beta':<25} {getattr(self, 'lif1').beta.item():.3f}")
        print(f"  {'Threshold':<25} {getattr(self, 'lif1').threshold.item():.3f}")
        print(f"  {'WS Scale':<25} {self.ws_scale}")
        print(f"  {'Weight Standardization':<25} {'yes' if conv_type == 'ScaledWSConv2d' else 'no'}")
        print(f"  {'Parameters':<25} {n_params:,}")
        print(f"{'='*60}\n")

    def _initialize_weights(self):
        """
        Initialise network weights following the OTTT-SNN paper.

        Conv layers: Kaiming normal (He initialisation), fan_out mode.
            - Designed for ReLU-like activations; works well for LIF surrogates too.
            - fan_out mode preserves variance in the backward pass.
            - Bias initialised to zero.

        Linear (FC) layer: Normal distribution with mean=0, std=0.01.
            - Small std prevents the untrained classifier from producing large logits.
            - Bias initialised to zero.

        Note: ScaledWSConv2d is a subclass of nn.Conv2d, so isinstance checks
        against nn.Conv2d already catch it. The explicit tuple handles both.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, ScaledWSConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Quick sanity check: run T timesteps with a batch of B random inputs
    # and verify the output shape and that no NaN/Inf values appear.
    B, T = 2, 20
    model = OTTT_VGG9(in_channels=2, num_classes=11)
    # Reset membranes before the sequence starts.
    model.init_states()

    out_sum = torch.zeros(B, 11)
    for t in range(T):
        # Each call to model() processes one (B, 2, 128, 128) frame.
        # spk_list[-1] is the FC output (logits), shape (B, 11).
        spk_list, mem_list = model(torch.randn(B, 2, 128, 128))
        # Accumulate logits over timesteps — argmax of the sum is the prediction.
        out_sum += spk_list[-1]

    n_params = sum(p.numel() for p in model.parameters())
    print(f'OTTT_VGG9  output={spk_list[-1].shape}  params={n_params:,}')
    assert spk_list[-1].shape == (B, 11)
    assert torch.isfinite(out_sum).all()
    print('Smoke test passed.')
