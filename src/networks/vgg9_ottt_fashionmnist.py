"""
Spiking VGG-9 OTTT variant for Fashion-MNIST.

Adapted from the CIFAR-10 architecture (Xiao et al., NeurIPS 2022) for Fashion-MNIST:
  - in_channels=1  (grayscale images)
  - num_classes=10 (Fashion-MNIST classes)
  - Lighter channel progression due to smaller 28×28 input
  - Pooling after blocks 1, 3 to reduce spatial dimensions early

Architecture:
  8 conv blocks [32, 64, 128, 128, 256, 256, 256, 256].
  Each block: ScaledWSConv2d(bias=True) → LIF → Scale(2.74) → (AvgPool after blocks 1, 3).
  Classifier: AdaptiveAvgPool2d(1,1) → Linear(256, 10) — no LIF in head.

Input per timestep: (B, 1, 28, 28) — rate-coded Fashion-MNIST frame.
Spatial reduction: 28×28 → 14×14 → 7×7 → 3×3 (via two pooling layers).
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
                               WS signal suppression. Matches OTTT paper's design.
            eps (float): Small constant added to variance for numerical stability.
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
        fan_in = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        mean = self.weight.mean(dim=[1, 2, 3], keepdim=True)
        var = self.weight.var(dim=[1, 2, 3], keepdim=True)
        w = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            w = w * self.gain
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class Scale(nn.Module):
    """
    Multiply every activation by a fixed constant.

    In OTTT-SNN, a Scale(2.74) layer is inserted after each LIF neuron when
    weight standardization is active. The value 2.74 is derived empirically:
    WS keeps weights at unit variance, so spike rates tend to be low. Multiplying
    by ~2.74 compensates for this and restores the effective signal magnitude.
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Constant multiplier applied to every element of the input.
        """
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


# ---------------------------------------------------------------------------
# OTTT VGG-9 for Fashion-MNIST
# ---------------------------------------------------------------------------

class OTTT_VGG9_FashionMNIST(nn.Module):
    """
    Spiking VGG-9 OTTT variant for Fashion-MNIST (1-channel grayscale, 10 classes).

    Architecture overview
    ---------------------
    8 convolutional blocks with channel progression [32, 64, 128, 128, 256, 256, 256, 256].
    Each block follows the pattern:
        ScaledWSConv2d  →  LIF neuron  →  Scale(2.74)  →  (AvgPool at blocks 1, 3)

    After all conv blocks:
        AdaptiveAvgPool2d(1, 1)  →  flatten  →  Linear(256, 10)

    Spatial resolution over blocks (Fashion-MNIST input 28×28):
      Block 1-2: 28×28 → 14×14 (pool after block 1)
      Block 3-4: 14×14 → 7×7   (pool after block 3)
      Block 5-8: 7×7   → 3×3   (final adaptive pool)

    Time dimension
    --------------
    forward() processes ONE timestep at a time — shape (B, C, H, W).
    The caller (trainer or test loop) is responsible for iterating over timesteps
    and resetting membrane potentials between sequences via reset().
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=10,
        beta=0.5,
        threshold=1.0,
        spike_grad=None,
        ws_scale=2.74,
        verbose=False,
    ):
        """
        Args:
            in_channels (int): Input channels. Fashion-MNIST uses 1 (grayscale).
            num_classes (int): Output classes. Fashion-MNIST has 10 classes.
            beta (float): LIF membrane decay factor. Default 0.5 (from tau=2).
            threshold (float): Spike threshold. Neuron fires when mem >= threshold.
            spike_grad (callable): Surrogate gradient function. OTTT uses sigmoid with slope=4.
            ws_scale (float): Constant applied by Scale layers after each LIF. Default 2.74.
            verbose (bool): If True, print config summary after init.
        """
        super().__init__()
        self._verbose = verbose

        if spike_grad is None:
            spike_grad = surrogate.sigmoid(slope=4)

        self.ws_scale = ws_scale

        # Channel progression through 8 conv blocks (lighter than CIFAR-10 variant)
        channels = [32, 64, 128, 128, 256, 256, 256, 256]

        # Pool after blocks 1 and 3 (early pooling for 28×28 input)
        pool_after = {1, 3}

        # LIF neuron configuration
        lif_kwargs = dict(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            reset_mechanism='subtract',
            init_hidden=False,
            learn_beta=False,
            learn_threshold=False,
        )

        # Build 8 conv blocks dynamically
        prev_ch = in_channels
        for i, ch in enumerate(channels, start=1):
            setattr(self, f'conv{i}', ScaledWSConv2d(prev_ch, ch, 3, padding=1, bias=True))
            setattr(self, f'lif{i}', snn.Leaky(**lif_kwargs))
            setattr(self, f'scale{i}', Scale(ws_scale))
            if i in pool_after:
                setattr(self, f'pool{i}', nn.AvgPool2d(2, 2))
            prev_ch = ch

        # Classifier head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Bookkeeping
        self._channels = channels
        self._pool_after = pool_after
        self._num_blocks = len(channels)

        # Initialize membrane potentials
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
                              For Fashion-MNIST: (B, 1, 28, 28).

        Returns:
            spk_list (list[Tensor]): Spike (or scaled spike) output of each block,
                                     plus the FC logits as the last element.
            mem_list (list[Tensor]): Membrane potential after each LIF neuron,
                                     plus the FC logits repeated as the last element.
        """
        spk_list = []
        mem_list = []

        for i in range(1, self._num_blocks + 1):
            # Convolution with weight standardization
            x = getattr(self, f'conv{i}')(x)

            # LIF neuron — integrate and fire
            lif = getattr(self, f'lif{i}')
            mem = getattr(self, f'mem{i}')
            spk, mem = lif(x, mem)
            setattr(self, f'mem{i}', mem)

            # Scale — amplify sparse spikes
            spk = getattr(self, f'scale{i}')(spk)

            # Spatial pooling (blocks 1, 3 only)
            if i in self._pool_after:
                spk = getattr(self, f'pool{i}')(spk)

            spk_list.append(spk)
            mem_list.append(mem)
            x = spk

        # Classifier head
        out = self.global_pool(x).flatten(1)
        out = self.fc(out)
        spk_list.append(out)
        mem_list.append(out)

        return spk_list, mem_list

    def init_states(self):
        """
        Reset all membrane potentials to zero.

        Must be called at the start of each new sequence (before the first timestep).
        """
        for i in range(1, self._num_blocks + 1):
            lif = getattr(self, f'lif{i}')
            setattr(self, f'mem{i}', lif.init_leaky())

    def reset(self):
        """
        Alias for init_states(), required by the trainer interface.
        """
        self.init_states()

    def _print_config(self):
        """Print a one-time config summary."""
        conv_type = type(getattr(self, 'conv1')).__name__
        n_params = sum(p.numel() for p in self.parameters())

        print(f"\n{'='*60}")
        print(f"  OTTT_VGG9_FashionMNIST")
        print(f"{'='*60}")
        print(f"  {'Conv Type':<25} {conv_type}")
        print(f"  {'Channels':<25} {self._channels}")
        print(f"  {'Pooling After':<25} {sorted(self._pool_after)}")
        print(f"  {'Beta':<25} {getattr(self, 'lif1').beta.item():.3f}")
        print(f"  {'Threshold':<25} {getattr(self, 'lif1').threshold.item():.3f}")
        print(f"  {'WS Scale':<25} {self.ws_scale}")
        print(f"  {'Parameters':<25} {n_params:,}")
        print(f"{'='*60}\n")

    def _initialize_weights(self):
        """
        Initialize network weights following the OTTT-SNN paper.

        Conv layers: Kaiming normal (He initialization), fan_in mode.
        Linear (FC) layer: Normal distribution with mean=0, std=0.01.
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
    B, T = 2, 20
    model = OTTT_VGG9_FashionMNIST(in_channels=1, num_classes=10)
    model.init_states()

    out_sum = torch.zeros(B, 10)
    for t in range(T):
        spk_list, mem_list = model(torch.randn(B, 1, 28, 28))
        out_sum += spk_list[-1]

    n_params = sum(p.numel() for p in model.parameters())
    print(f'OTTT_VGG9_FashionMNIST  output={spk_list[-1].shape}  params={n_params:,}')
    assert spk_list[-1].shape == (B, 10)
    assert torch.isfinite(out_sum).all()
    print('Smoke test passed.')
