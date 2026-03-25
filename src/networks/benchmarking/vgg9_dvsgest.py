"""
Spiking VGG-9 faithfully replicating traces_propagation BP_CNN variant 9
(Fernandez-Musoles et al.).

Source: temp_repos/traces_propagation/models/bp_cnn.py   (BP_CNN, VGG_CONFIGS[9])
        temp_repos/traces_propagation/models/neuron_layers.py (LIFLayerCNN, LILayer)

Architecture (DVSGesture config, vgg_variant=9, norm="weight"):
  8 conv blocks [64,128,256,256,512,512,512,512].
  Each block: WSConv2d(scale=1.8, no gain, no bias) -> LIF -> Pool.
  Pool: MaxPool(2) after blocks 2,4,6; AdaptiveAvgPool(2,2) after block 8.
  Classifier: LI readout (leaky integrate, no fire, leak=1.0, WS linear, no bias).
  Classifier input: 512 * 2 * 2 = 2048 features.

Neuron defaults (DVSGesture BPTT experiment):
  beta = 0.53 (l_leak_m), threshold = 1.0 (l_vth), soft (subtract) reset.
  Surrogate type "1": grad = 1 / (1 + (pi * x)^2).

Init: kaiming_normal for all weights (via _setup_norm with norm="weight").

forward() processes a single timestep (B, C, H, W).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn


# ---------------------------------------------------------------------------
# Custom surrogate — type "1" from traces_propagation/models/spike_activation.py
#   grad = scale / (1 + (pi * (v - vth))^2)
# This does not match any snntorch built-in exactly.
# ---------------------------------------------------------------------------

class _ATanSurrogateFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, scale):
        ctx.save_for_backward(input_)
        ctx.scale = scale
        return (input_ >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad = ctx.scale / (1.0 + (math.pi * input_) ** 2)
        return grad * grad_output, None


class ATanSurrogate(nn.Module):
    """Surrogate matching traces_propagation type '1' (scale=1.0 default)."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, input_):
        return _ATanSurrogateFn.apply(input_, self.scale)


# ---------------------------------------------------------------------------
# Weight-standardized Conv2d — traces_propagation style
#   w = 1.8 * (w - mean) / sqrt(var_biased * fan_in + eps)
#   No learnable gain, no bias.  eps = 1e-5.
# ---------------------------------------------------------------------------

class WSConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.eps = eps

    def forward(self, x):
        w = self.conv.weight
        fan_in = w[0].numel()
        mean = w.mean(dim=[1, 2, 3], keepdim=True)
        var = w.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        w_std = 1.8 * (w - mean) / torch.sqrt(var * fan_in + self.eps)
        return F.conv2d(x, w_std, bias=None, stride=self.conv.stride,
                        padding=self.conv.padding)

    @property
    def weight(self):
        return self.conv.weight


# ---------------------------------------------------------------------------
# Leaky Integrator head — traces_propagation LILayer
#   mem = leak * mem + WS_linear(x)
#   Returns membrane potential (no spike, no threshold).
# ---------------------------------------------------------------------------

class LeakyIntegrator(nn.Module):

    def __init__(self, in_features, out_features, leak=1.0, eps=1e-5):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.leak = leak
        self.eps = eps
        self.mem = torch.zeros(1)

    def forward(self, x):
        w = self.fc.weight
        fan_in = w.size(1)
        mean = w.mean(dim=1, keepdim=True)
        var = w.var(dim=1, keepdim=True, unbiased=False)
        w_std = 1.8 * (w - mean) / torch.sqrt(var * fan_in + self.eps)
        cur = F.linear(x, w_std, bias=None)
        self.mem = self.leak * self.mem + cur
        return self.mem

    def reset(self):
        self.mem = torch.zeros(1, device=self.fc.weight.device)


# ---------------------------------------------------------------------------
# traces_propagation VGG-9 (BP_CNN variant 9)
# ---------------------------------------------------------------------------

class TP_VGG9(nn.Module):
    """Exact replica of traces_propagation BP_CNN with VGG_CONFIGS[9]."""

    #                  (out_ch, pool_size, pool_type)
    VGG9_CFG = [
        (64,  1, 'none'),
        (128, 2, 'max'),
        (256, 1, 'none'),
        (256, 2, 'max'),
        (512, 1, 'none'),
        (512, 2, 'max'),
        (512, 1, 'none'),
        (512, 2, 'aavg'),
    ]

    def __init__(
        self,
        in_channels=2,
        num_classes=11,
        beta=0.53,
        threshold=1.0,
        spike_grad=None,
        verbose=False,
    ):
        super().__init__()

        if spike_grad is None:
            spike_grad = ATanSurrogate(scale=1.0)

        lif_kwargs = dict(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            reset_mechanism='subtract',
            init_hidden=False,
            learn_beta=False,
            learn_threshold=False,
        )

        prev_ch = in_channels
        for i, (ch, pool_sz, pool_type) in enumerate(self.VGG9_CFG, start=1):
            setattr(self, f'conv{i}', WSConv2d(prev_ch, ch, 3, padding=1))
            setattr(self, f'lif{i}', snn.Leaky(**lif_kwargs))

            if pool_type == 'max':
                setattr(self, f'pool{i}', nn.MaxPool2d(pool_sz, pool_sz))
            elif pool_type == 'aavg':
                setattr(self, f'pool{i}', nn.AdaptiveAvgPool2d((pool_sz, pool_sz)))
            # 'none' → no pool attribute created

            prev_ch = ch

        # Classifier: LI readout with leak=1.0 (pure integration, no fire).
        # After block 8 AdaptiveAvgPool2d(2,2): spatial = 2x2, channels = 512.
        self.head = LeakyIntegrator(512 * 2 * 2, num_classes, leak=1.0)

        self._num_blocks = len(self.VGG9_CFG)

        for i in range(1, self._num_blocks + 1):
            setattr(self, f'mem{i}', torch.zeros(1))

        self._initialize_weights()

        if verbose:
            pool_strs = []
            for _, pool_sz, pool_type in self.VGG9_CFG:
                if pool_type == 'none':
                    pool_strs.append('—')
                elif pool_type == 'max':
                    pool_strs.append(f'max{pool_sz}')
                elif pool_type == 'aavg':
                    pool_strs.append(f'aavg{pool_sz}x{pool_sz}')
            channels = [ch for ch, _, _ in self.VGG9_CFG]
            n_params = sum(p.numel() for p in self.parameters())
            sg_name = type(spike_grad).__name__ if spike_grad is not None else 'ATanSurrogate'
            sg_scale = getattr(spike_grad, 'scale', 1.0)
            print(
                f"\n[VERBOSE] PRINTING NETWORK INFORMATIONS\n"
                f"TP_VGG9  in_ch={in_channels}  classes={num_classes}"
                f"  beta={beta}  threshold={threshold}"
                f"  params={n_params:,}\n"
                f"  channels : {channels}\n"
                f"  pool     : {pool_strs}\n"
                f"  head     : LeakyIntegrator({channels[-1]*2*2}->{num_classes}, leak=1.0)  WS(scale=1.8)\n"
                f"  surrogate: {sg_name}(scale={sg_scale})  reset=subtract\n"
            )

    def forward(self, x):
        spk_list = []
        mem_list = []

        for i, (_, _, pool_type) in enumerate(self.VGG9_CFG, start=1):
            x = getattr(self, f'conv{i}')(x)

            lif = getattr(self, f'lif{i}')
            mem = getattr(self, f'mem{i}')
            spk, mem = lif(x, mem)
            setattr(self, f'mem{i}', mem)

            if pool_type != 'none':
                spk = getattr(self, f'pool{i}')(spk)

            spk_list.append(spk)
            mem_list.append(mem)
            x = spk

        # LI readout head
        out = self.head(x.flatten(1))
        spk_list.append(out)
        mem_list.append(out)

        return spk_list, mem_list

    def init_states(self):
        for i in range(1, self._num_blocks + 1):
            lif = getattr(self, f'lif{i}')
            setattr(self, f'mem{i}', lif.init_leaky())
        self.head.reset()

    def reset(self):
        self.init_states()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    B, T = 2, 20
    model = TP_VGG9(in_channels=2, num_classes=11)
    model.init_states()

    out_sum = torch.zeros(B, 11)
    for t in range(T):
        spk_list, mem_list = model(torch.randn(B, 2, 128, 128))
        out_sum += spk_list[-1]

    n_params = sum(p.numel() for p in model.parameters())
    print(f'TP_VGG9  output={spk_list[-1].shape}  params={n_params:,}')
    assert spk_list[-1].shape == (B, 11)
    assert torch.isfinite(out_sum).all()
    print('Smoke test passed.')
