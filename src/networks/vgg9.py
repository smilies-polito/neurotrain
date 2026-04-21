"""
Unified parameterized Spiking VGG-9.

The class implements the BaseSNN trainer interface (hidden_weight_layers,
output_layer) so that it composes with any trainer that speaks that contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from networks.base_snn import BaseSNN
from networks._components import (
    ATanSurrogate,
    LeakyIntegrator,
    Scale,
    WSConv2d,
)


# ---------------------------------------------------------------------------
# Block-level building helpers
# ---------------------------------------------------------------------------

_POOL_NONE = "none"
_POOL_MAX  = "max"
_POOL_AVG  = "avg"
_POOL_AAVG = "aavg"

_HEAD_LI            = "leaky_integrator"   # WS-linear LI readout (TP-style)
_HEAD_GLOBAL_LINEAR = "global_linear"      # AdaptiveAvgPool(1,1) + Linear (OTTT-style)


@dataclass
class VGG9Config:
    """Hyper-parameters of one VGG9 recipe."""

    in_channels: int
    num_classes: int
    input_shape: Tuple[int, int, int]        # (C, H, W)
    channels: Sequence[int] = field(default_factory=lambda: (64, 128, 256, 256, 512, 512, 512, 512))

    # Per-block pooling. Length must equal len(channels). Each entry is a
    # (pool_type, pool_arg) tuple where pool_type is one of the _POOL_* constants
    # and pool_arg is the kernel size (or output-size for 'aavg'). 'none' ignores
    # pool_arg.
    pool_spec: Sequence[Tuple[str, int]] = field(default_factory=lambda: (
        (_POOL_NONE, 0),
        (_POOL_MAX,  2),
        (_POOL_NONE, 0),
        (_POOL_MAX,  2),
        (_POOL_NONE, 0),
        (_POOL_MAX,  2),
        (_POOL_NONE, 0),
        (_POOL_AAVG, 2),
    ))

    # Conv gain: fixed scalar applied during weight standardization.
    conv_gain: float = 1.8

    # Post-LIF Scale(k). Only applied when > 0.
    scale_after_lif: float = 0.0

    # LIF dynamics.
    beta: float = 0.53
    threshold: float = 1.0
    reset_mechanism: str = "subtract"
    surrogate_kind: str = "atan"    # one of: "atan", "sigmoid"
    surrogate_scale: float = 1.0
    surrogate_slope: float = 4.0

    # Head flavor + dimensioning.
    head_type: str = _HEAD_LI
    # For _HEAD_LI: spatial size after the AdaptiveAvgPool in the last block.
    li_head_spatial: int = 2
    li_head_leak: float = 1.0

    def __post_init__(self):
        if len(self.pool_spec) != len(self.channels):
            raise ValueError("pool_spec length must equal channels length.")
        if self.head_type not in (_HEAD_LI, _HEAD_GLOBAL_LINEAR):
            raise ValueError(f"Unknown head_type: {self.head_type}")


def _build_spike_grad(cfg: VGG9Config):
    if cfg.surrogate_kind == "atan":
        return ATanSurrogate(scale=float(cfg.surrogate_scale))
    if cfg.surrogate_kind == "sigmoid":
        return surrogate.sigmoid(slope=float(cfg.surrogate_slope))
    raise ValueError(f"Unknown surrogate_kind: {cfg.surrogate_kind}")


def _build_pool(pool_type: str, pool_arg: int) -> Optional[nn.Module]:
    if pool_type == _POOL_NONE:
        return None
    if pool_type == _POOL_MAX:
        return nn.MaxPool2d(pool_arg, pool_arg)
    if pool_type == _POOL_AVG:
        return nn.AvgPool2d(pool_arg, pool_arg)
    if pool_type == _POOL_AAVG:
        return nn.AdaptiveAvgPool2d((pool_arg, pool_arg))
    raise ValueError(f"Unknown pool_type: {pool_type}")


def _build_conv(cfg: VGG9Config, in_ch: int, out_ch: int) -> nn.Conv2d:
    return WSConv2d(in_ch, out_ch, 3, padding=1, gain=cfg.conv_gain)


# ---------------------------------------------------------------------------
# Unified VGG-9
# ---------------------------------------------------------------------------

class VGG9(BaseSNN):
    """
    Single parameterized Spiking VGG-9.

    forward() processes a single timestep of shape (B, C, H, W) and returns
    (spk_rec, mem_rec) with length `len(channels) + 1` (one entry per conv
    block + one for the head).

    Call reset() between sequences.
    """

    def __init__(self, cfg: VGG9Config, verbose: bool = False):
        super().__init__()
        self._cfg = cfg
        self._n_classes = int(cfg.num_classes)

        # Expose capability flags read by trainers and evaluators.
        self.beta = float(cfg.beta)
        # out_integrator=True tells evaluators to use the final mem, not accumulated spikes.
        self.out_integrator = (cfg.head_type == _HEAD_LI)
        # Exposed for TPTrainer._init_S when input_shape is not passed explicitly.
        self.input_shape = tuple(cfg.input_shape)

        spike_grad = _build_spike_grad(cfg)

        lif_kwargs = dict(
            beta=float(cfg.beta),
            threshold=float(cfg.threshold),
            spike_grad=spike_grad,
            reset_mechanism=str(cfg.reset_mechanism),
            init_hidden=True,   # snntorch manages lif.mem internally
            output=True,        # return (spk, mem) tuple from lif(x)
            learn_beta=False,
            learn_threshold=False,
        )

        # ---- build blocks ------------------------------------------------
        self._num_blocks = len(cfg.channels)
        prev_ch = int(cfg.in_channels)
        for i, (ch, (pool_type, pool_arg)) in enumerate(
            zip(cfg.channels, cfg.pool_spec), start=1
        ):
            setattr(self, f"conv{i}", _build_conv(cfg, prev_ch, int(ch)))
            setattr(self, f"lif{i}",  snn.Leaky(**lif_kwargs))
            if cfg.scale_after_lif > 0.0:
                setattr(self, f"scale{i}", Scale(float(cfg.scale_after_lif)))
            pool = _build_pool(pool_type, int(pool_arg))
            if pool is not None:
                setattr(self, f"pool{i}", pool)
            prev_ch = int(ch)

        # ---- head --------------------------------------------------------
        if cfg.head_type == _HEAD_LI:
            in_features = int(prev_ch * cfg.li_head_spatial * cfg.li_head_spatial)
            self.head = LeakyIntegrator(in_features, self._n_classes, leak=cfg.li_head_leak)
            self.global_pool = None
        else:  # _HEAD_GLOBAL_LINEAR
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(int(prev_ch), self._n_classes)

        self._initialize_weights()

        if verbose:
            self._print_config()

    # ------------------------------------------------------------------
    # Single-timestep forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        spk_list: List[torch.Tensor] = []
        mem_list: List[torch.Tensor] = []

        for i, (pool_type, _pool_arg) in enumerate(self._cfg.pool_spec, start=1):
            x = getattr(self, f"conv{i}")(x)

            lif = getattr(self, f"lif{i}")
            spk, mem = lif(x)   # init_hidden=True, output=True → returns (spk, mem)

            if hasattr(self, f"scale{i}"):
                spk = getattr(self, f"scale{i}")(spk)

            if pool_type != _POOL_NONE:
                spk = getattr(self, f"pool{i}")(spk)

            spk_list.append(spk)
            mem_list.append(mem)
            x = spk

        if self._cfg.head_type == _HEAD_LI:
            out = self.head(x.flatten(1))
        else:
            out = self.head(self.global_pool(x).flatten(1))

        spk_list.append(out)
        mem_list.append(out)
        return spk_list, mem_list

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def init_states(self) -> None:
        for i in range(1, self._num_blocks + 1):
            getattr(self, f"lif{i}").init_leaky()
        if isinstance(self.head, LeakyIntegrator):
            self.head.reset()

    def reset(self) -> None:
        self.init_states()

    @property
    def n_classes(self) -> int:
        return self._n_classes

    # ------------------------------------------------------------------
    # BaseSNN trainer-facing interface
    # ------------------------------------------------------------------

    def hidden_weight_layers(self):
        out: List[Tuple[nn.Module, Optional[Callable]]] = []
        for i, (pool_type, _) in enumerate(self._cfg.pool_spec, start=1):
            conv = getattr(self, f"conv{i}")
            pool = getattr(self, f"pool{i}", None) if pool_type != _POOL_NONE else None
            out.append((conv, pool))
        return out

    def output_layer(self):
        # Trainers need the underlying nn.Linear. For the LI head that is head.fc;
        # for the global-linear head it is head itself.
        if isinstance(self.head, LeakyIntegrator):
            return self.head.fc
        return self.head

    @property
    def VGG9_CFG(self):
        # Compatibility shim: TPTrainer duck-types on hasattr(net, 'VGG9_CFG').
        # Returns a list of 3-tuples so the trainer's enumerate loop unpacks correctly.
        # Trainers use this only to count blocks and locate conv{i}/pool{i} by name.
        return [
            (ch, pool_arg, pool_type)
            for ch, (pool_type, pool_arg) in zip(self._cfg.channels, self._cfg.pool_spec)
        ]

    # ------------------------------------------------------------------
    # Init + diagnostics
    # ------------------------------------------------------------------

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                if self._cfg.head_type == _HEAD_GLOBAL_LINEAR:
                    # OTTT paper uses tight N(0, 0.01) init on the Linear head.
                    nn.init.normal_(m.weight, 0.0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(m.weight)

    def _print_config(self) -> None:
        cfg = self._cfg
        n_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print(f"  VGG9 (gain={cfg.conv_gain} / {cfg.head_type})")
        print(f"{'='*60}")
        print(f"  {'Input Shape':<25} {cfg.input_shape}")
        print(f"  {'Num Classes':<25} {cfg.num_classes}")
        print(f"  {'Channels':<25} {list(cfg.channels)}")
        print(f"  {'Pool Spec':<25} {list(cfg.pool_spec)}")
        print(f"  {'Conv Gain':<25} {cfg.conv_gain}")
        print(f"  {'Beta':<25} {cfg.beta}")
        print(f"  {'Threshold':<25} {cfg.threshold}")
        print(f"  {'Surrogate':<25} {cfg.surrogate_kind}")
        print(f"  {'Head':<25} {cfg.head_type}")
        print(f"  {'Parameters':<25} {n_params:,}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Pool-spec helpers
# ---------------------------------------------------------------------------

def _tp_pool_spec() -> List[Tuple[str, int]]:
    # MaxPool after blocks 2, 4, 6; AdaptiveAvgPool(2,2) after block 8.
    return [
        (_POOL_NONE, 0),
        (_POOL_MAX,  2),
        (_POOL_NONE, 0),
        (_POOL_MAX,  2),
        (_POOL_NONE, 0),
        (_POOL_MAX,  2),
        (_POOL_NONE, 0),
        (_POOL_AAVG, 2),
    ]


def _ottt_pool_spec(pool_after_blocks: Sequence[int]) -> List[Tuple[str, int]]:
    # AvgPool(2,2) after the blocks listed in pool_after_blocks; 'none' elsewhere.
    pool_set = set(int(i) for i in pool_after_blocks)
    return [(_POOL_AVG, 2) if i in pool_set else (_POOL_NONE, 0) for i in range(1, 9)]


# ---------------------------------------------------------------------------
# Unified factory — registered as "vgg9" in NETWORK_REGISTRY
# ---------------------------------------------------------------------------

def vgg9(
    in_channels: int,
    num_classes: int,
    input_shape: Tuple,
    head_type: str,
    pool_after_blocks: Optional[Sequence[int]] = None,
    use_tp_pool: bool = False,
    **kwargs,
) -> VGG9:
    """
    Unified VGG-9 factory.

    Args:
        in_channels:       Number of input channels (e.g. 3 for RGB, 2 for DVS).
        num_classes:       Number of output classes.
        input_shape:       (C, H, W) spatial shape — used to size the LI head.
        head_type:         "leaky_integrator" (TP-style) or "global_linear" (OTTT-style).
        pool_after_blocks: Block indices after which AvgPool(2,2) is inserted
                           (OTTT topology). Ignored when use_tp_pool=True.
        use_tp_pool:       If True, use the TP fixed pool spec (MaxPool after
                           {2,4,6}, AdaptiveAvgPool(2,2) after 8).
        **kwargs:          Passed through to VGG9Config (beta, threshold,
                           conv_gain, scale_after_lif, surrogate_kind,
                           surrogate_slope, surrogate_scale, li_head_spatial,
                           li_head_leak, channels, verbose).
    """
    if use_tp_pool:
        pool_spec = _tp_pool_spec()
    else:
        pool_spec = _ottt_pool_spec(pool_after_blocks or (2, 4))

    cfg = VGG9Config(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        input_shape=tuple(input_shape),
        channels=kwargs.pop("channels", (64, 128, 256, 256, 512, 512, 512, 512)),
        pool_spec=pool_spec,
        conv_gain=float(kwargs.pop("conv_gain", 1.0)),
        scale_after_lif=float(kwargs.pop("scale_after_lif", 0.0)),
        beta=float(kwargs.pop("beta", 0.5)),
        threshold=float(kwargs.pop("threshold", 1.0)),
        reset_mechanism=str(kwargs.pop("reset_mechanism", "subtract")),
        surrogate_kind=str(kwargs.pop("surrogate_kind", "atan")),
        surrogate_scale=float(kwargs.pop("surrogate_scale", 1.0)),
        surrogate_slope=float(kwargs.pop("surrogate_slope", 4.0)),
        head_type=str(head_type),
        li_head_spatial=int(kwargs.pop("li_head_spatial", 2)),
        li_head_leak=float(kwargs.pop("li_head_leak", 1.0)),
    )
    return VGG9(cfg, verbose=bool(kwargs.pop("verbose", False)))


# ---------------------------------------------------------------------------
# Legacy preset factories — kept for existing tests (bptt_*, legacy test files)
# ---------------------------------------------------------------------------

def vgg9_cifar10(in_channels: int = 3, num_classes: int = 10, **kwargs) -> VGG9:
    """TP-style VGG-9 for CIFAR-10 (3×32×32, 10 classes)."""
    cfg = VGG9Config(
        in_channels=in_channels, num_classes=num_classes,
        input_shape=(in_channels, 32, 32),
        channels=(64, 128, 256, 256, 512, 512, 512, 512),
        pool_spec=_tp_pool_spec(),
        conv_gain=1.8,
        beta=kwargs.pop("beta", 0.53),
        threshold=kwargs.pop("threshold", 1.0),
        surrogate_kind="atan", surrogate_scale=1.0,
        head_type=_HEAD_LI,
    )
    return VGG9(cfg, verbose=kwargs.pop("verbose", False))


def vgg9_svhn(in_channels: int = 3, num_classes: int = 10, **kwargs) -> VGG9:
    """TP-style VGG-9 for SVHN (3×32×32, 10 classes)."""
    cfg = VGG9Config(
        in_channels=in_channels, num_classes=num_classes,
        input_shape=(in_channels, 32, 32),
        channels=(64, 128, 256, 256, 512, 512, 512, 512),
        pool_spec=_tp_pool_spec(),
        conv_gain=1.8,
        beta=kwargs.pop("beta", 0.53),
        threshold=kwargs.pop("threshold", 1.0),
        surrogate_kind="atan", surrogate_scale=1.0,
        head_type=_HEAD_LI,
    )
    return VGG9(cfg, verbose=kwargs.pop("verbose", False))


def vgg9_dvsgest(in_channels: int = 2, num_classes: int = 11, **kwargs) -> VGG9:
    """TP-style VGG-9 for DVSGesture (2×128×128, 11 classes)."""
    cfg = VGG9Config(
        in_channels=in_channels, num_classes=num_classes,
        input_shape=(in_channels, 128, 128),
        channels=(64, 128, 256, 256, 512, 512, 512, 512),
        pool_spec=_tp_pool_spec(),
        conv_gain=1.8,
        beta=kwargs.pop("beta", 0.53),
        threshold=kwargs.pop("threshold", 1.0),
        surrogate_kind="atan", surrogate_scale=1.0,
        head_type=_HEAD_LI,
    )
    return VGG9(cfg, verbose=kwargs.pop("verbose", False))


def vgg9_ottt_dvsgest(in_channels: int = 2, num_classes: int = 11, **kwargs) -> VGG9:
    """OTTT-style VGG-9 for DVSGesture (2×128×128, 11 classes). Pool after {2,4,6}."""
    cfg = VGG9Config(
        in_channels=in_channels, num_classes=num_classes,
        input_shape=(in_channels, 128, 128),
        channels=(64, 128, 256, 256, 512, 512, 512, 512),
        pool_spec=_ottt_pool_spec(pool_after_blocks=(2, 4, 6)),
        conv_gain=kwargs.pop("conv_gain", 1.0),
        scale_after_lif=2.74,
        beta=kwargs.pop("beta", 0.5),
        threshold=kwargs.pop("threshold", 1.0),
        surrogate_kind="sigmoid", surrogate_slope=4.0,
        head_type=_HEAD_GLOBAL_LINEAR,
    )
    return VGG9(cfg, verbose=kwargs.pop("verbose", False))


def vgg9_ottt_cifar10(in_channels: int = 3, num_classes: int = 10, **kwargs) -> VGG9:
    """OTTT-style VGG-9 for CIFAR-10 (3×32×32, 10 classes). Pool after {2,4}."""
    cfg = VGG9Config(
        in_channels=in_channels, num_classes=num_classes,
        input_shape=(in_channels, 32, 32),
        channels=(64, 128, 256, 256, 512, 512, 512, 512),
        pool_spec=_ottt_pool_spec(pool_after_blocks=(2, 4)),
        conv_gain=kwargs.pop("conv_gain", 1.0),
        scale_after_lif=2.74,
        beta=kwargs.pop("beta", 0.5),
        threshold=kwargs.pop("threshold", 1.0),
        surrogate_kind="sigmoid", surrogate_slope=4.0,
        head_type=_HEAD_GLOBAL_LINEAR,
    )
    return VGG9(cfg, verbose=kwargs.pop("verbose", False))


def vgg9_ottt_fashionmnist(in_channels: int = 1, num_classes: int = 10, **kwargs) -> VGG9:
    """OTTT-style VGG-9 for Fashion-MNIST (1×28×28). Lighter channels, pool after {1,3}."""
    cfg = VGG9Config(
        in_channels=in_channels, num_classes=num_classes,
        input_shape=(in_channels, 28, 28),
        channels=(32, 64, 128, 128, 256, 256, 256, 256),
        pool_spec=_ottt_pool_spec(pool_after_blocks=(1, 3)),
        conv_gain=kwargs.pop("conv_gain", 1.0),
        scale_after_lif=2.74,
        beta=kwargs.pop("beta", 0.5),
        threshold=kwargs.pop("threshold", 1.0),
        surrogate_kind="sigmoid", surrogate_slope=4.0,
        head_type=_HEAD_GLOBAL_LINEAR,
    )
    return VGG9(cfg, verbose=kwargs.pop("verbose", False))
