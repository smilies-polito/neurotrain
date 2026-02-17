"""
Network factory with algorithm-model compatibility matrix.

Returns the appropriate network for a given algorithm and model architecture.
"""

from typing import Any, Optional, Union

import torch

from networks.base_snn import BaseSNN
from networks.benchmarking.conv_snn import ConvSNN
from networks.benchmarking.fc_snn import FCSNN
from networks.benchmarking.r_snn import RSNN
from networks.benchmarking.vg11_snn import VG11SNN
from networks.fc_network import FCNetwork
from networks.recurrent_fc_network import RecurrentFCNetwork
from networks.recurrent_srnn import RecurrentSRNN
from networks.spiking_resnet18 import SpikingResNet18
from networks.spiking_vgg11 import SpikingVGG11

# Compatibility: (algorithm, model_architecture) -> use this model
# Algorithms that require specific models will override model_architecture
_ALGORITHM_MODEL_OVERRIDE = {
    "eprop": "recurrent",
    "esd_rtrl": "recurrent",
    "ell": "local_classifier",
    "fell": "local_classifier",
    "bell": "local_classifier",
    "stllr": "stllr",
}


def _infer_input_shape(input_size: int) -> tuple[int, int, int]:
    """Infer canonical image-like shape from flattened input size."""
    if int(input_size) == 784:
        return (1, 28, 28)
    if int(input_size) == 1156:
        return (1, 34, 34)
    if int(input_size) == 3072:
        return (3, 32, 32)
    if int(input_size) == 16384:
        return (1, 128, 128)
    raise ValueError(
        f"Cannot infer input shape from input size {input_size}. "
        "Provide input_shape explicitly."
    )


def get_network(
    algorithm_name: str,
    model_architecture: str,
    layer_sizes: list,
    beta: float = 0.9,
    **kwargs: Any,
) -> BaseSNN:
    """
    Create the appropriate network for the given algorithm and model architecture.

    Args:
        algorithm_name: Name of the learning algorithm (bptt, stsf, eprop, ell, fell, bell, etc.).
        model_architecture: Requested model ("fc", "local_classifier", "recurrent").
        layer_sizes: Network layer sizes [input, hidden..., output].
        beta: LIF neuron leak factor (for fc and local_classifier).
        **kwargs: Additional arguments passed to network constructors.

    Returns:
        Network instance conforming to BaseSNN.

    Raises:
        ValueError: If algorithm-model combination is incompatible.
    """
    # Some algorithms require a specific model regardless of config
    effective_arch = _ALGORITHM_MODEL_OVERRIDE.get(algorithm_name, model_architecture)

    if effective_arch == "recurrent":
        recurrent_type = str(kwargs.get("recurrent_type", "standard")).lower()
        if len(layer_sizes) < 3:
            raise ValueError("Recurrent requires layer_sizes=[n_in, n_rec, ..., n_out]")
        if algorithm_name in ("eprop", "esd_rtrl"):
            if recurrent_type not in ("standard", "srnn"):
                raise ValueError(
                    "For eprop/esd_rtrl, recurrent_type must be one of: standard, srnn."
                )
            return RecurrentSRNN(
                n_in=layer_sizes[0],
                n_rec=layer_sizes[1],
                n_out=layer_sizes[-1],
                threshold=kwargs.get("threshold", 1.0),
                tau_mem=2.0,
                tau_out=0.02,
                dt=1e-3,
            )

        if algorithm_name == "ostl":
            if recurrent_type not in ("snu", "ssnu"):
                raise ValueError(
                    "For ostl recurrent model, recurrent_type must be one of: snu, ssnu."
                )
            return RecurrentFCNetwork(
                layer_sizes=layer_sizes,
                beta=beta,
                quant=kwargs.get("quant", False),
                threshold=kwargs.get("threshold", 1.0),
                recurrent_type=recurrent_type,
            )

        raise ValueError(
            f"Recurrent architecture is only compatible with eprop, esd_rtrl, or ostl, got {algorithm_name}"
        )

    if effective_arch == "stllr":
        if algorithm_name != "stllr":
            raise ValueError(
                f"STLLRNetwork is only compatible with stllr, got {algorithm_name}"
            )
        from networks.stllr_network import STLLRNetwork

        return STLLRNetwork(
            layer_sizes=layer_sizes,
            threshold=kwargs.get("threshold", 0.6),
            leak=kwargs.get("leak", 2.0),
            factors=kwargs.get("factors_stdp"),
            **kwargs,
        )

    if effective_arch == "local_classifier":
        if algorithm_name not in (
            "ell",
            "fell",
            "bell",
            "bptt",
            "stsf",
            "decolle",
            "ottt",
        ):
            raise ValueError(
                f"LocalClassifierNetwork is for ell/fell/bell or fc-compatible algorithms, "
                f"got {algorithm_name}"
            )
        from networks.local_classifier_network import LocalClassifierNetwork

        # ell/fell/bell use their specific mode; others use bell (full gradient flow)
        mode = (
            "ell"
            if algorithm_name == "ell"
            else "fell"
            if algorithm_name == "fell"
            else "bell"
        )
        lc_kwargs = {
            k: v for k, v in kwargs.items() if k in ("threshold", "bias", "fa")
        }
        return LocalClassifierNetwork(
            layer_sizes=layer_sizes,
            beta=beta,
            tau=kwargs.get("tau"),
            mode=mode,
            **lc_kwargs,
        )

    if effective_arch == "vgg11":
        return SpikingVGG11(
            input_channels=kwargs.get("input_channels", 3),
            num_classes=kwargs.get("num_classes", layer_sizes[-1]),
            beta=beta,
            threshold=kwargs.get("threshold", 1.0),
            base_channels=kwargs.get("base_channels", 64),
            surrogate=kwargs.get("surrogate", "exp"),
        )

    if effective_arch == "vg11_snn":
        if len(layer_sizes) < 2:
            raise ValueError("vg11_snn requires layer_sizes=[n_in, hidden..., n_out].")
        input_shape = kwargs.get("input_shape")
        if input_shape is None:
            input_shape = _infer_input_shape(int(layer_sizes[0]))
        input_shape = tuple(int(v) for v in input_shape)
        feature_cfg = kwargs.get("feature_cfg")
        if feature_cfg is None:
            # Keep pooling depth compatible with spatial size.
            # 28x28 inputs cannot sustain 5 pools; use 4 pools there.
            if min(input_shape[1], input_shape[2]) <= 28:
                feature_cfg = [1, "M", 2, "M", 4, 4, "M", 8, 8, "M"]
            else:
                feature_cfg = [1, "M", 2, "M", 4, 4, "M", 8, 8, "M", 8, 8, "M"]
        return VG11SNN(
            in_shape=input_shape,
            num_classes=int(layer_sizes[-1]),
            feature_cfg=feature_cfg,
            classifier_hidden_sizes=tuple(int(v) for v in layer_sizes[1:-1]),
            base_channels=int(kwargs.get("base_channels", 16 if input_shape[0] == 3 else 10)),
            use_batch_norm=bool(kwargs.get("use_batch_norm", True)),
            pool_kernel=int(kwargs.get("pool_kernel", 2)),
            pool_stride=int(kwargs.get("pool_stride", 2)),
            beta=beta,
            threshold=kwargs.get("threshold", 1.0),
        )

    if effective_arch == "fc_snn":
        if len(layer_sizes) < 2:
            raise ValueError("fc_snn requires layer_sizes=[n_in, hidden..., n_out].")
        return FCSNN(
            in_shape=(int(layer_sizes[0]),),
            num_classes=int(layer_sizes[-1]),
            hidden_sizes=tuple(int(v) for v in layer_sizes[1:-1]),
            beta=beta,
            threshold=kwargs.get("threshold", 1.0),
        )

    if effective_arch == "r_snn":
        if len(layer_sizes) < 2:
            raise ValueError("r_snn requires layer_sizes=[n_in, hidden..., n_out].")
        return RSNN(
            in_shape=(int(layer_sizes[0]),),
            num_classes=int(layer_sizes[-1]),
            hidden_sizes=tuple(int(v) for v in layer_sizes[1:-1]),
            beta=beta,
            threshold=kwargs.get("threshold", 1.0),
        )

    if effective_arch == "conv_snn":
        if len(layer_sizes) < 1:
            raise ValueError(
                "conv_snn requires layer_sizes=[classifier_hidden..., n_out]."
            )
        input_shape = kwargs.get("input_shape")
        if input_shape is None:
            input_shape = _infer_input_shape(int(layer_sizes[0]))
        input_shape = tuple(int(v) for v in input_shape)

        conv_layers = kwargs.get("conv_layers", [])
        if conv_layers:
            conv_channels = tuple(int(layer["out_channels"]) for layer in conv_layers)
            pool_after = tuple(
                bool(int(layer.get("pool_kernel", 0)) > 0) for layer in conv_layers
            )
            first = conv_layers[0]
            conv_kernel_size = int(first.get("kernel_size", 3))
            conv_stride = int(first.get("stride", 1))
            conv_padding = int(first.get("padding", 1))
            pool_kernel = int(first.get("pool_kernel", 2))
            pool_stride = int(first.get("pool_stride", 2))
        else:
            conv_channels = (32, 64)
            pool_after = (True, True)
            conv_kernel_size = 3
            conv_stride = 1
            conv_padding = 1
            pool_kernel = 2
            pool_stride = 2

        return ConvSNN(
            in_shape=input_shape,
            num_classes=int(layer_sizes[-1]),
            conv_channels=conv_channels,
            fc_hidden_sizes=tuple(int(v) for v in layer_sizes[:-1]),
            use_batch_norm=bool(kwargs.get("use_batch_norm", True)),
            pool_after=pool_after,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
            beta=beta,
            threshold=kwargs.get("threshold", 1.0),
        )

    if effective_arch == "resnet18":
        return SpikingResNet18(
            input_channels=kwargs.get("input_channels", 3),
            num_classes=kwargs.get("num_classes", layer_sizes[-1]),
            beta=beta,
            threshold=kwargs.get("threshold", 1.0),
            base_channels=kwargs.get("base_channels", 64),
            surrogate=kwargs.get("surrogate", "exp"),
        )

    # Default: FCNetwork
    if effective_arch != "fc":
        raise ValueError(
            f"Unknown model architecture '{effective_arch}'. "
            "Use 'fc', 'conv', 'local_classifier', 'recurrent', 'stllr', "
            "'vgg11', 'resnet18', 'fc_snn', 'r_snn', 'conv_snn', or 'vg11_snn'."
        )
    return FCNetwork(
        layer_sizes=layer_sizes,
        beta=beta,
        quant=kwargs.get("quant", False),
        threshold=kwargs.get("threshold", 1.0),
    )
