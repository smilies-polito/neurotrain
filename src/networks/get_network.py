"""
Network factory with algorithm-model compatibility matrix.

Returns the appropriate network for a given algorithm and model architecture.
"""

from typing import Any, Optional, Union

import torch

from networks.base_snn import BaseSNN
from networks.fc_network import FCNetwork
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
        if algorithm_name not in ("eprop", "esd_rtrl"):
            raise ValueError(
                f"RecurrentSRNN is only compatible with eprop or esd_rtrl, got {algorithm_name}"
            )
        if len(layer_sizes) < 3:
            raise ValueError(
                "Recurrent (eprop/esd_rtrl) requires layer_sizes=[n_in, n_rec, n_out]"
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
            "'vgg11', or 'resnet18'."
        )
    return FCNetwork(
        layer_sizes=layer_sizes,
        beta=beta,
        quant=kwargs.get("quant", False),
        threshold=kwargs.get("threshold", 1.0),
    )
