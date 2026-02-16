"""
Network factory with algorithm-model compatibility matrix.

Returns the appropriate network for a given algorithm and model architecture.

Supports two modes via network_mode:
- "benchmarking" (B): Uses networks from networks/benchmarking/ (FCSNN, RSNN, etc.)
- "reproducibility" (R): Uses networks from flat structure (FCNetwork, LocalClassifier, etc.)
"""

from typing import Any, Literal, Optional, Union

import torch

from networks.base_snn import BaseSNN
from networks.fc_network import FCNetwork
from networks.recurrent_srnn import RecurrentSRNN
from networks.spiking_resnet18 import SpikingResNet18
from networks.spiking_vgg11 import SpikingVGG11

NetworkMode = Literal["benchmarking", "reproducibility"]

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


# Algorithms that require architectures not available in benchmarking/ (no LocalClassifier, STLLR)
_BENCHMARKING_UNSUPPORTED_ALGORITHMS = ("ell", "fell", "bell", "stllr")


def get_benchmarking_network(
    algorithm_name: str,
    layer_sizes: list,
    beta: float = 0.9,
    dataset: Optional[str] = None,
    **kwargs: Any,
) -> BaseSNN:
    """
    Create a network from networks/benchmarking/ for benchmarking mode.

    Uses FCSNN for feedforward algorithms, RSNN for recurrent (eprop, esd_rtrl).
    Maps layer_sizes [in, hidden..., out] to in_shape, hidden_sizes, num_classes.

    Raises ValueError for ell, fell, bell, stllr (no benchmarking equivalent).
    """
    if algorithm_name in _BENCHMARKING_UNSUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{algorithm_name}' has no benchmarking network equivalent. "
            f"Use mode R (reproducibility) for ell, fell, bell, stllr."
        )
    num_classes = layer_sizes[-1]
    hidden_sizes = tuple(layer_sizes[1:-1])
    input_size = layer_sizes[0]

    # Use flattened in_shape: (input_size,) for compatibility with rate-coded loaders
    in_shape = (input_size,)

    if algorithm_name in ("eprop", "esd_rtrl"):
        from networks.benchmarking.r_snn import RSNN

        return RSNN(
            in_shape=in_shape,
            num_classes=num_classes,
            hidden_sizes=hidden_sizes,
            beta=beta,
            threshold=kwargs.get("threshold", 1.0),
        )
    else:
        from networks.benchmarking.fc_snn import FCSNN

        return FCSNN(
            in_shape=in_shape,
            num_classes=num_classes,
            hidden_sizes=hidden_sizes,
            beta=beta,
            threshold=kwargs.get("threshold", 1.0),
        )


def get_network(
    algorithm_name: str,
    model_architecture: str,
    layer_sizes: list,
    beta: float = 0.9,
    network_mode: NetworkMode = "reproducibility",
    **kwargs: Any,
) -> BaseSNN:
    """
    Create the appropriate network for the given algorithm and model architecture.

    Args:
        algorithm_name: Name of the learning algorithm (bptt, stsf, eprop, ell, fell, bell, etc.).
        model_architecture: Requested model ("fc", "local_classifier", "recurrent").
        layer_sizes: Network layer sizes [input, hidden..., output].
        beta: LIF neuron leak factor (for fc and local_classifier).
        network_mode: "benchmarking" (B) uses networks/benchmarking/, "reproducibility" (R) uses flat structure.
        **kwargs: Additional arguments passed to network constructors.

    Returns:
        Network instance conforming to BaseSNN.

    Raises:
        ValueError: If algorithm-model combination is incompatible.
    """
    if network_mode == "benchmarking":
        return get_benchmarking_network(
            algorithm_name=algorithm_name,
            layer_sizes=layer_sizes,
            beta=beta,
            dataset=kwargs.get("dataset"),
            **kwargs,
        )

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

        raise ValueError(
            "Recurrent architecture is only compatible with eprop or esd_rtrl, "
            f"got {algorithm_name}"
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
