"""Neural network architectures for SNNs."""

from networks.fc_network import FCNetwork
from networks.base_network import ExampleNet
from networks.decolle_network import DecolleNetwork

__all__ = [
    "FCNetwork",
    "ExampleNet",
    "DecolleNetwork",
]
