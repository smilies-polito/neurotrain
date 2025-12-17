"""Neural network architectures for SNNs."""

from networks.fc_network import FCNetwork
from networks.recurrent_srnn import RecurrentSRNN
from networks.base_network import ExampleNet

__all__ = [
    "FCNetwork",
    "RecurrentSRNN",
    "ExampleNet",
]
