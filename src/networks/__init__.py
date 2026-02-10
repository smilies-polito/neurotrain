"""Neural network architectures for SNNs."""

from networks.base_network import ExampleNet
from networks.conv_network import ConvFCNetwork
from networks.fc_network import FCNetwork
from networks.recurrent_srnn import RecurrentSRNN
from networks.get_network import get_network
from networks.base_network import ExampleNet

__all__ = [
    "BaseSNN",
    "FCNetwork",
    "ConvFCNetwork",
    "RecurrentSRNN",
    "get_network",
    "ExampleNet",
]
