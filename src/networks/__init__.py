"""Neural network architectures for SNNs."""

from networks.base_network import ExampleNet
from networks.conv_network import ConvFCNetwork
from networks.etlp_network import ETLPNetwork
from networks.fc_network import FCNetwork
from networks.recurrent_srnn import RecurrentSRNN

__all__ = [
    "FCNetwork",
    "ConvFCNetwork",
    "RecurrentSRNN",
    "ExampleNet",
    "ETLPNetwork",
]
