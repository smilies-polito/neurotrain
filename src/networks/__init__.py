"""Neural network architectures for SNNs."""

from networks.base_network import ExampleNet
from networks.base_snn import BaseSNN
from networks.conv_network import ConvFCNetwork
from networks.fc_network import FCNetwork
from networks.get_network import get_network
from networks.base_network import ExampleNet
from networks.spiking_resnet18 import SpikingResNet18
from networks.spiking_vgg11 import SpikingVGG11
from networks.recurrent_fc_network import RecurrentFCNetwork
from networks.recurrent_srnn import RecurrentSRNN

__all__ = [
    "BaseSNN",
    "FCNetwork",
    "RecurrentFCNetwork",
    "ConvFCNetwork",
    "FCNetwork",
    "RecurrentSRNN",
    "SpikingVGG11",
    "SpikingResNet18",
    "get_network",
    "ExampleNet",
]
