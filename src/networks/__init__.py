"""Network registry for SNN training.

To add a new network:
  1. Create src/networks/my_network.py with a class or factory function.
  2. Add two lines here:
       from networks.my_network import MyNetwork
       NETWORK_REGISTRY["my_network"] = MyNetwork
"""

from networks.fc_snn import FCSNN
from networks.r_snn import RSNN
from networks.conv_snn import ConvSNN
from networks.vgg9 import (
    vgg9_cifar10,
    vgg9_svhn,
    vgg9_dvsgest,
    vgg9_ottt_cifar10,
    vgg9_ottt_dvsgest,
    vgg9_ottt_fashionmnist,
)

NETWORK_REGISTRY: dict[str, callable] = {
    "fc_snn":                 FCSNN,
    "r_snn":                  RSNN,
    "conv_snn":               ConvSNN,
    "vgg9_cifar10":           vgg9_cifar10,
    "vgg9_svhn":              vgg9_svhn,
    "vgg9_dvsgest":           vgg9_dvsgest,
    "vgg9_ottt_cifar10":      vgg9_ottt_cifar10,
    "vgg9_ottt_dvsgest":      vgg9_ottt_dvsgest,
    "vgg9_ottt_fashionmnist": vgg9_ottt_fashionmnist,
}

__all__ = ["NETWORK_REGISTRY"]
