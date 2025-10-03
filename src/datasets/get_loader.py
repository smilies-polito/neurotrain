from pathlib import Path
import os

import torch                                                            # type: ignore
import numpy as np                                                      # type: ignore
from torch.utils.data import DataLoader                                 # type: ignore
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST           # type: ignore
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda # type: ignore
from torchvision.datasets import SVHN                                   # type: ignore 
from tonic.datasets import DVSGesture                                   # type: ignore
# import custom dataset loaders
from datasets.mnist_loader import MNISTLoader
from datasets.cifar10_loader import CIFAR10Loader
from datasets.fashionmnist_loader import FashionMNISTLoader
from datasets.svhn_loader import SVHNLoader
from datasets.dvsgesture_loader import DVSGestureLoader

# Storage path for datasets
_REPO_ROOT = Path(__file__).resolve().parent  # .../stsf/main
DATA_ROOT  = Path(os.environ.get("STSF_DATA", (_REPO_ROOT / "../Data").resolve()))

def get_loader(name, batch_size, T):
    print(name)
    if name == "MNIST":
        return MNISTLoader(batch_size, T)
    elif name == "CIFAR10":
        return CIFAR10Loader(batch_size, T)
    elif name == "FashionMNIST":
        return FashionMNISTLoader(batch_size, T)
    elif name == "SVHN":
        return SVHNLoader(batch_size, T)
    elif name == "DVSGesture":
        return DVSGestureLoader(batch_size, T)
    #elif name == "NMNIST":
    #    return NMNISTLoader(batch_size, T)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            "Choose from 'MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', or 'NMNIST'."
        )
