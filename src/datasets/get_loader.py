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
from datasets.nmnist_loader import NMNISTLoader
# NeuroBench official benchmark loaders
from datasets.neurobench_loaders import (
    SpeechCommandsLoader,
    WISDMLoader,
    PrimateReachingLoader,
    MackeyGlassLoader,
    NEUROBENCH_DATASETS,
)

# Storage path for datasets
_REPO_ROOT = Path(__file__).resolve().parent  # .../stsf/main
DATA_ROOT  = Path(os.environ.get("STSF_DATA", (_REPO_ROOT / "../Data").resolve()))

# Standard image classification datasets (rate-coded)
RATE_CODED_DATASETS = ["MNIST", "CIFAR10", "FashionMNIST", "SVHN"]

# Event-based neuromorphic datasets (ideal for DECOLLE)
EVENT_BASED_DATASETS = ["NMNIST", "DVSGesture"]

# All standard datasets
STANDARD_DATASETS = RATE_CODED_DATASETS + EVENT_BASED_DATASETS

# NeuroBench official benchmarks
NEUROBENCH_CLASSIFICATION = ["SpeechCommands", "WISDM"]
NEUROBENCH_REGRESSION = ["PrimateReaching", "MackeyGlass"]

ALL_DATASETS = STANDARD_DATASETS + NEUROBENCH_CLASSIFICATION + NEUROBENCH_REGRESSION


def get_loader(name, batch_size, T):
    print(name)
    # Standard image datasets
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
    # Event-based neuromorphic datasets
    elif name == "NMNIST":
        return NMNISTLoader(batch_size, T)
    # NeuroBench official benchmarks (classification)
    elif name == "SpeechCommands":
        return SpeechCommandsLoader(batch_size, T)
    elif name == "WISDM":
        return WISDMLoader(batch_size, T)
    # NeuroBench official benchmarks (regression)
    elif name == "PrimateReaching":
        return PrimateReachingLoader(batch_size, T)
    elif name == "MackeyGlass":
        return MackeyGlassLoader(batch_size, T)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {ALL_DATASETS}"
        )
