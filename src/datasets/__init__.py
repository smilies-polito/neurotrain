"""Dataset registry for SNN training.

To add a new dataset:
  1. Create src/datasets/my_loader.py with a loader function returning (train_loader, test_loader).
  2. Add two lines here:
       from datasets.my_loader import MyLoader
       LOADER_REGISTRY["MyDataset"] = MyLoader
"""

from datasets.mnist_loader import MNISTLoader
from datasets.cifar10_loader import CIFAR10Loader
from datasets.fashionmnist_loader import FashionMNISTLoader
from datasets.svhn_loader import SVHNLoader
from datasets.dvsgesture_loader import DVSGestureLoader
from datasets.nmnist_loader import NMNISTLoader
from datasets.shd_loader import SHDLoader
from datasets.dvscifar10_loader import DVSCifar10Loader
from datasets.neurobench_loaders import (
    SpeechCommandsLoader,
    WISDMLoader,
    PrimateReachingLoader,
    MackeyGlassLoader,
)

LOADER_REGISTRY: dict[str, callable] = {
    "MNIST":           MNISTLoader,
    "CIFAR10":         CIFAR10Loader,
    "FashionMNIST":    FashionMNISTLoader,
    "SVHN":            SVHNLoader,
    "DVSGesture":      DVSGestureLoader,
    "NMNIST":          NMNISTLoader,
    "SHD":             SHDLoader,
    "DVSCIFAR10":      DVSCifar10Loader,
    "SpeechCommands":  SpeechCommandsLoader,
    "WISDM":           WISDMLoader,
    "PrimateReaching": PrimateReachingLoader,
    "MackeyGlass":     MackeyGlassLoader,
}

__all__ = ["LOADER_REGISTRY"]
