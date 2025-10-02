from pathlib import Path
import os

import torch                                                            # type: ignore
import numpy as np                                                      # type: ignore
from torch.utils.data import DataLoader                                 # type: ignore
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST           # type: ignore
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda # type: ignore
from torchvision.datasets import SVHN                                   # type: ignore 
from tonic.datasets import DVSGesture                                   # type: ignore

# Storage path for datasets
_REPO_ROOT = Path(__file__).resolve().parent  # .../stsf/main
DATA_ROOT  = Path(os.environ.get("STSF_DATA", (_REPO_ROOT / "../Data").resolve()))


class Rate:
    """
    Simulate rate-coded spike trains from static images over T timesteps.
    """
    def __init__(self, T):
        self.T = T

    def __call__(self, input):
        # Flatten image to vector
        input = input.view(-1)
        # Allocate spike tensor [T x features]
        output = torch.zeros((self.T, *input.shape), device=input.device)
        for t in range(self.T):
            # Probabilistic firing based on pixel intensity
            output[t] = torch.rand_like(input).le(input).float()
        return output


def MNISTLoader(batch_size, T):
    """
    Returns DataLoaders for MNIST, with rate-coded spikes over T timesteps.
    """
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Rate(T),
        Lambda(lambda x: torch.flatten(x, start_dim=1))
    ])
    trainloader = DataLoader(
        MNIST(DATA_ROOT.as_posix(), train=True, download=True, transform=transform),
        batch_size=batch_size, num_workers=4, shuffle=True
    )
    testloader = DataLoader(
        MNIST(DATA_ROOT.as_posix(), train=False, download=True, transform=transform),
        batch_size=batch_size, num_workers=4, shuffle=False
    )
    return trainloader, testloader


def CIFAR10Loader(batch_size, T):
    """
    Returns DataLoaders for CIFAR-10, with rate-coded spikes over T timesteps.
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)
    transform = Compose([
        ToTensor(),
        Normalize(cifar_mean, cifar_std),
        Rate(T),
        Lambda(lambda x: torch.flatten(x, start_dim=1))
    ])
    trainloader = DataLoader(
        CIFAR10(DATA_ROOT.as_posix(), train=True, download=True, transform=transform),
        batch_size=batch_size, num_workers=4, shuffle=True
    )
    testloader = DataLoader(
        CIFAR10(DATA_ROOT.as_posix(), train=False, download=True, transform=transform),
        batch_size=batch_size, num_workers=4, shuffle=False
    )
    return trainloader, testloader


def FashionMNISTLoader(batch_size, T):
    """
    Returns DataLoaders for FashionMNIST, with rate-coded spikes over T timesteps.
    """
    transform = Compose([
        ToTensor(),
        Normalize((0.2860,), (0.3530,)),
        Rate(T),
        Lambda(lambda x: torch.flatten(x, start_dim=1))
    ])
    trainloader = DataLoader(
        FashionMNIST(DATA_ROOT.as_posix(), train=True, download=True, transform=transform),
        batch_size=batch_size, num_workers=4, shuffle=True
    )
    testloader = DataLoader(
        FashionMNIST(DATA_ROOT.as_posix(), train=False, download=True, transform=transform),
        batch_size=batch_size, num_workers=4, shuffle=False
    )
    return trainloader, testloader

def SVHNLoader(batch_size, T):
    """
    Returns DataLoaders for SVHN (10 classes, 32×32 RGB).
    A digit-classification task harder than MNIST but simpler than CIFAR-10.
    """
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        Rate(T),
        Lambda(lambda x: x.flatten(start_dim=1))
    ])
    trainloader = DataLoader(
        SVHN("../Data/SVHN", split="train", download=True, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    testloader = DataLoader(
        SVHN("../Data/SVHN", split="test", download=True, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=4
    )
    return trainloader, testloader

def DVSGestureLoader(batch_size, T):
    """
    Returns DataLoaders for IBM’s DVS Gesture dataset.
    Bins events into T frames of size 128×128 and flattens them.
    """
    def tonic_to_tensor(sample):
        # sample['events']: array of shape [N,4] with columns [x,y,p,t]
        # sample['timestamps']: array of shape [N,] with event times
        events = sample["events"]
        ts     = sample["timestamps"]
        frames = np.zeros((T, 128, 128), dtype=np.float32)
        # split the full time range into T bins
        edges = np.linspace(ts.min(), ts.max(), T + 1)
        for i in range(T):
            m = (ts >= edges[i]) & (ts < edges[i+1])
            xs = events[m, 0].astype(int)
            ys = events[m, 1].astype(int)
            frames[i, ys, xs] = 1.0
        # flatten to a single vector of length T*128*128
        return torch.from_numpy(frames).flatten(), sample["label"]

    # point to wherever you want the raw .tar.gz to live
    train_ds = DVSGesture(save_to="../Data/DVSGesture", train=True)
    test_ds  = DVSGesture(save_to="../Data/DVSGesture", train=False)

    train_ds.transform       = tonic_to_tensor
    test_ds.transform        = tonic_to_tensor

    trainloader = DataLoader(train_ds,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)
    testloader  = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)
    return trainloader, testloader


'''
def NMNISTLoader(batch_size, T):
    """
    Returns DataLoaders for N-MNIST (neuromorphic MNIST): event-based 34×34 pixels.
    Uses the Tonic library to download and bin events into T frames.
    """
    def tonic_to_tensor(sample):
        # sample['events']: ndarray[N×4] columns [x, y, p, t]
        events = sample["events"]
        ts     = sample["timestamps"]
        # make T time-bins over the full span
        frames = np.zeros((T, 34, 34), dtype=np.float32)
        edges  = np.linspace(ts.min(), ts.max(), T + 1)
        for i in range(T):
            m = (ts >= edges[i]) & (ts < edges[i+1])
            xs = events[m, 0].astype(int)
            ys = events[m, 1].astype(int)
            frames[i, ys, xs] = 1.0
        # flatten to (T*34*34,)
        return torch.from_numpy(frames).flatten(), sample["label"]

    train_ds = tonic.datasets.NMNIST(save_to="../Data/NMNIST", train=True)
    test_ds  = tonic.datasets.NMNIST(save_to="../Data/NMNIST", train=False)
    train_ds.transform = tonic_to_tensor
    test_ds.transform  = tonic_to_tensor

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader
'''


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
