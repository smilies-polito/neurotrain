from pathlib import Path
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from datasets.rate import Rate

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"


def MNISTLoaderRaw(batch_size, T, flatten: bool = True, pin_memory: bool = False, seed=None):
    """
    Returns DataLoaders for MNIST with raw [0, 1] pixels (no Normalize, no rate coding).

    Each sample is the same image repeated T times so batch shape is (B, T, 784).
    Matches the reference (Deep Spike Learning with Local Classifiers): ToTensor() only.
    Used for ELL/FELL/BELL to feed the same raw image every timestep.
    """
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: x.view(-1)),
        Lambda(lambda x: x.unsqueeze(0).expand(T, -1)),
    ])
    train_kw = dict(
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=pin_memory,
    )
    if seed is not None:
        train_kw["generator"] = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(
        MNIST(DATA_ROOT.as_posix(), train=True, download=True, transform=transform),
        **train_kw,
    )
    testloader = DataLoader(
        MNIST(DATA_ROOT.as_posix(), train=False, download=True, transform=transform),
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return trainloader, testloader


def MNISTLoader(batch_size, T, pin_memory: bool = False, seed=None):
    """
    Returns DataLoaders for MNIST, with rate-coded spikes over T timesteps.

    Args:
        batch_size: Batch size.
        T: Number of timesteps for rate coding.
        pin_memory: If True, use pinned memory for faster CPU->GPU transfer (CUDA).
        seed: Optional int. If set, train DataLoader uses this seed for shuffle (deterministic order).
    """
    transforms = [
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Rate(T, flatten=flatten),
    ]
    if flatten:
        transforms.append(Lambda(lambda x: torch.flatten(x, start_dim=1)))
    transform = Compose(transforms)
    train_kw = dict(
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=pin_memory,
    )
    if seed is not None:
        train_kw["generator"] = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(
        MNIST(DATA_ROOT.as_posix(), train=True, download=True, transform=transform),
        **train_kw,
    )
    testloader = DataLoader(
        MNIST(DATA_ROOT.as_posix(), train=False, download=True, transform=transform),
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return trainloader, testloader
