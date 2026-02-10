from pathlib import Path
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from datasets.rate import Rate

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

def FashionMNISTLoader(batch_size, T, pin_memory: bool = False, seed=None):
    """
    Returns DataLoaders for FashionMNIST, with rate-coded spikes over T timesteps.
    """
    transform = Compose([
        ToTensor(),
        Normalize((0.2860,), (0.3530,)),
        Rate(T),
        Lambda(lambda x: torch.flatten(x, start_dim=1))
    ])
    train_kw = dict(batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=pin_memory)
    if seed is not None:
        train_kw["generator"] = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(
        FashionMNIST(DATA_ROOT.as_posix(), train=True, download=True, transform=transform),
        **train_kw,
    )
    testloader = DataLoader(
        FashionMNIST(DATA_ROOT.as_posix(), train=False, download=True, transform=transform),
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return trainloader, testloader