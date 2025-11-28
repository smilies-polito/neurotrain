from pathlib import Path
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from datasets.rate import Rate

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

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