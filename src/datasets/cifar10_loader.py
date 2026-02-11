from pathlib import Path
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader
from datasets.rate import Rate

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

def CIFAR10Loader(batch_size, T, flatten: bool = True, pin_memory: bool = False, seed=None):
    """
    Returns DataLoaders for CIFAR-10, with rate-coded spikes over T timesteps.
    """
    transforms = [
        ToTensor(),
        # Rate coding expects probabilities in [0, 1]. Keep raw pixel scale here.
        Rate(T, flatten=flatten),
    ]
    if flatten:
        transforms.append(Lambda(lambda x: torch.flatten(x, start_dim=1)))
    transform = Compose(transforms)
    train_kw = dict(batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=pin_memory)
    if seed is not None:
        train_kw["generator"] = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(
        CIFAR10(DATA_ROOT.as_posix(), train=True, download=True, transform=transform),
        **train_kw,
    )
    testloader = DataLoader(
        CIFAR10(DATA_ROOT.as_posix(), train=False, download=True, transform=transform),
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return trainloader, testloader
