from pathlib import Path
import torch
from torchvision.datasets import SVHN
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader
from datasets.rate import Rate

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

def SVHNLoader(
    batch_size,
    T,
    flatten: bool = True,
    pin_memory: bool = False,
    seed=None,
):
    """
    Returns DataLoaders for SVHN (10 classes, 32×32 RGB).
    A digit-classification task harder than MNIST but simpler than CIFAR-10.
    """
    transforms = [
        ToTensor(),
        # Rate coding expects probabilities in [0, 1]. Keep raw pixel scale here.
        Rate(T, flatten=flatten),
    ]
    if flatten:
        transforms.append(Lambda(lambda x: x.flatten(start_dim=1)))
    transform = Compose(transforms)
    train_kw = dict(batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    if seed is not None:
        train_kw["generator"] = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(
        SVHN(DATA_ROOT.as_posix(), split="train", download=True, transform=transform),
        **train_kw,
    )
    testloader = DataLoader(
        SVHN(DATA_ROOT.as_posix(), split="test", download=True, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )
    return trainloader, testloader
