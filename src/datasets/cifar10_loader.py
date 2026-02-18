from pathlib import Path

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Lambda, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
from torch.utils.data import DataLoader

from datasets.rate import Rate

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"


class _RepeatStatic:
    """Repeat a static frame over T timesteps."""

    def __init__(self, timesteps: int):
        self.timesteps = int(timesteps)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        reps = [self.timesteps] + [1] * x.dim()
        return x.unsqueeze(0).repeat(*reps)


class _Cutout:
    """Simple square cutout augmentation used by the official OTTT CIFAR recipe."""

    def __init__(self, size: int = 16, p: float = 0.5):
        self.size = int(size)
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return x

        _, h, w = x.shape
        y = int(torch.randint(0, h, ()).item())
        x0 = int(torch.randint(0, w, ()).item())
        half = self.size // 2
        y1, y2 = max(0, y - half), min(h, y + half)
        x1, x2 = max(0, x0 - half), min(w, x0 + half)

        x = x.clone()
        x[:, y1:y2, x1:x2] = 0.0
        return x


def CIFAR10Loader(
    batch_size,
    T,
    flatten: bool = True,
    pin_memory: bool = False,
    seed=None,
    static_input: bool = False,
    use_augmentation: bool = False,
    use_normalization: bool = False,
):
    """
    Returns DataLoaders for CIFAR-10.

    - static_input=False: Bernoulli rate-coded spikes (legacy benchmark behavior).
    - static_input=True: repeat one transformed image across T timesteps.
    """

    if static_input:
        normalize = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transforms = []
        if use_augmentation:
            train_transforms.extend([RandomCrop(32, padding=4), RandomHorizontalFlip()])
        train_transforms.extend([ToTensor()])
        if use_normalization:
            train_transforms.append(normalize)
        if use_augmentation:
            train_transforms.append(_Cutout(size=16, p=0.5))
        train_transforms.append(_RepeatStatic(T))
        if flatten:
            train_transforms.append(Lambda(lambda x: torch.flatten(x, start_dim=1)))

        test_transforms = [ToTensor()]
        if use_normalization:
            test_transforms.append(normalize)
        test_transforms.append(_RepeatStatic(T))
        if flatten:
            test_transforms.append(Lambda(lambda x: torch.flatten(x, start_dim=1)))

        train_transform = Compose(train_transforms)
        test_transform = Compose(test_transforms)
    else:
        transforms = [
            ToTensor(),
            # Rate coding expects probabilities in [0, 1]. Keep raw pixel scale here.
            Rate(T, flatten=flatten),
        ]
        if flatten:
            transforms.append(Lambda(lambda x: torch.flatten(x, start_dim=1)))
        train_transform = Compose(transforms)
        test_transform = train_transform

    train_kw = dict(batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=pin_memory)
    if seed is not None:
        train_kw["generator"] = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(
        CIFAR10(DATA_ROOT.as_posix(), train=True, download=True, transform=train_transform),
        **train_kw,
    )
    testloader = DataLoader(
        CIFAR10(DATA_ROOT.as_posix(), train=False, download=True, transform=test_transform),
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return trainloader, testloader
