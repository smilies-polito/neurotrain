from pathlib import Path
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from datasets.rate import Rate

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

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