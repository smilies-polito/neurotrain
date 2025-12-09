from pathlib import Path
from torchvision.datasets import SVHN
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from datasets.rate import Rate

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

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