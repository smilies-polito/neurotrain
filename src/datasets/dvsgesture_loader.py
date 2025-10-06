from pathlib import Path
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from datasets.rate import Rate

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

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