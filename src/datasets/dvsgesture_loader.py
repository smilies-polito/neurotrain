"""
DVS Gesture dataset loader using Tonic library.

DVS Gesture is an event-based dataset recorded with a Dynamic Vision Sensor (DVS).
Contains 11 hand gesture classes recorded by 29 subjects under 3 lighting conditions.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tonic.datasets import DVSGesture
    from tonic import transforms as tonic_transforms

    TONIC_AVAILABLE = True
except ImportError:
    TONIC_AVAILABLE = False

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"


def DVSGestureLoader(batch_size, T, flatten: bool = True, pin_memory: bool = False, seed=None):
    """
    Returns DataLoaders for IBM's DVS Gesture dataset.

    Bins events into T frames of size 128×128 and flattens to [T, 128*128].
    This is EVENT-BASED data, not rate-coded - ideal for DECOLLE.

    Args:
        batch_size: Batch size for DataLoader
        T: Number of time bins to divide events into

    Returns:
        trainloader, testloader: DataLoader instances

    Note:
        - Input size: 128*128 = 16384 (flattened)
        - Output classes: 11 gestures
        - Requires tonic library: pip install tonic
    """
    if not TONIC_AVAILABLE:
        raise ImportError(
            "DVSGesture requires the 'tonic' library. Install with: pip install tonic"
        )

    sensor_size = (128, 128, 2)  # DVS128 sensor: 128x128, 2 polarities

    # Transform pipeline: bin events into frames
    transform = tonic_transforms.Compose(
        [
            tonic_transforms.Denoise(filter_time=10000),  # Remove noise
            tonic_transforms.ToFrame(
                sensor_size=sensor_size,
                n_time_bins=T,
            ),
        ]
    )

    def collate_fn(batch):
        """Custom collate to handle variable-length event sequences."""
        frames_list = []
        labels_list = []

        for events, label in batch:
            # events shape: [T, 2, 128, 128] (2 polarities)
            # Merge polarities and flatten spatial dims
            if events.ndim == 4:
                # Sum polarities: [T, 2, 128, 128] -> [T, 128, 128]
                frame = events.sum(axis=1)
            else:
                frame = events

            if flatten:
                # Flatten spatial: [T, 128, 128] -> [T, 16384]
                frame = frame.reshape(T, -1)
            else:
                # Convolutional models expect channel-first frames.
                frame = frame.reshape(T, 1, 128, 128)

            # Normalize to [0, 1] (events are counts)
            frame = np.clip(frame, 0, 1).astype(np.float32)

            frames_list.append(torch.from_numpy(frame))
            labels_list.append(label)

        # Keep DataLoader output batch-first: [batch, T, features].
        # The training loop handles conversion to time-major format.
        frames = torch.stack(frames_list)
        labels = torch.tensor(labels_list, dtype=torch.long)

        return frames, labels

    # Load datasets
    train_ds = DVSGesture(
        save_to=str(DATA_ROOT / "DVSGesture"),
        train=True,
        transform=transform,
    )
    test_ds = DVSGesture(
        save_to=str(DATA_ROOT / "DVSGesture"),
        train=False,
        transform=transform,
    )

    train_kw = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    if seed is not None:
        train_kw["generator"] = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(train_ds, **train_kw)
    testloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


# Dataset info for benchmarking
DVSGESTURE_INFO = {
    "input_size": 128 * 128,  # 16384
    "num_classes": 11,
    "default_timesteps": 50,  # Events spread over 50 bins
    "type": "event-based",
}
