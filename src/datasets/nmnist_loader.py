"""
N-MNIST (Neuromorphic MNIST) dataset loader using Tonic library.

N-MNIST is created by recording MNIST digits with a DVS camera while
performing saccadic eye movements. This creates event-based spike trains
that preserve temporal information - ideal for SNNs and DECOLLE.

Reference: Orchard et al., "Converting Static Image Datasets to Spiking 
Neuromorphic Datasets Using Saccades", 2015.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tonic.datasets import NMNIST
    from tonic import transforms as tonic_transforms
    TONIC_AVAILABLE = True
except ImportError:
    TONIC_AVAILABLE = False

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"


def NMNISTLoader(batch_size, T, pin_memory: bool = False):
    """
    Returns DataLoaders for N-MNIST (Neuromorphic MNIST) dataset.
    
    Bins events into T frames of size 34×34 and flattens to [T, 34*34].
    This is EVENT-BASED data recorded with a DVS camera - ideal for DECOLLE.
    
    Args:
        batch_size: Batch size for DataLoader
        T: Number of time bins to divide events into
        
    Returns:
        trainloader, testloader: DataLoader instances
        
    Note:
        - Input size: 34*34 = 1156 (flattened)
        - Output classes: 10 digits
        - Requires tonic library: pip install tonic
    """
    if not TONIC_AVAILABLE:
        raise ImportError(
            "N-MNIST requires the 'tonic' library. "
            "Install with: pip install tonic"
        )
    
    sensor_size = (34, 34, 2)  # N-MNIST sensor: 34x34, 2 polarities
    
    # Transform pipeline: bin events into frames
    transform = tonic_transforms.Compose([
        tonic_transforms.Denoise(filter_time=10000),  # Remove noise
        tonic_transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=T,
        ),
    ])
    
    def collate_fn(batch):
        """Custom collate to handle variable-length event sequences."""
        frames_list = []
        labels_list = []
        
        for events, label in batch:
            # events shape: [T, 2, 34, 34] (2 polarities)
            # Merge polarities and flatten spatial dims
            if events.ndim == 4:
                # Sum polarities: [T, 2, 34, 34] -> [T, 34, 34]
                frame = events.sum(axis=1)
            else:
                frame = events
            
            # Flatten spatial: [T, 34, 34] -> [T, 1156]
            frame = frame.reshape(T, -1)
            
            # Normalize to [0, 1] (events are counts)
            frame = np.clip(frame, 0, 1).astype(np.float32)
            
            frames_list.append(torch.from_numpy(frame))
            labels_list.append(label)
        
        # Stack: [batch, T, features] then transpose to [T, batch, features]
        frames = torch.stack(frames_list).transpose(0, 1)
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        return frames, labels
    
    # Load datasets
    train_ds = NMNIST(
        save_to=str(DATA_ROOT / "NMNIST"),
        train=True,
        transform=transform,
    )
    test_ds = NMNIST(
        save_to=str(DATA_ROOT / "NMNIST"),
        train=False,
        transform=transform,
    )
    
    trainloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
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
NMNIST_INFO = {
    "input_size": 34 * 34,  # 1156
    "num_classes": 10,
    "default_timesteps": 25,  # Events spread over 25 bins
    "type": "event-based",
}

