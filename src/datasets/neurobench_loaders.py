"""
NeuroBench official dataset loaders.

These are the standardized benchmarks from NeuroBench for evaluating
neuromorphic algorithms. See: https://neurobench.ai

Classification Tasks:
- SpeechCommands: Google Speech Commands (keyword spotting)
- WISDM: Human Activity Recognition

Regression Tasks:
- PrimateReaching: Non-human primate motor prediction
- MackeyGlass: Chaotic time series prediction

Few-shot Learning:
- MSWC: Multilingual Spoken Word Commands (FSCIL)
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# NeuroBench dataset imports
from neurobench.datasets import SpeechCommands, PrimateReaching, MackeyGlass, WISDM

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data" / "NeuroBench"


class SpeechCommandsWrapper(Dataset):
    """Wrapper to make SpeechCommands compatible with our training loop."""
    
    def __init__(self, dataset, T: int):
        self.dataset = dataset
        self.T = T
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        
        # data shape: [channels, time] or [time, features]
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        # Ensure shape is [T, features]
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        # Resample/pad to match desired T
        if data.shape[0] != self.T:
            data = self._resample(data, self.T)
        
        # Flatten features if needed
        data = data.reshape(self.T, -1)
        
        return data, label
    
    def _resample(self, data, target_len):
        """Resample temporal data to target length."""
        current_len = data.shape[0]
        if current_len == target_len:
            return data
        
        # Use linear interpolation for resampling
        indices = torch.linspace(0, current_len - 1, target_len).long()
        return data[indices]


class WISDMWrapper(Dataset):
    """Wrapper to make WISDM compatible with our training loop."""
    
    def __init__(self, dataset, T: int):
        self.dataset = dataset
        self.T = T
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        # Ensure shape is [T, features]
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        # Resample to match T
        if data.shape[0] != self.T:
            data = self._resample(data, self.T)
        
        data = data.reshape(self.T, -1)
        
        return data, label
    
    def _resample(self, data, target_len):
        current_len = data.shape[0]
        if current_len == target_len:
            return data
        indices = torch.linspace(0, current_len - 1, target_len).long()
        return data[indices]


class PrimateReachingWrapper(Dataset):
    """Wrapper for PrimateReaching (regression task)."""
    
    def __init__(self, dataset, T: int):
        self.dataset = dataset
        self.T = T
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()
        
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        if data.shape[0] != self.T:
            data = self._resample(data, self.T)
        
        data = data.reshape(self.T, -1)
        
        return data, target
    
    def _resample(self, data, target_len):
        current_len = data.shape[0]
        if current_len == target_len:
            return data
        indices = torch.linspace(0, current_len - 1, target_len).long()
        return data[indices]


class MackeyGlassWrapper(Dataset):
    """Wrapper for MackeyGlass chaotic time series (regression task)."""
    
    def __init__(self, dataset, T: int):
        self.dataset = dataset
        self.T = T
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()
        
        if data.dim() == 1:
            data = data.unsqueeze(-1)  # Add feature dimension
        
        if data.shape[0] != self.T:
            data = self._resample(data, self.T)
        
        # Ensure consistent output shape [T, features]
        data = data.reshape(self.T, -1)
        
        return data, target
    
    def _resample(self, data, target_len):
        current_len = data.shape[0]
        if current_len == target_len:
            return data
        indices = torch.linspace(0, current_len - 1, target_len).long()
        return data[indices]


def SpeechCommandsLoader(batch_size: int, T: int, pin_memory: bool = False, seed=None):
    """
    Returns DataLoaders for Google Speech Commands (keyword spotting).
    
    Task: Classify spoken keywords (12 classes)
    Input: Audio waveforms
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    # NeuroBench v2.x uses 'path' parameter, not 'root'
    train_dataset = SpeechCommands(path=DATA_ROOT.as_posix(), subset="training")
    test_dataset = SpeechCommands(path=DATA_ROOT.as_posix(), subset="testing")
    
    train_wrapped = SpeechCommandsWrapper(train_dataset, T)
    test_wrapped = SpeechCommandsWrapper(test_dataset, T)
    
    trainloader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        test_wrapped,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


def WISDMLoader(batch_size: int, T: int, pin_memory: bool = False, seed=None):
    """
    Returns DataLoaders for WISDM Human Activity Recognition.
    
    Task: Classify human activities from accelerometer data (6 classes)
    Input: 3-axis accelerometer time series
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    train_dataset = WISDM(root=DATA_ROOT.as_posix(), split="train", download=True)
    test_dataset = WISDM(root=DATA_ROOT.as_posix(), split="test", download=True)
    
    train_wrapped = WISDMWrapper(train_dataset, T)
    test_wrapped = WISDMWrapper(test_dataset, T)
    
    trainloader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        test_wrapped,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


def PrimateReachingLoader(batch_size: int, T: int, pin_memory: bool = False, seed=None):
    """
    Returns DataLoaders for Primate Reaching motor prediction.
    
    Task: Predict hand velocity from neural recordings (regression)
    Input: Multi-channel neural spike data
    Output: 2D velocity vector
    
    Note: This is a REGRESSION task, not classification!
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    train_dataset = PrimateReaching(
        root=DATA_ROOT.as_posix(),
        split="train",
        download=True
    )
    test_dataset = PrimateReaching(
        root=DATA_ROOT.as_posix(),
        split="test",
        download=True
    )
    
    train_wrapped = PrimateReachingWrapper(train_dataset, T)
    test_wrapped = PrimateReachingWrapper(test_dataset, T)
    
    trainloader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        test_wrapped,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


def MackeyGlassLoader(batch_size: int, T: int, pin_memory: bool = False, seed=None):
    """
    Returns DataLoaders for Mackey-Glass chaotic time series prediction.
    
    Task: Predict next value in chaotic time series (regression)
    Input: Time series window
    Output: Next value prediction
    
    Note: This is a REGRESSION task, not classification!
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    train_dataset = MackeyGlass(
        root=DATA_ROOT.as_posix(),
        split="train",
        download=True
    )
    test_dataset = MackeyGlass(
        root=DATA_ROOT.as_posix(),
        split="test", 
        download=True
    )
    
    train_wrapped = MackeyGlassWrapper(train_dataset, T)
    test_wrapped = MackeyGlassWrapper(test_dataset, T)
    
    trainloader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        test_wrapped,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


# Dataset metadata for automatic configuration
NEUROBENCH_DATASETS = {
    "SpeechCommands": {
        "loader": SpeechCommandsLoader,
        "task": "classification",
        "input_size": 16000,  # 1 second at 16kHz, will be resampled
        "num_classes": 12,
        "description": "Google Speech Commands keyword spotting",
    },
    "WISDM": {
        "loader": WISDMLoader,
        "task": "classification",
        "input_size": 3,  # 3-axis accelerometer
        "num_classes": 6,
        "description": "Human Activity Recognition from accelerometer",
    },
    "PrimateReaching": {
        "loader": PrimateReachingLoader,
        "task": "regression",
        "input_size": 96,  # Neural channels (varies)
        "output_size": 2,  # 2D velocity
        "description": "Motor prediction from neural recordings",
    },
    "MackeyGlass": {
        "loader": MackeyGlassLoader,
        "task": "regression",
        "input_size": 1,
        "output_size": 1,
        "description": "Chaotic time series prediction",
    },
}

