"""
Helper functions for the main training script.
Contains utility functions for device selection, configuration validation, and logging.
"""

import torch
from pathlib import Path
from typing import List
import numpy as np

def get_device(quantization_mode: bool, assigned_gpus: List[int]) -> str:
    """
    Determine the appropriate device for training with proper validation.
    Args:
        quantization_mode (bool): If True, force CPU usage for quantization
        assigned_gpus (list): List of assigned GPU IDs from Ray
    Returns:
        str: Device string (e.g., "cpu", "cuda:0")
    """
    if quantization_mode:
        return "cpu"
    
    if assigned_gpus and torch.cuda.is_available():
        gpu_id = int(assigned_gpus[0])
        if torch.cuda.device_count() > gpu_id:
            try:
                # Test if the GPU is actually accessible
                torch.cuda.set_device(gpu_id)
                device = f"cuda:{gpu_id}"
                return device
            except (RuntimeError, AssertionError) as e:
                print(f"Warning: GPU {gpu_id} not accessible ({e}), falling back to CPU")
    
    print("Using CPU device")
    return "cpu"


def setup_storage_path(exp_name: str) -> str:
    """
    Set up and validate the storage path for experiments.
    Args:
        exp_name: Name of the experiment
    Returns:
        str: URI path for storage
    Raises:
        RuntimeError: If storage directory cannot be created
    """
    _storage_path = Path("../Log").resolve().as_posix()
    storage_path_uri = "file:" + _storage_path
    path = storage_path_uri + "/" + exp_name
    
    # Ensure the storage directory exists
    try:
        Path(_storage_path).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Could not create storage directory {_storage_path}: {e}")
    
    return path

# Fixes seeds for both NumPy and PyTorch random number generators. Ensures reproducibility.
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)