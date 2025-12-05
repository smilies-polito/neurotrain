"""
NeuroBench integration for SNN evaluation.

Uses NeuroBench's built-in model wrapper, metrics, and benchmark harness
to compute standardized SNN metrics. No custom metric implementations.
"""

from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader

# NeuroBench imports
from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark
from neurobench.metrics.static import Footprint, ConnectionSparsity, ParameterCount
from neurobench.metrics.workload import (
    ActivationSparsity,
    ActivationSparsityByLayer,
    SynapticOperations,
    MembraneUpdates,
    ClassificationAccuracy,
)


def run_neurobench(
    network: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    num_timesteps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run NeuroBench evaluation using their built-in harness.
    
    This function wraps the network with NeuroBench's SNNTorchModel,
    configures static and workload metrics, and runs the benchmark.
    
    Args:
        network: Trained snnTorch network (FCNetwork)
        test_loader: DataLoader for test dataset
        device: Device to run evaluation on ("cpu", "cuda")
        num_timesteps: Number of timesteps for inference (if not inferrable)
        
    Returns:
        Dictionary containing all NeuroBench metrics:
        - Static metrics: footprint, connection_sparsity, parameter_count
        - Workload metrics: activation_sparsity, activation_sparsity_by_layer,
          synaptic_operations, membrane_updates, classification_accuracy
    """
    # Move network to device and set to eval mode
    network.to(device)
    network.eval()
    
    # Create a wrapper that handles the forward pass properly for NeuroBench
    # NeuroBench expects the model to return spikes in a specific format
    wrapped_model = NeuroBenchWrapper(network, num_timesteps)
    
    # Wrap with NeuroBench's SNNTorchModel
    nb_model = SNNTorchModel(wrapped_model)
    
    # Configure static metrics (computed once on model)
    static_metrics = [
        Footprint(),           # Memory footprint of the model
        ConnectionSparsity(),  # Sparsity of weight connections
        ParameterCount(),      # Total number of parameters
    ]
    
    # Configure workload metrics (computed during inference)
    workload_metrics = [
        ActivationSparsity(),       # Overall spike sparsity
        ActivationSparsityByLayer(),# Per-layer spike sparsity breakdown
        SynapticOperations(),       # Number of synaptic operations (MACs)
        MembraneUpdates(),          # Number of membrane potential updates
        ClassificationAccuracy(),   # Classification accuracy
    ]
    
    # Create and run benchmark
    benchmark = Benchmark(
        model=nb_model,
        dataloader=test_loader,
        static_metrics=static_metrics,
        workload_metrics=workload_metrics,
    )
    
    results = benchmark.run()
    
    return results


class NeuroBenchWrapper(torch.nn.Module):
    """
    Wrapper to make FCNetwork compatible with NeuroBench's SNNTorchModel.
    
    NeuroBench expects the model's forward() to return spike tensors.
    This wrapper handles the timestep iteration and returns accumulated spikes.
    """
    
    def __init__(self, network: torch.nn.Module, num_timesteps: Optional[int] = None):
        """
        Initialize wrapper.
        
        Args:
            network: FCNetwork instance
            num_timesteps: Number of timesteps (inferred from input if None)
        """
        super().__init__()
        self.network = network
        self.num_timesteps = num_timesteps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that handles temporal dimension.
        
        Args:
            x: Input tensor. Expected shape [batch, timesteps, features] or
               [timesteps, batch, features]
               
        Returns:
            Spike output tensor of shape [batch, num_classes]
        """
        # Handle different input formats
        if x.dim() == 3:
            # Check if shape is [batch, timesteps, features] or [timesteps, batch, features]
            # Our convention is [timesteps, batch, features]
            if x.shape[0] > x.shape[1]:
                # Likely [timesteps, batch, features]
                num_timesteps = x.shape[0]
            else:
                # Likely [batch, timesteps, features], transpose
                x = x.transpose(0, 1)
                num_timesteps = x.shape[0]
        else:
            num_timesteps = self.num_timesteps or 1
        
        # Reset network state
        self.network.reset()
        
        # Forward through all timesteps
        spk_sum = None
        all_spikes = []
        
        for t in range(num_timesteps):
            spks, _ = self.network(x[t])
            all_spikes.append(spks[-1])  # Output layer spikes
            if spk_sum is None:
                spk_sum = spks[-1]
            else:
                spk_sum = spk_sum + spks[-1]
        
        # Return spike sum for classification
        return spk_sum
    
    def reset(self):
        """Reset network state."""
        self.network.reset()


def compute_static_metrics(network: torch.nn.Module) -> Dict[str, Any]:
    """
    Compute static metrics on the model without running inference.
    
    Args:
        network: Neural network to analyze
        
    Returns:
        Dictionary with static metrics
    """
    nb_model = SNNTorchModel(network)
    
    static_metrics = [
        Footprint(),
        ConnectionSparsity(),
        ParameterCount(),
    ]
    
    # Static metrics can be computed without a dataloader
    results = {}
    for metric in static_metrics:
        metric_name = metric.__class__.__name__
        try:
            results[metric_name] = metric(nb_model)
        except Exception as e:
            results[metric_name] = f"Error: {e}"
    
    return results

