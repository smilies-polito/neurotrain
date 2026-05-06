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


def spike_to_prediction(
    preds: torch.Tensor,
    expected_timesteps: Optional[int] = None,
) -> torch.Tensor:
    """
    Postprocessor to convert spike outputs to class predictions.
    
    NeuroBench's ClassificationAccuracy expects predictions of shape [batch]
    matching label shape. Depending on NeuroBench/custom wrapper internals, spike
    tensors can arrive as either [batch, timesteps, classes] or
    [timesteps, batch, classes]. This function handles both layouts.
    
    Note: Returns predictions on CPU to match labels from DataLoader.
    
    Args:
        preds: Spike output tensor
        expected_timesteps: Optional expected timestep count used to disambiguate
            layout when possible
        
    Returns:
        Class predictions of shape [batch] on CPU
    """
    if preds.dim() == 3:
        if expected_timesteps is not None:
            if preds.shape[1] == expected_timesteps:
                # [batch, timesteps, classes]
                spike_sum = preds.sum(dim=1)
            elif preds.shape[0] == expected_timesteps:
                # [timesteps, batch, classes]
                spike_sum = preds.sum(dim=0)
            else:
                # Fall back to batch-major convention used by this wrapper.
                spike_sum = preds.sum(dim=1)
        else:
            # Fall back to batch-major convention used by this wrapper.
            spike_sum = preds.sum(dim=1)
    elif preds.dim() == 2:
        # Already aggregated to [batch, classes].
        spike_sum = preds
    else:
        raise ValueError(f"Unsupported prediction shape for NeuroBench: {tuple(preds.shape)}")
    # Get predicted class: [batch, classes] -> [batch]
    # Move to CPU to match labels from DataLoader
    return spike_sum.argmax(dim=1).cpu()


def run_neurobench(
    network: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    num_timesteps: Optional[int] = None,
    include_synaptic_operations: bool = False,
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
        include_synaptic_operations: Whether to include SynapticOperations.
            Disabled by default because some model stacks trigger repeated
            deepcopy errors in current NeuroBench/PyTorch combinations.
        
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
    # Pass device so wrapper can move inputs to correct device
    wrapped_model = NeuroBenchWrapper(network, num_timesteps, device)
    wrapped_model.to(device)
    
    # Wrap with NeuroBench's SNNTorchModel
    # Use custom_forward=True since our wrapper handles the full temporal loop
    # and returns spikes in the correct format [timesteps, batch, classes]
    nb_model = SNNTorchModel(wrapped_model, custom_forward=True)
    
    # Configure static metrics (computed once on model)
    # NeuroBench v2.x expects metric CLASSES, not instances
    static_metrics = [
        Footprint,             # Memory footprint of the model
        ConnectionSparsity,    # Sparsity of weight connections
        ParameterCount,        # Total number of parameters
    ]
    
    # Configure workload metrics (computed during inference)
    workload_metrics = [
        ActivationSparsity,         # Overall spike sparsity
        ActivationSparsityByLayer,  # Per-layer spike sparsity breakdown
        MembraneUpdates,            # Number of membrane potential updates
        ClassificationAccuracy,     # Classification accuracy
    ]
    if include_synaptic_operations:
        workload_metrics.insert(2, SynapticOperations)  # Effective/Dense MAC statistics
    
    # Create and run benchmark
    # NeuroBench v2.x uses metric_list=[static_metrics, workload_metrics]
    # Postprocessor converts spike outputs [batch, T, classes] to predictions [batch]
    benchmark = Benchmark(
        model=nb_model,
        dataloader=test_loader,
        preprocessors=[],
        postprocessors=[
            lambda preds: spike_to_prediction(preds, expected_timesteps=num_timesteps)
        ],
        metric_list=[static_metrics, workload_metrics],
    )

    try:
        results = benchmark.run()
    finally:
        # Explicitly delete NeuroBench wrappers so they don't keep 'network' alive
        # through the exception traceback when an OOM is raised inside benchmark.run().
        del benchmark, nb_model, wrapped_model

    return results


class NeuroBenchWrapper(torch.nn.Module):
    """
    Wrapper to make FCNetwork compatible with NeuroBench's SNNTorchModel.
    
    NeuroBench expects the model's forward() to return spike tensors.
    This wrapper handles the timestep iteration and returns accumulated spikes.
    """
    
    def __init__(self, network: torch.nn.Module, num_timesteps: Optional[int] = None, device: str = "cpu"):
        """
        Initialize wrapper.
        
        Args:
            network: FCNetwork instance
            num_timesteps: Number of timesteps (inferred from input if None)
            device: Device to run inference on (for moving inputs)
        """
        super().__init__()
        self.network = network
        self.num_timesteps = num_timesteps
        self.device = device

    def _to_time_major(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Normalize batch/time layout to [timesteps, batch, ...]."""
        if x.dim() < 2:
            return x.unsqueeze(0), 1

        if self.num_timesteps is not None:
            if x.shape[0] == self.num_timesteps:
                return x, int(x.shape[0])
            if x.shape[1] == self.num_timesteps:
                return x.transpose(0, 1), int(x.shape[1])

        # Fallback heuristic: if first dim is likely batch and second dim likely
        # timesteps, transpose; otherwise keep as-is.
        if x.shape[0] > x.shape[1]:
            return x.transpose(0, 1), int(x.shape[1])
        return x, int(x.shape[0])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that handles temporal dimension.
        
        Args:
            x: Input tensor. Expected shape [batch, timesteps, features] or
               [timesteps, batch, features]
               
        Returns:
            Spike output tensor of shape [batch, num_classes]
        """
        # Move input to the same device as the network
        x = x.to(self.device)
        
        # Handle both vector/image layouts and both [B, T, ...] / [T, B, ...].
        x, num_timesteps = self._to_time_major(x)
        
        # Reset network state
        self.network.reset()
        
        # Forward through all timesteps, collecting spikes for NeuroBench
        all_spikes = []
        
        for t in range(num_timesteps):
            out = self.network(x[t])

            # Handle different network return styles:
            # - (spk_rec, mem_rec)
            # - tensor readout
            # - [layer_outputs, ...]
            if isinstance(out, (tuple, list)):
                first = out[0]
                if isinstance(first, (tuple, list)) and len(first) > 0:
                    readout = first[-1]
                else:
                    readout = first
            else:
                readout = out

            if isinstance(readout, torch.Tensor) and readout.dim() == 1:
                readout = readout.unsqueeze(0)

            all_spikes.append(readout)
        
        # Stack spikes as [batch, timesteps, classes].
        spk_out = torch.stack(all_spikes, dim=1)
        
        return spk_out
    
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
        Footprint,
        ConnectionSparsity,
        ParameterCount,
    ]
    
    # Static metrics can be computed without a dataloader
    results = {}
    for metric_cls in static_metrics:
        metric_name = metric_cls.__name__
        try:
            # Instantiate and call the metric
            results[metric_name] = metric_cls()(nb_model)
        except Exception as e:
            results[metric_name] = f"Error: {e}"
    
    return results
