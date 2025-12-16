"""
BPTT trainer wrapping snnTorch's built-in backprop functions.

This is a thin wrapper around snnTorch's BPTT functionality to match
the BaseTrainer interface for fair algorithm comparison.
"""

import torch
import torch.nn as nn
import snntorch.functional as SF

from trainers.base_trainer import BaseTrainer


class BPTTTrainer(BaseTrainer):
    """
    BPTT (Backpropagation Through Time) trainer using snnTorch's functional API.
    
    This trainer uses standard gradient-based optimization with surrogate gradients
    for the non-differentiable spike function. It serves as a baseline for comparing
    against local learning rules like STSF.
    
    Attributes:
        network: FCNetwork instance to train
        lr: Learning rate
        batch_size: Batch size for training
        loss_type: Type of loss function ("mse_count", "ce_count", "ce_rate")
        optimizer: PyTorch optimizer instance
        loss_fn: snnTorch loss function
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        loss_type: str = "ce_rate",
        quant: bool = False,
        use_optimizer: bool = True,
        optimizer=None,
        **kwargs,
    ):
        """
        Initialize BPTT trainer.
        
        Args:
            network: Neural network to train
            lr: Learning rate
            batch_size: Training batch size
            loss_type: Loss function type ("mse_count", "ce_count", "ce_rate")
            quant: Quantization flag (unused for BPTT, kept for interface compatibility)
            use_optimizer: Whether to use optimizer (always True for BPTT)
            optimizer: Pre-configured optimizer (if None, creates Adam)
            **kwargs: Additional arguments (ignored)
        """
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        
        # Store provided optimizer or mark for lazy creation after .to(device)
        self._external_optimizer = optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            # Create optimizer - will be recreated in to() if device changes
            self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        
        # Use snnTorch's built-in loss functions
        loss_functions = {
            "mse_count": SF.mse_count_loss(),
            "ce_count": SF.ce_count_loss(),
            "ce_rate": SF.ce_rate_loss(),
        }
        self.loss_fn = loss_functions.get(loss_type, SF.ce_rate_loss())
        self.loss_type = loss_type

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on a single batch using BPTT.
        
        Uses gradient-based backpropagation through all timesteps with
        snnTorch's surrogate gradient functions.
        
        Args:
            data: Input tensor of shape [num_timesteps, batch_size, in_features]
            target: Target labels of shape [batch_size]
            
        Returns:
            loss: Scalar loss tensor
            pred: Predictions of shape [batch_size, 1]
        """
        num_timesteps = data.shape[0]
        
        # Enable gradients for BPTT
        with torch.enable_grad():
            # Reset network state
            self.network.reset()
            
            # Forward pass through all timesteps, accumulating spikes
            spk_rec = []
            mem_rec = []
            
            for t in range(num_timesteps):
                spks, mems = self.network(data[t])
                spk_rec.append(spks[-1])  # Output layer spikes
                mem_rec.append(mems[-1])  # Output layer membrane
            
            # Stack into tensors [num_steps, batch, classes]
            spk_out = torch.stack(spk_rec, dim=0)
            mem_out = torch.stack(mem_rec, dim=0)
            
            # Compute loss using snnTorch's functional loss
            # These functions expect [num_steps, batch, num_classes] format
            loss = self.loss_fn(spk_out, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Compute predictions from spike sum (no gradients needed)
        with torch.no_grad():
            spk_sum = spk_out.sum(dim=0)  # Sum over time
            pred = spk_sum.argmax(dim=1, keepdim=True)
        
        return loss.detach(), pred

    def reset(self):
        """Reset all LIF neuron states in the network."""
        self.network.reset()

    def to(self, device):
        """
        Move trainer and network to device, recreating optimizer with new parameters.
        
        This is necessary because optimizer holds references to parameter tensors.
        When network.to(device) is called, new tensors are created on the target device,
        but the optimizer still references the old CPU tensors.
        """
        # Move network to device (this creates new parameter tensors on target device)
        super().to(device)
        
        # Recreate optimizer with the new device parameters
        # (only if we created it ourselves, not if it was externally provided)
        if self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        
        return self
