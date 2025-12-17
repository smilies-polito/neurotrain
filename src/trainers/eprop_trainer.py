"""
E-prop (Eligibility Propagation) trainer for snnTorch-based networks.

Implements the e-prop learning algorithm for recurrent spiking neural networks
(RSNNs) using snnTorch neurons. Based on:

    [G. Bellec et al., "A solution to the learning dilemma for recurrent networks
     of spiking neurons," Nature communications, vol. 11, no. 3625, 2020]

This implementation follows the recurrent formulation and performs *online*
updates: weights are updated at each timestep rather than once per full sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from networks.recurrent_srnn import RecurrentSRNN


class EpropTrainer(BaseTrainer):
    """
    E-prop (Eligibility Propagation) trainer for recurrent SNNs (RSNNs).
    
    Implements a local learning rule that approximates BPTT by combining
    eligibility traces with a learning signal computed from output errors.
    
    Attributes:
        network: RecurrentSRNN instance to train
        lr: Learning rate
        batch_size: Batch size for training
        gamma: Surrogate gradient magnitude parameter
        kappa: Output layer membrane time constant decay factor
        alpha: Hidden layer membrane time constant decay factor
        lr_layer_norm: Per-layer learning rate modulation (input, hidden, output)
    """
    
    def __init__(
        self,
        network: RecurrentSRNN,
        lr: float,
        batch_size: int,
        gamma: float = 0.3,
        tau_mem: float = 0.9,  # Corresponds to beta in snnTorch
        tau_out: float = 0.9,  # Output layer time constant
        lr_layer_norm: tuple = (1.0, 1.0, 1.0),
        quant: bool = False,
        use_optimizer: bool = False,
        optimizer=None,
        update_last: bool = False,
        update_every: int = 1,
        seq_batch_size: int = 1,
        **kwargs,
    ):
        """
        Initialize E-prop trainer.
        
        Args:
            network: Neural network to train
            lr: Learning rate
            batch_size: Training batch size
            gamma: Surrogate derivative magnitude parameter (default: 0.3)
            tau_mem: Membrane time constant for hidden layers (default: 0.9)
            tau_out: Membrane time constant for output layer (default: 0.9)
            lr_layer_norm: Per-layer learning rate modulation tuple
            quant: Quantization flag (kept for interface compatibility)
            use_optimizer: Whether to use PyTorch optimizer
            optimizer: Pre-configured optimizer (if None and use_optimizer, creates Adam)
            update_last: Update only on last timestep
            update_every: Update every N timesteps
            seq_batch_size: Sequential batch size (kept for interface compatibility)
            **kwargs: Additional arguments (ignored)
        """
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.quant = quant
        self.use_optimizer = use_optimizer
        self.update_last = update_last
        self.update_every = update_every
        self.seq_batch_size = seq_batch_size
        
        # Time constants (decay factors)
        if not (hasattr(network, "is_recurrent") and network.is_recurrent):
            raise TypeError(
                "EpropTrainer now supports recurrent RSNNs only; "
                f"got network={type(network).__name__}."
            )
        self.alpha = float(network.alpha)
        self.kappa = float(network.kappa)
        self.threshold = float(network.threshold)
        
        # Per-layer learning rate modulation
        self.lr_layer = lr_layer_norm
        
        # Loss function for monitoring
        self.loss_fn = nn.MSELoss()
        
        # Setup optimizer if requested
        self._external_optimizer = optimizer
        if use_optimizer:
            if optimizer is not None:
                self.optimizer = optimizer
            else:
                self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        else:
            self.optimizer = None

    def _surrogate_gradient(self, mem: torch.Tensor) -> torch.Tensor:
        """
        Compute surrogate gradient for the spike function.
        
        Uses the pseudo-derivative from Bellec et al.:
        h = γ * max(0, 1 - |v - θ| / θ)
        
        This provides a smooth gradient approximation around the threshold.
        
        Args:
            mem: Membrane potential tensor
            
        Returns:
            Surrogate gradient tensor
        """
        # Triangular surrogate gradient centered at threshold
        return self.gamma * torch.clamp(
            1.0 - torch.abs((mem - self.threshold) / self.threshold),
            min=0.0
        )
    
    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on a single batch using recurrent e-prop with online updates.

        Args:
            data: Input tensor of shape [num_timesteps, batch_size, in_features]
            target: Target labels of shape [batch_size]

        Returns:
            loss: Scalar loss tensor
            pred: Predictions of shape [batch_size, 1]
        """
        num_timesteps, batch_size, _ = data.shape
        device = data.device

        n_out = self.network.n_out

        tgt_onehot = torch.zeros(batch_size, n_out, device=device)
        tgt_onehot.scatter_(1, target.view(-1, 1), 1.0)
        return self._train_recurrent_online(data, tgt_onehot)

    def _train_recurrent_online(self, data: torch.Tensor, tgt_onehot: torch.Tensor):
        """Online recurrent e-prop: update parameters each timestep."""
        num_timesteps, batch_size, _ = data.shape
        device = data.device

        n_in = self.network.n_in
        n_rec = self.network.n_rec
        n_out = self.network.n_out

        self.network.reset(device=device)

        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad()

        x_in_bar = torch.zeros(batch_size, n_in, device=device)
        z_bar = torch.zeros(batch_size, n_rec, device=device)

        trace_in = torch.zeros(batch_size, n_rec, n_in, device=device)
        trace_rec = torch.zeros(batch_size, n_rec, n_rec, device=device)
        trace_out = torch.zeros(batch_size, n_rec, device=device)

        vo = torch.zeros(batch_size, n_out, device=device)
        vo_sum = None

        for t in range(num_timesteps):
            z_t, v_t, vo = self.network.step(data[t], vo)

            if z_t.shape == (n_rec, batch_size):
                z_t = z_t.t()
            if v_t.shape == (n_rec, batch_size):
                v_t = v_t.t()
            if vo.shape == (n_out, batch_size):
                vo = vo.t()

            if z_t.shape != (batch_size, n_rec):
                raise RuntimeError(
                    f"Recurrent spike shape mismatch: expected {(batch_size, n_rec)}, got {tuple(z_t.shape)}"
                )
            if v_t.shape != (batch_size, n_rec):
                raise RuntimeError(
                    f"Recurrent membrane shape mismatch: expected {(batch_size, n_rec)}, got {tuple(v_t.shape)}"
                )
            if vo.shape != (batch_size, n_out):
                raise RuntimeError(
                    f"Output membrane shape mismatch: expected {(batch_size, n_out)}, got {tuple(vo.shape)}"
                )

            vo_sum = vo.clone() if vo_sum is None else (vo_sum + vo)

            if self.update_last and t < num_timesteps - 1:
                continue
            if not ((t + 1) % self.update_every == 0):
                continue

            yo_t = F.softmax(vo, dim=1)
            err_t = yo_t - tgt_onehot

            h_t = self._surrogate_gradient(v_t)

            x_in_bar = self.alpha * x_in_bar + data[t]
            z_bar = self.alpha * z_bar + z_t

            e_in = h_t.unsqueeze(2) * x_in_bar.unsqueeze(1)
            e_rec = h_t.unsqueeze(2) * z_bar.unsqueeze(1)

            trace_in = self.kappa * trace_in + e_in
            trace_rec = self.kappa * trace_rec + e_rec
            trace_out = self.kappa * trace_out + z_t

            L_t = err_t @ self.network.w_out

            w_in_grad_t = self.lr_layer[0] * torch.einsum("br,bri->ri", L_t, trace_in)
            w_rec_grad_t = self.lr_layer[1] * torch.einsum("br,brj->rj", L_t, trace_rec)
            w_out_grad_t = self.lr_layer[2] * (err_t.t() @ trace_out)

            if self.use_optimizer and self.optimizer is not None:
                self.network.w_in.grad = (
                    w_in_grad_t
                    if self.network.w_in.grad is None
                    else self.network.w_in.grad + w_in_grad_t
                )
                self.network.w_rec.grad = (
                    w_rec_grad_t
                    if self.network.w_rec.grad is None
                    else self.network.w_rec.grad + w_rec_grad_t
                )
                self.network.w_out.grad = (
                    w_out_grad_t
                    if self.network.w_out.grad is None
                    else self.network.w_out.grad + w_out_grad_t
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                self.network.w_in.data -= self.lr * w_in_grad_t / batch_size
                self.network.w_rec.data -= self.lr * w_rec_grad_t / batch_size
                self.network.w_out.data -= self.lr * w_out_grad_t / batch_size

        with torch.no_grad():
            pred = vo_sum.argmax(dim=1, keepdim=True)
            loss = self.loss_fn(vo_sum, tgt_onehot)

        return loss.detach(), pred
    
    def reset(self):
        """Reset all LIF neuron states in the network."""
        self.network.reset()
    
    def to(self, device):
        """
        Move trainer and network to device, recreating optimizer if needed.
        
        Args:
            device: Target device
            
        Returns:
            self
        """
        super().to(device)
        
        # Recreate optimizer with new device parameters if we created it
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        
        return self
