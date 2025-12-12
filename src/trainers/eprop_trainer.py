"""
E-prop (Eligibility Propagation) trainer for snnTorch-based networks.

Implements the e-prop learning algorithm adapted for feedforward SNNs
with snnTorch neurons. Based on:

    [G. Bellec et al., "A solution to the learning dilemma for recurrent networks
     of spiking neurons," Nature communications, vol. 11, no. 3625, 2020]

This implementation adapts the original recurrent e-prop algorithm for feedforward
networks, computing eligibility traces and learning signals per layer.

The key e-prop components:
1. Surrogate gradient for non-differentiable spike function
2. Eligibility traces that track which weights contributed to recent activity  
3. Learning signals that propagate error information through feedback weights

For feedforward networks, we use symmetric feedback (output weights) to compute
the learning signal for hidden layers.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from networks.fc_network import FCNetwork
from networks.recurrent_srnn import RecurrentSRNN


class EpropTrainer(BaseTrainer):
    """
    E-prop (Eligibility Propagation) trainer for feedforward SNNs.
    
    Implements a local learning rule that approximates BPTT by:
    1. Computing eligibility traces from local activity
    2. Propagating learning signals through feedback weights
    3. Combining traces and signals for weight updates
    
    Attributes:
        network: FCNetwork instance to train
        lr: Learning rate
        batch_size: Batch size for training
        gamma: Surrogate gradient magnitude parameter
        kappa: Output layer membrane time constant decay factor
        alpha: Hidden layer membrane time constant decay factor
        lr_layer_norm: Per-layer learning rate modulation (input, hidden, output)
    """
    
    def __init__(
        self,
        network: FCNetwork,
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
        if hasattr(network, "is_recurrent") and network.is_recurrent:
            # For recurrent SRNN we derive from network alpha/kappa
            self.alpha = network.alpha
            self.kappa = network.kappa
            self.threshold = network.threshold
        else:
            # Use network's beta if available, otherwise provided defaults
            self.alpha = network.layers[1].beta if hasattr(network.layers[1], "beta") else tau_mem
            self.kappa = tau_out
            self.threshold = network.layers[1].threshold if hasattr(network.layers[1], "threshold") else 1.0
        
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
            
        # Initialize feedback weights for learning signal propagation
        # Feedback weights connect output error to hidden layer eligibility traces
        self._init_feedback_weights()
        
    def _init_feedback_weights(self):
        """Initialize feedback weights for learning signal computation.
        
        For feedforward networks, we use the output layer weights (transposed)
        as feedback weights, implementing symmetric e-prop.
        """
        # For symmetric e-prop, feedback weights are initialized 
        # proportionally to output weights and fixed during training
        n_out = self.network.n_classes
        hidden_sizes = self.network.hidden_size
        
        self.feedback = nn.ParameterList()
        for h in hidden_sizes:
            # Initialize feedback with random weights (random e-prop)
            # Alternative: use w_out.T for symmetric e-prop
            fb = torch.randn(n_out, h) * (1.0 / math.sqrt(h))
            self.feedback.append(nn.Parameter(fb, requires_grad=False))
    
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
        Train on a single batch using e-prop.
        
        Implements e-prop for feedforward SNNs:
        1. Forward pass collecting spikes and membrane potentials
        2. Compute eligibility traces per layer
        3. Compute learning signals from output error
        4. Update weights using trace * signal
        
        Args:
            data: Input tensor of shape [num_timesteps, batch_size, in_features]
            target: Target labels of shape [batch_size]
            
        Returns:
            loss: Scalar loss tensor
            pred: Predictions of shape [batch_size, 1]
        """
        num_timesteps, batch_size, in_features = data.shape
        n_classes = self.network.n_classes if hasattr(self.network, "n_classes") else self.network.n_out
        device = data.device

        # One-hot encode target
        tgt = torch.zeros(batch_size, n_classes, device=device)
        tgt.scatter_(1, target.unsqueeze(1), 1.0)

        if hasattr(self.network, "is_recurrent") and self.network.is_recurrent:
            return self._train_recurrent(data, tgt, target)

        # -------- Feedforward path (existing implementation) -------- #
        self.network.reset()

        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad()

        num_layers = len(self.network.hidden_size) + 1  # hidden layers + output
        weight_grads = [
            torch.zeros_like(self.network.layers[i * 2].weight)
            for i in range(num_layers)
        ]

        all_spks = []
        all_mems = []
        spk_sum = None

        vo = torch.zeros(batch_size, n_classes, device=device)
        vo_rec = []

        for t in range(num_timesteps):
            spks, mems = self.network(data[t])
            all_spks.append(spks)
            all_mems.append(mems)

            if spk_sum is None:
                spk_sum = spks[-1].clone()
            else:
                spk_sum = spk_sum + spks[-1]

            vo = self.kappa * vo + spks[-1]
            vo_rec.append(vo.clone())

        output = F.softmax(torch.stack(vo_rec, dim=0), dim=2)  # [T, B, C]
        error = output - tgt.unsqueeze(0).expand(num_timesteps, -1, -1)

        for t in range(num_timesteps):
            if self.update_last and t < num_timesteps - 1:
                continue
            if not ((t + 1) % self.update_every == 0):
                continue

            spks_t = all_spks[t]
            mems_t = all_mems[t]

            surrogates = [
                self._surrogate_gradient(mems_t[i])
                for i in range(len(self.network.hidden_size))
            ]

            err_t = error[t]

            for layer_idx in range(len(self.network.hidden_size)):
                x_pre = data[t] if layer_idx == 0 else spks_t[layer_idx - 1]
                x_post = spks_t[layer_idx]
                h_t = surrogates[layer_idx]

                L = err_t @ self.feedback[layer_idx]

                modulated_post = L * h_t * x_post
                dw = modulated_post.T @ x_pre
                dw = self.lr_layer[min(layer_idx, len(self.lr_layer) - 1)] * dw
                weight_grads[layer_idx] += dw

            x_out_pre = spks_t[-2] if len(spks_t) > 1 else spks_t[0]
            dw_out = err_t.T @ x_out_pre
            dw_out = self.lr_layer[-1] * dw_out
            weight_grads[-1] += dw_out

        if self.use_optimizer and self.optimizer is not None:
            for i in range(num_layers):
                self.network.layers[i * 2].weight.grad = weight_grads[i]
            self.optimizer.step()
        else:
            for i in range(num_layers):
                self.network.layers[i * 2].weight.data -= (
                    self.lr * weight_grads[i] / batch_size
                )

        loss = self.loss_fn(spk_sum, tgt)
        pred = spk_sum.argmax(dim=1, keepdim=True)

        return loss.detach(), pred

    def _train_recurrent(
        self, data: torch.Tensor, tgt_onehot: torch.Tensor, target: torch.Tensor
    ):
        """
        Recurrent e-prop path closely matching the reference implementation.
        """
        num_timesteps, batch_size, _ = data.shape
        device = data.device
        n_in = self.network.n_in
        n_rec = self.network.n_rec
        n_out = self.network.n_out

        # Reset state holders
        self.network.reset(device=device)

        # Prepare tensors for full sequence (for eligibility computation)
        v = torch.zeros(num_timesteps, batch_size, n_rec, device=device)
        z = torch.zeros_like(v)
        vo = torch.zeros(num_timesteps, batch_size, n_out, device=device)

        # Forward unroll (no autograd)
        for t in range(num_timesteps - 1):
            v[t + 1] = (
                self.alpha * v[t]
                + torch.mm(z[t], self.network.w_rec.t())
                + torch.mm(data[t], self.network.w_in.t())
                - z[t] * self.network.threshold
            )
            z[t + 1] = (v[t + 1] > self.network.threshold).float()
            vo[t + 1] = (
                self.kappa * vo[t]
                + torch.mm(z[t + 1], self.network.w_out.t())
                + self.network.b_out
            )

        yo = F.softmax(vo, dim=2)

        # Surrogate derivatives
        h = self.gamma * torch.clamp(
            1.0 - torch.abs((v - self.network.threshold) / self.network.threshold), min=0.0
        )

        # Eligibility traces (vectorized, matching reference)
        alpha_conv = torch.tensor(
            [self.alpha ** (num_timesteps - i - 1) for i in range(num_timesteps)],
            device=device,
        ).float().view(1, 1, -1)

        trace_in = F.conv1d(
            data.permute(1, 2, 0),
            alpha_conv.expand(n_in, -1, -1),
            padding=num_timesteps,
            groups=n_in,
        )[:, :, 1 : num_timesteps + 1].unsqueeze(1).expand(-1, n_rec, -1, -1)
        trace_in = torch.einsum("tbr,brit->brit", h, trace_in)

        trace_rec = F.conv1d(
            z.permute(1, 2, 0),
            alpha_conv.expand(n_rec, -1, -1),
            padding=num_timesteps,
            groups=n_rec,
        )[:, :, :num_timesteps].unsqueeze(1).expand(-1, n_rec, -1, -1)
        trace_rec = torch.einsum("tbr,brit->brit", h, trace_rec)
        trace_reg = trace_rec

        kappa_conv = torch.tensor(
            [self.kappa ** (num_timesteps - i - 1) for i in range(num_timesteps)],
            device=device,
        ).float().view(1, 1, -1)

        trace_out = F.conv1d(
            z.permute(1, 2, 0),
            kappa_conv.expand(n_rec, -1, -1),
            padding=num_timesteps,
            groups=n_rec,
        )[:, :, 1 : num_timesteps + 1]

        trace_in = F.conv1d(
            trace_in.reshape(batch_size, n_in * n_rec, num_timesteps),
            kappa_conv.expand(n_in * n_rec, -1, -1),
            padding=num_timesteps,
            groups=n_in * n_rec,
        )[:, :, 1 : num_timesteps + 1].reshape(batch_size, n_rec, n_in, num_timesteps)

        trace_rec = F.conv1d(
            trace_rec.reshape(batch_size, n_rec * n_rec, num_timesteps),
            kappa_conv.expand(n_rec * n_rec, -1, -1),
            padding=num_timesteps,
            groups=n_rec * n_rec,
        )[:, :, 1 : num_timesteps + 1].reshape(
            batch_size, n_rec, n_rec, num_timesteps
        )

        err = yo - tgt_onehot
        L = torch.einsum("tbo,or->brt", err, self.network.w_out)

        # Gradients
        w_in_grad = (
            self.lr_layer[0] * torch.sum(L.unsqueeze(2).expand(-1, -1, n_in, -1) * trace_in, dim=(0, 3))
        )
        w_rec_grad = (
            self.lr_layer[1] * torch.sum(L.unsqueeze(2).expand(-1, -1, n_rec, -1) * trace_rec, dim=(0, 3))
        )
        w_out_grad = self.lr_layer[2] * torch.einsum("tbo,brt->or", err, trace_out)

        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad()
            self.network.w_in.grad = w_in_grad
            self.network.w_rec.grad = w_rec_grad
            self.network.w_out.grad = w_out_grad
            self.optimizer.step()
        else:
            self.network.w_in.data -= self.lr * w_in_grad / batch_size
            self.network.w_rec.data -= self.lr * w_rec_grad / batch_size
            self.network.w_out.data -= self.lr * w_out_grad / batch_size

        # Predictions from summed output membrane
        with torch.no_grad():
            spk_sum = vo.sum(dim=0)
            pred = spk_sum.argmax(dim=1, keepdim=True)
            loss = self.loss_fn(spk_sum, tgt_onehot)

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

