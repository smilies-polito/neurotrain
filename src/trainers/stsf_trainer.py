import math
import numpy as np
import torch
import torch.nn as nn

from trainers.base_trainer import BaseTrainer
from networks.fc_network import FCNetwork
from utils.quantizer import fixed_point, check_range, clamp_int_
from utils.parameters import FP_DEC, BW  # adjust if constants live elsewhere

def SFMatrix(size: tuple, lr: float, batch_size: int, loss_value: float, quant: bool = False, optimizer: bool = False, layer_idx: int = None) -> torch.Tensor:
    """
    Generates a sparse, fixed-connectivity feedback matrix for DFA.
    Returns a tensor of shape [n_classes, n_hidden].
    Ensures each row and column of the output has at least one non-zero element.
    Each element is multiplied:
    - multiplied by the learning rate (lr)
    - multiplied by the batch size (batch_size)
    - multiplied by the loss value (loss_value) (for 10 classes, this is 0.2), we can do that since with the actual loss we can have 0.2, -0.2 or 0 with an easy multiplication.
    """
    n_classes, n_hidden = size
    
    # Permutate randomly the hidden neurons indexes and take the first n_classes, those are the feedback connections
    perm = torch.randperm(math.ceil((n_classes // n_hidden) + 1) * n_hidden)
    index = (perm % n_hidden)[:n_classes]
    # Create a the matrix with random values for those connections
    mask = torch.zeros(size, device=perm.device).scatter_(1, index.unsqueeze(1), 1)
    # Ensure each column of mask has at least one non-zero element
    zero_cols = (mask.sum(dim=0) == 0).nonzero().squeeze(1)
    if len(zero_cols) > 0:
        for col in zero_cols:
            row = torch.randint(0, n_classes, (1,), device=perm.device)
            mask[row, col] = 1
    
    bd = np.sqrt(n_hidden / n_classes)
    mat = (2 * bd * torch.rand(size, device=perm.device) - bd) * mask

    # Weight each element by the learning rate and batch size
    # @TODO: implement batch_size also quantized (now only supports batch_size=1), probably I can just divide and everything would be alright
    if not optimizer:
        mat *= (lr)*loss_value
        if quant:
            mat = fixed_point(mat, FP_DEC, BW)
            # In columns with all zeros add a 1 in a random row
            zero_cols = (mat.sum(dim=0) == 0).nonzero().squeeze(1)
            if len(zero_cols) > 0:
                for col in zero_cols:
                    row = torch.randint(0, n_classes, (1,), device=perm.device)
                    mat[row, col] = 1
        else:
            mat /= batch_size  # Normalize by batch size if not quantized

    # Print and save feedback matrix if run_dir is provided
    nonzero_indices = torch.nonzero(mat)
    list_val = []
    for idx in nonzero_indices:
        value = (mat[idx[0], idx[1]], idx[0], idx[1])
        list_val.append(value)
    list_val.sort(key=lambda x: x[2])  # Sort only by column index
    
    return mat
    
class STSFTrainer(BaseTrainer):
    """
    Trainer for Spiking Time Sparse Feedback (STSF). ASSUMES AT LEAST 1 HIDDEN LAYER.
    - network     : FCNetwork
    - lr          : learning rate
    - beta        : LIF beta for local classifiers
    - batch_size  : batch size for the training
    - quant       : use quantization
    - use_optimizer: whether to use optimizer
    - optimizer   : optimizer instance
    - update_last : update only on last timestep
    - update_every: update every N timesteps
    """

    def __init__(
        self,
        network: FCNetwork,
        lr: float,
        batch_size: int,
        quant: bool = False,
        use_optimizer: bool = False,
        optimizer=None,
        update_last: bool = False,
        update_every: int = 1,
        seq_batch_size: int = 1,
    ):
        super().__init__()
        self.network        = network
        self.lr             = lr
        self.loss_fn        = nn.MSELoss()
        self.loss_value     = 2/network.n_classes
        self.quant          = quant
        self.use_optimizer  = use_optimizer
        self.optimizer      = optimizer
        self.update_last    = update_last
        self.update_every   = update_every
        self.seq_batch_size = seq_batch_size
        
        self.stop_requested = False
        
        # For seq_batch_size > 1, accumulate weight updates before applying
        if self.seq_batch_size > 1:
            # Accumulators for hidden and output layer weight updates
            self.dw_hidden_accum = torch.zeros_like(self.network.layers[-4].weight.data)
            self.dw_out_accum = torch.zeros_like(self.network.layers[-2].weight.data)
            self.accum_count = 0

        # fixed feedback matrices
        n_out            = network.n_classes
        hidden_sizes     = network.hidden_size
        self.feedback    = nn.ParameterList([
            nn.Parameter(SFMatrix((n_out, h), lr, batch_size, self.loss_value, self.quant, self.use_optimizer, i), requires_grad=False)
            for i, h in enumerate(hidden_sizes)
        ])

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        data:   [num_timesteps, batch_size, in_features]
        target: [batch_size] labels
        Returns:
            loss: torch scalar
            pred: [batch_size,1] predictions
        """
        # PREPARATION
        num_timesteps, batch_size, _    = data.shape
        num_classes                     = self.network.n_classes
        device                          = data.device
        # one-hot encode target
        tgt = torch.zeros(batch_size, num_classes, device=device)
        tgt.scatter_(1, target.unsqueeze(1), 1.0)
        # RESET
        self.network.reset()
        if self.use_optimizer: self.optimizer.zero_grad()
        spk_sum = None

        for t in range(num_timesteps):
            
            # Always do forward pass regardless of update conditions
            spks, mems = self.network(data[t])                                  # Forward pass through the network to obtain the output spikes and membrane potentials
            spk_sum = spks[-1] if spk_sum is None else spk_sum + spks[-1]       # Accumulate the output spikes over time
            error   = spks[-1] - tgt    # TODO: here is where I could try to change the error function used
            
            # UPDATE LAST
            if self.update_last and (t<num_timesteps-1):
                # If not the last time step, continue to the next time step
                continue
            # UPDATE EVERY N
            if not ((t+1) % self.update_every == 0):
                 continue  # Skip weight updates if not at the specified update interval
            # QUANTIZATION CHECK
            if self.quant:
                check_range(self.network.layers[-2].weight.data, BW, "hidden layer weights")
                check_range(self.network.layers[-4].weight.data, BW, "output layer weights")

            for current_layer in range(len(self.network.hidden_size)):
                # DATA EXTRACTION ----------------------------------------------------------------------------------
                x_pre     = data[t] if current_layer == 0 else spks[current_layer - 1]  # Input spikes for the hidden layer
                x_post    = spks[current_layer]                                         # Output spikes for the current hidden layer

                # HIDDEN LAYER WEIGTH UPDATE -----------------------------------------------------------------------
                loss_hidden = error @ self.feedback[current_layer]  # Direct‐feedback into hidden: [batch_size, hidden_k]
                dw = (loss_hidden * x_post).T @ x_pre               # Combine loss with local plasticity
                if self.use_optimizer:
                    self.network.layers[current_layer*2].weight.grad = dw  # Weight update for optimizer
                elif self.seq_batch_size > 1:
                    # Accumulate weight updates
                    self.dw_hidden_accum += dw
                    if t==num_timesteps-1:  # If last time step, apply accumulated updates
                        self.accum_count += 1
                        # Apply accumulated updates when we reach seq_batch_size
                        if self.accum_count >= self.seq_batch_size:
                            self.network.layers[current_layer*2].weight.data -= self.dw_hidden_accum
                            self.dw_hidden_accum.zero_()
                            self.accum_count = 0
                else:
                    self.network.layers[current_layer*2].weight.data += dw   # Weight update
                if self.quant: clamp_int_(self.network.layers[current_layer*2].weight.data, BW)  # Saturate the weights to ensure no overflow

            # OUTPUT LAYER WEIGHT UPDATE -----------------------------------------------------------------------
            if self.use_optimizer:
                 loss_grad = error * self.loss_value
            else:
                if self.quant:
                    loss_grad = fixed_point(error * self.loss_value * self.lr, FP_DEC, BW)
                else:
                    loss_grad = error * self.loss_value * self.lr / batch_size  # Compute output gradient
            dw_out = loss_grad.T @ x_post  # Combine loss with local plasticity
            if self.use_optimizer:
                self.network.layers[-2].weight.grad = dw_out
            elif self.seq_batch_size > 1:
                # Accumulate output layer weight updates
                self.dw_out_accum += dw_out
                # Apply accumulated updates when we reach seq_batch_size (only need to check once)
                if t==num_timesteps-1:  # If last time step, apply accumulated updates
                    if self.accum_count == 0:  # We already reset in the hidden layer code
                        self.network.layers[-2].weight.data -= self.dw_out_accum
                        self.dw_out_accum.zero_()
            else:
                self.network.layers[-2].weight.data -= dw_out  # Weight update
            # QUANTIZATION CLAMP
            if self.quant:
                clamp_int_(self.network.layers[-2].weight.data, BW)  # Saturate weights to prevent overflow

        # OPTIMIZER STEP
        if self.use_optimizer: self.optimizer.step()

        # LOSS AND PREDICTION
        loss = self.loss_fn(spk_sum, tgt)
        pred = spk_sum.argmax(dim=1, keepdim=True)
            
        return loss, pred


    def reset(self):
        """Reset all LIF states and zero gradients."""
        self.network.reset()