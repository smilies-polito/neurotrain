"""
Trace Propagation (TP) Trainer.

Strictly implements the "Trace Propagation" algorithm from:
    Pes et al. (2026) - "Traces propagation: memory-efficient and scalable
    forward-only learning in spiking neural networks"

Algorithm 1 (Page 17) Implementation.
"""

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class TPTrainer(BaseTrainer):
    """
    Trace Propagation trainer: forward-only learning using eligibility traces.
    
    Implements:
    - Separate Input (Green) and Target (Purple) paths.
    - Local Contrastive Loss (Eq 13-15).
    - Eligibility Traces (Eq 11-12).
    - Forward-only weight updates (Eq 18) applied PER TIME STEP.
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        alpha: float = 0.77,  # Membrane decay (alpha in paper Eq 1)
        beta: float = 0.98,   # Trace decay (beta in paper Eq 11, 12)
        vth: float = 0.66,    # Threshold
        surrogate_scale: float = 1.0, # Paper uses scale 1.0
        train_target_propagator: bool = True,
        use_optimizer: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ):
        super().__init__()

        if batch_size < 2:
            raise ValueError("TP requires batch_size >= 2 for contrastive loss.")

        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        
        # Paper Notation mappings:
        # Paper Eq 1: v_l^t = alpha_l * v_l^{t-1} + ... (Membrane uses alpha)
        # Paper Eq 11: eps_l^t = beta_l * eps_l^{t-1} + ... (Trace uses beta)
        
        self.membrane_decay = alpha    # alpha from args -> membrane decay
        self.trace_decay = beta        # beta from args -> trace decay
        
        self.vth = vth
        self.surrogate_scale = surrogate_scale
        self.train_target_propagator = train_target_propagator

        # Extract layers
        self._extract_layers()
        
        # Initialize Projection Matrix S for Layer 1 Target Path
        # "In the purple path, the one-hot encoded target vector c* is projected to the first layer via S..."
        first_layer_input_dim = self.n_classes
        first_layer_output_dim = self.linear_layers[0].out_features
        
        self.S = nn.Linear(first_layer_input_dim, first_layer_output_dim, bias=False)
        nn.init.kaiming_normal_(self.S.weight)
        
        # Optimizer Setup
        self._external_optimizer = optimizer
        if optimizer:
            self.optimizer = optimizer
        elif use_optimizer:
            params = list(self.network.parameters())
            if self.train_target_propagator:
                params.append(self.S.weight)
            self.optimizer = torch.optim.SGD(params, lr=lr)
        else:
            self.optimizer = None

    def _extract_layers(self):
        """Extract linear layers and LIF parameters."""
        self.linear_layers = []
        self.n_classes = self.network.n_classes
        
        for layer in self.network.layers:
            if isinstance(layer, nn.Linear):
                self.linear_layers.append(layer)
        
        self.n_layers = len(self.linear_layers)

    def _surrogate_gradient(self, input):
        """ArcTan surrogate gradient."""
        return self.surrogate_scale / (1 + (torch.pi * input).pow(2))

    class SpikeFunction(torch.autograd.Function):
        """
        Spike function with surrogate gradient.
        We need this to allow local autograd on the loss to backprop through the spike operation
        to the weights *internally* for the local update.
        """
        @staticmethod
        def forward(ctx, input, vth, scale):
            ctx.save_for_backward(input)
            ctx.vth = vth
            ctx.scale = scale
            return (input >= vth).float()

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            surrogate = ctx.scale / (1 + (torch.pi * (input - ctx.vth)).pow(2))
            return grad_input * surrogate, None, None

    def spike(self, input):
        return self.SpikeFunction.apply(input, self.vth, self.surrogate_scale)

    def reset(self):
        """Reset network state (not weights)."""
        self.network.reset()
    
    def train_sample(self, data, target):
        """
        Algorithm 1: Trace propagation algorithm.
        
        Args:
            data: [T, B, F]
            target: [B] (class indices)
        """
        T, B, n_features = data.shape
        device = data.device
        
        # 1. Initialize State Variables
        
        # Membrane potentials v_l (Student/Input Path)
        v_student = [torch.zeros(B, lay.out_features, device=device) for lay in self.linear_layers]
        
        # Membrane potentials \tilde{v}_l (Target Path)
        # Note: Usually target path applies to hidden layers. The output layer is linear accumulator.
        # We will track target path up to L-2 (last hidden layer).
        v_target = [torch.zeros(B, lay.out_features, device=device) for lay in self.linear_layers[:-1]]
        
        # Spikes s_l (Student) and \tilde{s}_l (Target)
        s_student = [torch.zeros(B, lay.out_features, device=device) for lay in self.linear_layers]
        s_target = [torch.zeros(B, lay.out_features, device=device) for lay in self.linear_layers[:-1]]
        
        # Eligibility Traces \epsilon_l (Student) and \tilde{\epsilon}_l (Target)
        eps_student = [torch.zeros(B, lay.out_features, device=device) for lay in self.linear_layers]
        eps_target = [torch.zeros(B, lay.out_features, device=device) for lay in self.linear_layers[:-1]]
        
        # Input Trace (Layer 0)
        eps_in = torch.zeros(B, self.linear_layers[0].in_features, device=device)
        eps_in_target = torch.zeros(B, self.n_classes, device=device) # Target trace input
        
        # One-hot target c*
        c_star = F.one_hot(target, self.n_classes).float()
        
        # Output accumulation
        output_sum = torch.zeros(B, self.n_classes, device=device)
        
        # Keep previous spikes for reset mechanism (v_th * s^{t-1})
        s_student_prev = [torch.zeros(B, lay.out_features, device=device) for lay in self.linear_layers]
        s_target_prev = [torch.zeros(B, lay.out_features, device=device) for lay in self.linear_layers[:-1]]

        # Optimization: Zero grad initially
        if self.optimizer:
            self.optimizer.zero_grad()

        # =========================================================================
        # Time Loop (Algorithm 1 Line 2)
        # =========================================================================
        for t in range(T):
            x_t = data[t] # [B, F]
            
            # Reset gradients for per-step update (Algorithm 1 explicitly updates per step)
            # W^{t+1} = W^t - eta * Delta W
            if self.optimizer:
                self.optimizer.zero_grad() # Ensure clean slate for this time step's gradients
            
            # --- Line 4-6: Input and Target Trace Update (Layer 0) ---
            eps_in = self.trace_decay * eps_in.detach() + x_t
            # epsilon^t_0 = beta * epsilon^{t-1}_0 + s^t_0
            # \tilde{\epsilon}^t_0 = beta * \tilde{\epsilon}^{t-1}_0 + c
            eps_in_target = self.trace_decay * eps_in_target.detach() + c_star
            
            # =====================================================================
            # Forward Propagation Loop (Algorithm 1 Line 7-20)
            # =====================================================================
            
            current_input_student = x_t
            # Target path input for Layer 1 is c* (Line 13)
            current_input_target = c_star 
            
            for l_idx, layer in enumerate(self.linear_layers):
                is_output_layer = (l_idx == len(self.linear_layers) - 1)
                
                # --- Student Path (Green) ---
                # v^t_l = alpha * v^{t-1}_l + s^{t}_{l-1} W_l - s^{t-1}_l v_th (Line 9)
                linear_out_student = layer(current_input_student)

                v_student[l_idx] = (
                    self.membrane_decay * v_student[l_idx].detach() 
                    + linear_out_student
                    - self.vth * s_student_prev[l_idx].detach()
                )
                
                if not is_output_layer:
                    # s^t_l = Theta(v^t_l - v_th) (Line 10)
                    s_curr = self.spike(v_student[l_idx])
                    s_student[l_idx] = s_curr
                    s_student_prev[l_idx] = s_curr.detach() # Explicit detach for next step reset
                    
                    # epsilon^t_l = beta * epsilon^{t-1}_l + s^t_l (Line 11)
                    eps_student[l_idx] = self.trace_decay * eps_student[l_idx].detach() + s_curr
                    
                    current_input_student = s_curr # Input for next layer
                else:
                    # Output Layer (Linear Integrator)
                    output_sum += v_student[l_idx] 
                    # No spike, no trace update for output layer in standard TP logic

                # --- Target Path (Purple) ---
                if not is_output_layer:
                    # Line 12-15: If l==1, W_l = S, s~_0 = c
                    if l_idx == 0:
                        linear_out_target = self.S(current_input_target)
                    else:
                        linear_out_target = layer(current_input_target)
                    
                    # \tilde{v}^t_l = alpha \tilde{v}^{t-1}_l + ...
                    v_target[l_idx] = (
                        self.membrane_decay * v_target[l_idx].detach()
                        + linear_out_target
                        - self.vth * s_target_prev[l_idx].detach()
                    )
                    
                    # \tilde{s}^t_l = Theta(...)
                    s_curr_t = self.spike(v_target[l_idx])
                    s_target[l_idx] = s_curr_t
                    s_target_prev[l_idx] = s_curr_t.detach()
                    
                    # \tilde{\epsilon}^t_l = beta * ...
                    eps_target[l_idx] = self.trace_decay * eps_target[l_idx].detach() + s_curr_t
                    
                    current_input_target = s_curr_t

            # =====================================================================
            # Contrastive Similarity and Synaptic Update (Line 22-29)
            # =====================================================================
            # Loop hidden layers
            for l_idx in range(len(self.linear_layers) - 1):
                layer = self.linear_layers[l_idx]
                
                # Get traces
                e_l = eps_student[l_idx]        # [B, H]
                e_l_tilde = eps_target[l_idx]   # [B, H]
                
                # Previous layer traces
                if l_idx == 0:
                    e_prev_tilde = eps_in_target # [B, C]
                else:
                    e_prev_tilde = eps_target[l_idx - 1] # [B, H_prev]
                
                # z^t_l (Eq 14)
                z_l = torch.matmul(e_l, e_l_tilde.t()) # [B, B]
                
                # y^t_l (Eq 15)
                # Compute pairwise distance on e_prev_tilde
                e_prev_flat = e_prev_tilde.flatten(1)
                diff = e_prev_flat.unsqueeze(1) - e_prev_flat.unsqueeze(0)
                dist_sq = diff.pow(2).sum(-1)
                
                # y_l is target of CE loss
                y_l = F.softmax(-dist_sq, dim=1).detach()
                
                # Soft-target Cross Entropy Loss
                loss_l = torch.sum(-y_l * F.log_softmax(z_l, dim=1), dim=1).mean()
                
                # Gradients
                if l_idx == 0 and self.train_target_propagator:
                    grads = torch.autograd.grad(loss_l, [layer.weight, self.S.weight], retain_graph=False)
                    grad_w = grads[0]
                    grad_s = grads[1]
                    
                    if self.S.weight.grad is None:
                        self.S.weight.grad = grad_s
                    else:
                        self.S.weight.grad += grad_s
                else:
                    grad_w = torch.autograd.grad(loss_l, layer.weight, retain_graph=False)[0]
                
                if layer.weight.grad is None:
                    layer.weight.grad = grad_w
                else:
                    layer.weight.grad += grad_w

            # =====================================================================
            # Output Layer Update (Delta Rule)
            # =====================================================================
            # Section 3.1: "final layer update... standard error"
            output_layer = self.linear_layers[-1]
            
            # Using instantaneous prediction at step t
            # Ideally this should probably use accumulated potential at end or running average?
            # But consistent with per-step update:
            y_t = F.softmax(v_student[-1], dim=1) # [B, C] using v_student[-1] which includes current input
            err = (y_t - c_star)
            
            # Input to output layer
            input_trace_to_out = eps_student[-2] if len(eps_student) > 1 else eps_in
            
            # Gradient dW = err * input^T  (Note: standard CE grad is y-t, but on linear output it's (y-t) if MSE? Or CE if CrossEntropyLossWithLogits?
            # Paper mentions "simple integrator... predicted class corresponds to the neuron with highest integration value".
            # Equation 13 uses softmax on similarities for hidden.
            # Output often uses CE on potential.
            # dL/dW_out = (y - t) * x^T.
            
            grad_out = torch.matmul(err.t(), input_trace_to_out) / B
            
            if output_layer.weight.grad is None:
                output_layer.weight.grad = grad_out
            else:
                output_layer.weight.grad += grad_out

            # ===================================================================
            # UPDATE WEIGHTS (Per Time Step)
            # ===================================================================
            # Algorithm 1: Line 28: W^{t+1} = W^t - eta * Delta W
            if self.optimizer:
                self.optimizer.step()
                self.optimizer.zero_grad() 
                # Zeroing here prevents accumulation to next step. 
                # Note: This is computationally expensive with SGD optimizer.
                # Manual update would be faster but optimizer allows momentum etc (although paper implies simple SGD).
                # SNNTorch benchmarks use optimizers.
                
        # =========================================================================
        # End of Time Loop
        # =========================================================================
        
        # Calculate final loss/acc for logging
        with torch.no_grad():
            final_loss = F.cross_entropy(output_sum, target)
            pred = output_sum.argmax(dim=1)
            
        return final_loss, pred

    def to(self, device):
        super().to(device)
        self.S = self.S.to(device)
        return self
