"""
OTPE (Online Training with Predictive Error) trainer.

Implements a forward-in-time learning rule with eligibility traces and predictive
error signals, without modifying the network's forward pass. Uses eligibility traces
to track synaptic activity and predictive error signals for weight updates.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


class OTPETrainer(BaseTrainer):
    """
    Trainer implementing the Online Training with Predictive Error rule.

    Args:
        network: FCNetwork to train (forward pass left untouched)
        lr: Learning rate for manual updates
        batch_size: Training batch size
        trace_decay: Eligibility trace decay (lambda); defaults to neuron leak if None
        surrogate_slope: Slope for sigmoid surrogate derivative
        online_updates: If True apply updates every timestep (online) and step the
            optimizer each step when enabled, otherwise accumulate over the sequence
        quant: Kept for interface compatibility (unused)
        use_optimizer: If True, populate .grad and call optimizer.step()
        optimizer: Optional optimizer instance
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        trace_decay: Optional[float] = None,
        surrogate_slope: float = 10.0,
        online_updates: bool = False,
        quant: bool = False,
        use_optimizer: bool = False,
        optimizer=None,
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.trace_decay = trace_decay
        self.surrogate_slope = surrogate_slope
        self.online_updates = online_updates
        self.quant = quant
        self.use_optimizer = use_optimizer
        self._external_optimizer = optimizer
        self.optimizer = optimizer
        self.threshold = 1.0

        # Linear layers for weight access (network forward left untouched)
        self.linear_layers: List[nn.Linear] = [
            layer
            for layer in getattr(self.network, "layers", [])
            if isinstance(layer, nn.Linear)
        ]
        self.num_layers = len(self.linear_layers)

        if self.trace_decay is None:
            self.trace_decay = self._infer_trace_decay()

        if self.use_optimizer and self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def surrogate_derivative(self, membrane: torch.Tensor) -> torch.Tensor:
        """Sigmoid surrogate derivative around the firing threshold."""
        x = (membrane - self.threshold) * self.surrogate_slope
        sig = torch.sigmoid(x)
        return self.surrogate_slope * sig * (1 - sig)

    def _infer_trace_decay(self) -> float:
        """Default trace decay to the network leak (beta) when available."""
        for layer in getattr(self.network, "layers", []):
            beta = getattr(layer, "beta", None)
            if beta is None:
                continue
            try:
                return float(beta)
            except (TypeError, ValueError):
                try:
                    return float(beta.item())
                except Exception:
                    continue
        return 0.9

    def _compute_layer_gradients(
        self, layer_idx: int, pre_act: torch.Tensor, prev_mem: Optional[torch.Tensor] = None
    ):
        """
        Compute gradients for a single layer using PyTorch autograd.
        
        Equivalent to JAX's nn.vjp for summed_spike and summed_carry.
        Recomputes the forward pass with gradients enabled.
        
        Args:
            layer_idx: Index of the linear layer
            pre_act: Presynaptic activity [batch, in_features]
            prev_mem: Previous membrane potential [batch, out_features] (optional, for LIF dynamics)
        
        Returns:
            ds_dtheta_cur: [in_features, out_features] gradient of spike sum w.r.t. weights
            ds_du_prev: [batch, out_features] gradient of spike sum w.r.t. membrane
            du_cur_dtheta_cur: [in_features, out_features] gradient of membrane sum w.r.t. weights
            du_cur_du_prev: scalar gradient of membrane w.r.t. previous membrane (leak factor)
        """
        layer = self.linear_layers[layer_idx]
        
        with torch.enable_grad():
            # Enable gradients for weight
            layer.weight.requires_grad_(True)
            
            # Recompute linear transformation with gradients enabled
            cur = torch.matmul(pre_act, layer.weight.t())  # [batch, out_features]
            
            # Recompute LIF dynamics with gradients enabled
            # mem_new = beta * mem_prev + cur
            if prev_mem is None:
                # First timestep: no previous membrane
                mem_new = cur.clone().requires_grad_(True)
            else:
                # Enable gradients for prev_mem
                prev_mem_grad = prev_mem.clone().requires_grad_(True)
                mem_new = self.trace_decay * prev_mem_grad + cur
            
            # Compute surrogate derivative for spike gradient
            # This approximates the gradient through the spike function
            ds_du_prev = self.surrogate_derivative(mem_new)  # [batch, out_features]
            
            # CRITICAL FIX: In original JAX, summed_spike returns jnp.sum(s) per sample (scalar)
            # We need to sum over features per sample, then average over batch to match batch processing
            # This ensures ds_dtheta_cur is the average gradient per sample
            spike_sum_per_sample = ds_du_prev.sum(dim=1)  # [batch] - sum over features per sample
            spike_sum = spike_sum_per_sample.mean()  # scalar - average over batch
            
            # Compute gradients w.r.t. weight
            # This gives us ds_dtheta_cur (averaged over batch, matching original per-sample behavior)
            ds_dtheta_cur, = torch.autograd.grad(
                outputs=spike_sum,
                inputs=[layer.weight],
                grad_outputs=torch.ones_like(spike_sum),
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )
            
            # CRITICAL FIX: ds_du_prev should be the surrogate derivative itself, not its gradient
            # In JAX's nn.vjp, ds_du_prev is the gradient of summed_spike w.r.t. u (membrane),
            # which equals the surrogate derivative since summed_spike = sum(surrogate_derivative(u))
            # We already computed ds_du_prev = self.surrogate_derivative(mem_new) above, so use that directly
            
            # Compute membrane sum gradient w.r.t. weights
            # Equivalent to JAX's summed_carry
            # CRITICAL FIX: Sum over features per sample, then average over batch
            mem_sum_per_sample = mem_new.sum(dim=1)  # [batch] - sum over features per sample
            mem_sum = mem_sum_per_sample.mean()  # scalar - average over batch
            du_cur_dtheta_cur, = torch.autograd.grad(
                outputs=mem_sum,
                inputs=[layer.weight],
                grad_outputs=torch.ones_like(mem_sum),
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )
            
            # Disable gradients
            layer.weight.requires_grad_(False)
        
        # du_cur_du_prev is the leak factor (trace_decay)
        du_cur_du_prev = self.trace_decay
        
        # Ensure correct shapes
        # ds_dtheta_cur should be [in_features, out_features]
        if ds_dtheta_cur is not None:
            if ds_dtheta_cur.shape[0] == layer.out_features and ds_dtheta_cur.shape[1] == layer.in_features:
                ds_dtheta_cur = ds_dtheta_cur.t()
        else:
            # Fallback to manual computation if autograd fails
            ds_dtheta_cur = torch.outer(pre_act.mean(dim=0), ds_du_prev.mean(dim=0))
        
        # du_cur_dtheta_cur should be [in_features, out_features]
        if du_cur_dtheta_cur.shape[0] == layer.out_features and du_cur_dtheta_cur.shape[1] == layer.in_features:
            du_cur_dtheta_cur = du_cur_dtheta_cur.t()
        
        # Return ds_du_prev (surrogate derivative) directly, not its gradient
        return ds_dtheta_cur, ds_du_prev, du_cur_dtheta_cur, du_cur_du_prev

    def _apply_update(
        self, layer: nn.Linear, grad_w: torch.Tensor, grad_b: Optional[torch.Tensor] = None
    ) -> None:
        """
        Apply or accumulate weight updates.
        
        Note: grad_w is expected to be [out_features, in_features] (PyTorch format).
        If passed as [in_features, out_features], it will be transposed.
        """
        # Ensure grad_w is in PyTorch format [out_features, in_features]
        if grad_w.shape[0] == layer.in_features and grad_w.shape[1] == layer.out_features:
            grad_w = grad_w.transpose(0, 1)
        
        if self.use_optimizer and self.optimizer is not None:
            if layer.weight.grad is None:
                layer.weight.grad = grad_w.clone()
            else:
                layer.weight.grad += grad_w
            if grad_b is not None and layer.bias is not None:
                if layer.bias.grad is None:
                    layer.bias.grad = grad_b.clone()
                else:
                    layer.bias.grad += grad_b
        else:
            layer.weight.data -= self.lr * grad_w
            if grad_b is not None and layer.bias is not None:
                layer.bias.data -= self.lr * grad_b

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """
        Train on a single batch using OTPE.

        Args:
            data: [num_timesteps, batch, in_features]
            target: [batch]

        Returns:
            loss: scalar tensor (no gradients attached)
            pred: [batch, 1] predictions from summed spikes
        """
        num_timesteps, batch_size, _ = data.shape
        device = data.device
        num_classes = self.network.n_classes

        # One-hot targets for error computation
        target_one_hot = F.one_hot(target, num_classes=num_classes).float()

        # Reset network state
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # Initialize eligibility traces E and predictive error traces R_hat
        # CRITICAL: Shape is [in_features, out_features] to match gradient computations
        E = [
            torch.zeros(layer.in_features, layer.out_features, device=device, dtype=layer.weight.dtype)
            for layer in self.linear_layers
        ]
        R_hat = [
            torch.zeros(layer.in_features, layer.out_features, device=device, dtype=layer.weight.dtype)
            for layer in self.linear_layers
        ]

        # Initialize g_bar (running average of ds_du_prev/sig_tau) per layer
        g_bar = [
            torch.zeros(batch_size, layer.out_features, device=device)
            for layer in self.linear_layers
        ]

        # Initialize ratio for g_bar updates (per layer)
        ratio = [torch.zeros(1, device=device) for _ in self.linear_layers]

        # sig_tau = sigmoid(trace_decay) for g_bar updates
        sig_tau = torch.sigmoid(torch.tensor(self.trace_decay, device=device))

        # Accumulate gradients across time if not updating online
        accum_grads_w = [
            torch.zeros(layer.in_features, layer.out_features, device=device, dtype=layer.weight.dtype)
            for layer in self.linear_layers
        ] if not self.online_updates else None

        total_loss = torch.zeros(1, device=device)
        spk_sum = None

        # Store all spikes and membranes for error propagation
        all_spks = []
        all_mems = []
        
        # Store previous membrane states for gradient computation
        prev_mems = [None] * self.num_layers

        with torch.no_grad():
            for t in range(num_timesteps):
                spks, mems = self.network(data[t])
                all_spks.append(spks)
                all_mems.append(mems)
                
                spk_sum = spks[-1] if spk_sum is None else spk_sum + spks[-1]

                # Presynaptic activities: input for first layer, spikes for subsequent layers
                pre_acts = [data[t]] + spks[:-1]

                # Process each layer
                for l in range(self.num_layers):
                    pre_act = pre_acts[l]  # [batch, in_features]
                    post_spk = spks[l]  # [batch, out_features]
                    post_mem = mems[l]  # [batch, out_features]

                    # Compute gradients using PyTorch autograd
                    # This replaces manual gradient computation with automatic differentiation
                    ds_dtheta_cur, ds_du_prev, du_cur_dtheta_cur, du_cur_du_prev = self._compute_layer_gradients(
                        layer_idx=l,
                        pre_act=pre_act,
                        prev_mem=prev_mems[l]
                    )
                    
                    # Update previous membrane state for next timestep
                    prev_mems[l] = post_mem.detach().clone()

                    # Compute ds_du_prev_mean for eligibility trace update
                    if ds_du_prev.dim() == 1:
                        ds_du_prev_mean = ds_du_prev
                        # Expand to batch dimension for g_bar update
                        ds_du_prev_batch = ds_du_prev.unsqueeze(0).expand(batch_size, -1)
                    else:
                        ds_du_prev_mean = ds_du_prev.mean(dim=0)
                        ds_du_prev_batch = ds_du_prev

                    # Update eligibility trace E (map_u: E = du_cur_du_prev * E + du_cur_dtheta_cur)
                    E[l] = du_cur_du_prev * E[l] + du_cur_dtheta_cur

                    # Compute ds_dtheta: gradient of spike output w.r.t. weights (with history)
                    # map_s: ds_dtheta = ds_du_prev * E + ds_dtheta_cur
                    ds_dtheta = E[l] * ds_du_prev_mean.unsqueeze(0) + ds_dtheta_cur  # [in_features, out_features]

                    # Update predictive error trace R_hat (map_r: R_hat = sig_tau * R_hat + ds_dtheta)
                    R_hat[l] = sig_tau * R_hat[l] + ds_dtheta
                    
                    # #region agent log
                    if t == 0 and l == 0:
                        import json
                        with open('/home/ldapusers/bardini/snn-training-benchmarking/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "id": f"log_otpe_rhat_{t}_{l}",
                                "timestamp": int(torch.tensor(0).item() * 1000),
                                "location": f"otpe_trainer.py:{348}",
                                "message": "R_hat update",
                                "data": {
                                    "t": t,
                                    "layer": l,
                                    "ds_dtheta_mean": float(ds_dtheta.mean().item()),
                                    "ds_dtheta_abs_mean": float(ds_dtheta.abs().mean().item()),
                                    "R_hat_before_mean": float((sig_tau * R_hat[l]).mean().item()),
                                    "R_hat_after_mean": float(R_hat[l].mean().item()),
                                    "sig_tau": float(sig_tau.item()),
                                },
                                "runId": "debug",
                                "hypothesisId": "D"
                            }) + "\n")
                    # #endregion

                    # Update g_bar (running average of ds_du_prev/sig_tau)
                    # ratio manages the ratio between current and previous effects
                    ratio_old = ratio[l]
                    ratio_new = sig_tau * ratio_old + 1.0
                    ratio_val = (sig_tau * ratio_old) / ratio_new
                    ratio[l] = ratio_new

                    # g_bar = ratio * g_bar + (1-ratio) * (ds_du_prev/sig_tau)
                    g_bar[l] = ratio_val * g_bar[l] + (1.0 - ratio_val) * (ds_du_prev_batch / sig_tau)
                    
                    # #region agent log
                    if t == num_timesteps - 1 and l == self.num_layers - 1:
                        import json
                        with open('/home/ldapusers/bardini/snn-training-benchmarking/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "id": f"log_otpe_gbar_final_{l}",
                                "timestamp": int(torch.tensor(0).item() * 1000),
                                "location": f"otpe_trainer.py:{381}",
                                "message": "g_bar final value after forward pass",
                                "data": {
                                    "layer": l,
                                    "timestep": t,
                                    "g_bar_mean": float(g_bar[l].mean().item()),
                                    "g_bar_max": float(g_bar[l].max().item()),
                                    "g_bar_min": float(g_bar[l].min().item()),
                                    "g_bar_abs_mean": float(g_bar[l].abs().mean().item()),
                                    "ds_du_prev_batch_mean": float(ds_du_prev_batch.mean().item()),
                                    "ds_du_prev_batch_abs_mean": float(ds_du_prev_batch.abs().mean().item()),
                                    "sig_tau": float(sig_tau.item()),
                                    "ratio_val": float(ratio_val.item()),
                                },
                                "runId": "debug",
                                "hypothesisId": "F"
                            }) + "\n")
                    # #endregion

                # Compute instantaneous loss (for monitoring)
                logits = mems[-1]
                loss_t = F.cross_entropy(logits, target, reduction="mean") / num_timesteps
                total_loss += loss_t

            # After all timesteps, compute error signal and propagate backward
            # Final output error: difference between prediction and target
            final_logits = all_mems[-1][-1]  # [batch, out_features]
            probs = torch.softmax(final_logits, dim=1)
            error_signal = probs - target_one_hot  # [batch, out_features]
            
            # #region agent log
            import json
            with open('/home/ldapusers/bardini/snn-training-benchmarking/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": "log_otpe_error",
                    "timestamp": int(torch.tensor(0).item() * 1000),
                    "location": f"otpe_trainer.py:{369}",
                    "message": "Error signal computation",
                    "data": {
                        "error_signal_mean": float(error_signal.mean().item()),
                        "error_signal_max": float(error_signal.max().item()),
                        "error_signal_min": float(error_signal.min().item()),
                        "error_signal_abs_mean": float(error_signal.abs().mean().item()),
                        "probs_mean": float(probs.mean().item()),
                        "target_one_hot_sum": int(target_one_hot.sum().item()),
                    },
                    "runId": "debug",
                    "hypothesisId": "C"
                }) + "\n")
            # #endregion

            # Initialize error propagation: g_u[-1] = error_signal
            # CRITICAL FIX: g_u should have one element per layer, each with that layer's output shape
            # Previous bug: was creating one per timestep, all with final layer shape
            g_u = [torch.zeros_like(all_mems[-1][l]) for l in range(self.num_layers)]
            g_u[-1] = error_signal  # [batch, out_features]

            # #region agent log
            import json
            with open('/home/ldapusers/bardini/snn-training-benchmarking/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": "log_otpe_g_u_init",
                    "timestamp": int(torch.tensor(0).item() * 1000),
                    "location": f"otpe_trainer.py:{395}",
                    "message": "g_u initialization",
                    "data": {
                        "g_u_final_shape": list(g_u[-1].shape),
                        "g_u_final_mean": float(g_u[-1].mean().item()),
                        "g_u_final_abs_mean": float(g_u[-1].abs().mean().item()),
                        "error_signal_mean": float(error_signal.mean().item()),
                        "num_layers": self.num_layers,
                    },
                    "runId": "debug",
                    "hypothesisId": "E"
                }) + "\n")
            # #endregion

            # Backpropagate error across layers (using g_bar)
            for l in reversed(range(self.num_layers - 1)):
                # g_u[l] = (g_u[l+1] * g_bar[l+1]) @ weight[l+1]
                # Note: weight[l+1] is [out_features_{l+1}, out_features_l] in PyTorch format
                # g_u[l+1] * g_bar[l+1] is [batch, out_features_{l+1}]
                # Result: [batch, out_features_l]
                
                # #region agent log
                g_u_before = g_u[l].clone()
                g_u_l1_before = g_u[l + 1].clone()
                g_bar_l1 = g_bar[l + 1]
                weight_l1 = self.linear_layers[l + 1].weight
                # #endregion
                
                g_u[l] = torch.matmul(g_u[l + 1] * g_bar[l + 1], self.linear_layers[l + 1].weight)
                
                # #region agent log
                with open('/home/ldapusers/bardini/snn-training-benchmarking/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": f"log_otpe_g_u_prop_{l}",
                        "timestamp": int(torch.tensor(0).item() * 1000),
                        "location": f"otpe_trainer.py:{408}",
                        "message": "g_u error propagation",
                        "data": {
                            "layer": l,
                            "g_u_l1_shape": list(g_u_l1_before.shape),
                            "g_u_l1_mean": float(g_u_l1_before.mean().item()),
                            "g_u_l1_abs_mean": float(g_u_l1_before.abs().mean().item()),
                            "g_bar_l1_shape": list(g_bar_l1.shape),
                            "g_bar_l1_mean": float(g_bar_l1.mean().item()),
                            "g_bar_l1_abs_mean": float(g_bar_l1.abs().mean().item()),
                            "weight_l1_shape": list(weight_l1.shape),
                            "g_u_l_after_shape": list(g_u[l].shape),
                            "g_u_l_after_mean": float(g_u[l].mean().item()),
                            "g_u_l_after_abs_mean": float(g_u[l].abs().mean().item()),
                        },
                        "runId": "debug",
                        "hypothesisId": "E"
                    }) + "\n")
                # #endregion

            # Compute weight updates using R_hat and error signal
            for l, layer in enumerate(self.linear_layers):
                # Weight gradient: R_hat[l] * g_u[l] (per-sample, then sum)
                # Original JAX: g_rec_params = tree_map(lambda x: jnp.squeeze(g[1])*x,R_hat)
                # R_hat[l] is [in_features, out_features]
                # g_u[l] is [batch, out_features]
                # We need: sum over batch of (R_hat[l] * g_u[l] for each sample)
                # R_hat[l] * g_u[l].unsqueeze(0) gives [batch, in_features, out_features] (broadcasted)
                # Then sum over batch: [in_features, out_features]
                grad_w = (R_hat[l].unsqueeze(0) * g_u[l].unsqueeze(1)).sum(dim=0)  # [in_features, out_features]

                # Bias gradient (if exists)
                grad_b = g_u[l].sum(dim=0) / batch_size if layer.bias is not None else None

                # #region agent log
                import json
                error_sum_for_log = g_u[l].sum(dim=0)  # For logging only
                with open('/home/ldapusers/bardini/snn-training-benchmarking/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": f"log_otpe_{l}",
                        "timestamp": int(torch.tensor(0).item() * 1000),
                        "location": f"otpe_trainer.py:{384}",
                        "message": "Weight update computation",
                        "data": {
                            "layer": l,
                            "error_sum_mean": float(error_sum_for_log.mean().item()),
                            "error_sum_max": float(error_sum_for_log.max().item()),
                            "error_sum_min": float(error_sum_for_log.min().item()),
                            "R_hat_mean": float(R_hat[l].mean().item()),
                            "R_hat_max": float(R_hat[l].max().item()),
                            "R_hat_min": float(R_hat[l].min().item()),
                            "grad_w_mean": float(grad_w.mean().item()),
                            "grad_w_max": float(grad_w.max().item()),
                            "grad_w_min": float(grad_w.min().item()),
                            "grad_w_abs_mean": float(grad_w.abs().mean().item()),
                            "lr": self.lr,
                            "weight_before_mean": float(layer.weight.data.mean().item()),
                        },
                        "runId": "debug",
                        "hypothesisId": "A"
                    }) + "\n")
                # #endregion

                if self.online_updates:
                    # Transpose grad_w to [out_features, in_features] for PyTorch format
                    grad_w_T = grad_w.transpose(0, 1)
                    self._apply_update(layer, grad_w_T, grad_b)
                else:
                    accum_grads_w[l] += grad_w

                if self.online_updates and self.use_optimizer and self.optimizer is not None:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

        # Apply accumulated updates after sequence
        if not self.online_updates and accum_grads_w is not None:
            for l, (layer, grad_w) in enumerate(zip(self.linear_layers, accum_grads_w)):
                # Transpose grad_w to [out_features, in_features] for PyTorch format
                grad_w_T = grad_w.transpose(0, 1)
                
                # #region agent log
                import json
                weight_before = layer.weight.data.clone()
                # #endregion
                
                self._apply_update(layer, grad_w_T, None)
                
                # #region agent log
                with open('/home/ldapusers/bardini/snn-training-benchmarking/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": f"log_otpe_apply_{l}",
                        "timestamp": int(torch.tensor(0).item() * 1000),
                        "location": f"otpe_trainer.py:{410}",
                        "message": "Weight update applied",
                        "data": {
                            "layer": l,
                            "grad_w_T_mean": float(grad_w_T.mean().item()),
                            "grad_w_T_abs_mean": float(grad_w_T.abs().mean().item()),
                            "weight_change_mean": float((layer.weight.data - weight_before).abs().mean().item()),
                            "weight_before_mean": float(weight_before.mean().item()),
                            "weight_after_mean": float(layer.weight.data.mean().item()),
                            "lr": self.lr,
                        },
                        "runId": "debug",
                        "hypothesisId": "B"
                    }) + "\n")
                # #endregion
            if self.use_optimizer and self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        # Predictions from spike counts
        pred = spk_sum.argmax(dim=1, keepdim=True)
        return total_loss.detach(), pred

    def reset(self):
        """Reset all LIF states and zero gradients if needed."""
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def to(self, device):
        """
        Move trainer and network to device, recreating optimizer if owned by this trainer.
        """
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self
