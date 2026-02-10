"""
ES-D-RTRL (Eligibility-based Structured Diagonal RTRL) trainer for snnTorch-based networks.

Implements the BrainTrace linear-memory online learning algorithm for recurrent
spiking neural networks. Based on:

    [Wang et al., "Model-agnostic linear-memory online learning in spiking
     neural networks," Nature Communications, 2026]

Differs from E-prop by: (1) single optimizer.step() per sequence (accumulate
gradients over all timesteps); (2) configurable etrace_decay for eligibility
traces. All training logic lives in the trainer; the network is passive.
"""

import torch
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from networks.recurrent_srnn import RecurrentSRNN


class ESDRTRLTrainer(BaseTrainer):
    """
    ES-D-RTRL trainer for recurrent SNNs (BrainTrace linear-memory online learning).

    Eligibility traces with etrace_decay; gradient accumulated over the full
    sequence; single optimizer.step() per batch. No BPTT; O(n) memory.

    Attributes:
        network: RecurrentSRNN instance to train
        lr: Learning rate
        batch_size: Batch size for training
        etrace_decay: Eligibility trace decay (default 0.9 from reference)
        gamma: Surrogate gradient magnitude parameter
        lr_layer_norm: Per-layer learning rate modulation (input, hidden, output)
    """

    def __init__(
        self,
        network: RecurrentSRNN,
        lr: float,
        batch_size: int,
        etrace_decay: float = 0.9,
        gamma: float = 0.3,
        lr_layer_norm: tuple = (1.0, 1.0, 1.0),
        use_optimizer: bool = True,
        optimizer=None,
        **kwargs,
    ):
        """
        Initialize ES-D-RTRL trainer.

        Args:
            network: RecurrentSNN to train (RecurrentSRNN).
            lr: Learning rate.
            batch_size: Training batch size.
            etrace_decay: Eligibility trace decay (default 0.9, reference).
            gamma: Surrogate derivative magnitude (default 0.3).
            lr_layer_norm: Per-layer LR modulation (input, hidden, output).
            use_optimizer: Use PyTorch Adam (default True).
            optimizer: Pre-configured optimizer (if None, creates Adam).
            **kwargs: Ignored.
        """
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.etrace_decay = float(etrace_decay)
        self.gamma = float(gamma)
        self.lr_layer = lr_layer_norm
        self.use_optimizer = use_optimizer

        if not (hasattr(network, "is_recurrent") and network.is_recurrent):
            raise TypeError(
                "ESDRTRLTrainer supports recurrent RSNNs only; "
                f"got network={type(network).__name__}."
            )
        self.threshold = float(network.threshold)

        self._external_optimizer = optimizer
        if use_optimizer:
            self.optimizer = (
                optimizer
                if optimizer is not None
                else torch.optim.Adam(network.parameters(), lr=lr)
            )
        else:
            self.optimizer = None

    def _surrogate_gradient(self, mem: torch.Tensor) -> torch.Tensor:
        """Triangular surrogate: gamma * max(0, 1 - |v - theta| / theta)."""
        return self.gamma * torch.clamp(
            1.0 - torch.abs((mem - self.threshold) / self.threshold),
            min=0.0,
        )

    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Train on one batch: scan over time, accumulate grads, single optimizer.step().

        Args:
            data: [T, B, F] input tensor.
            target: [B] class indices.

        Returns:
            loss: Scalar (CE on vo_sum) for reporting.
            pred: [B, 1] predicted class indices.
        """
        num_timesteps, batch_size, _ = data.shape
        device = data.device
        n_in = self.network.n_in
        n_rec = self.network.n_rec
        n_out = self.network.n_out

        tgt_onehot = torch.zeros(batch_size, n_out, device=device)
        tgt_onehot.scatter_(1, target.view(-1, 1), 1.0)

        self.network.reset(device=device)
        if self.optimizer is not None:
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

            vo_sum = vo.clone() if vo_sum is None else (vo_sum + vo)

            yo_t = F.softmax(vo, dim=1)
            err_t = yo_t - tgt_onehot
            h_t = self._surrogate_gradient(v_t)

            x_in_bar = self.etrace_decay * x_in_bar + data[t]
            z_bar = self.etrace_decay * z_bar + z_t

            e_in = h_t.unsqueeze(2) * x_in_bar.unsqueeze(1)
            e_rec = h_t.unsqueeze(2) * z_bar.unsqueeze(1)
            trace_in = self.etrace_decay * trace_in + e_in
            trace_rec = self.etrace_decay * trace_rec + e_rec
            trace_out = self.etrace_decay * trace_out + z_t

            L_t = err_t @ self.network.w_out
            w_in_grad_t = self.lr_layer[0] * torch.einsum("br,bri->ri", L_t, trace_in)
            w_rec_grad_t = self.lr_layer[1] * torch.einsum("br,brj->rj", L_t, trace_rec)
            w_out_grad_t = self.lr_layer[2] * (err_t.t() @ trace_out)

            if self.network.w_in.grad is None:
                self.network.w_in.grad = w_in_grad_t.clone()
                self.network.w_rec.grad = w_rec_grad_t.clone()
                self.network.w_out.grad = w_out_grad_t.clone()
            else:
                self.network.w_in.grad = self.network.w_in.grad + w_in_grad_t
                self.network.w_rec.grad = self.network.w_rec.grad + w_rec_grad_t
                self.network.w_out.grad = self.network.w_out.grad + w_out_grad_t

        if self.optimizer is not None:
            self.optimizer.step()

        with torch.no_grad():
            pred = vo_sum.argmax(dim=1, keepdim=True)
            loss = F.cross_entropy(vo_sum, target)

        return loss.detach(), pred

    def reset(self) -> None:
        """Reset network state."""
        self.network.reset()

    def to(self, device) -> "ESDRTRLTrainer":
        """Move trainer and network to device; recreate optimizer if needed."""
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), lr=self.lr
            )
        return self
