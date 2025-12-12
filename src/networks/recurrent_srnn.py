import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentSRNN(nn.Module):
    """
    Single-layer recurrent spiking network mirroring the original e-prop SRNN.

    This module keeps explicit recurrent, input, and output weight matrices and
    exposes a step-wise forward compatible with the framework's trainer and
    evaluation loops (returns [spk_rec], [mem_rec]).

    Key differences vs. feedforward FCNetwork:
    - Explicit recurrent connectivity (w_rec) with self-connections zeroed
    - Analog output membrane (vo) instead of spiking output; we return vo as
      the "spike" output for downstream aggregation (sum over time + argmax)
    - Manual state reset via reset()
    """

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        threshold: float = 0.6,
        tau_mem: float = 2.0,  # seconds (matches original e-prop default)
        tau_out: float = 0.02,  # seconds
        bias_out: float = 0.0,
        gamma: float = 0.3,
        dt: float = 1e-3,
        w_init_gain: tuple = (0.5, 0.1, 0.5),
        device: torch.device | None = None,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.threshold = threshold
        self.dt = dt
        self.alpha = math.exp(-dt / tau_mem)
        self.kappa = math.exp(-dt / tau_out)
        self.gamma = gamma
        self.b_out = bias_out
        self.is_recurrent = True  # flag for trainer to use recurrent path

        # Parameters
        self.w_in = nn.Parameter(torch.empty(n_rec, n_in))
        self.w_rec = nn.Parameter(torch.empty(n_rec, n_rec))
        self.w_out = nn.Parameter(torch.empty(n_out, n_rec))

        # State buffers (allocated in reset)
        self.register_buffer("v", None, persistent=False)   # recurrent mem
        self.register_buffer("vo", None, persistent=False)  # output mem
        self.register_buffer("z", None, persistent=False)   # recurrent spikes

        self.reset_parameters(w_init_gain)
        self.reset(device=device if device is not None else torch.device("cpu"))

    def reset_parameters(self, gain: tuple):
        nn.init.kaiming_normal_(self.w_in)
        self.w_in.data = gain[0] * self.w_in.data
        nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1] * self.w_rec.data
        nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2] * self.w_out.data

    def reset(self, device: torch.device | None = None):
        """Reset membrane and spike states."""
        dev = device if device is not None else (
            self.v.device if self.v is not None else torch.device("cpu")
        )
        self.v = torch.zeros(1, self.n_rec, device=dev)
        self.vo = torch.zeros(1, self.n_out, device=dev)
        self.z = torch.zeros(1, self.n_rec, device=dev)

    def forward(self, x: torch.Tensor):
        """
        Single-timestep forward.

        Args:
            x: Tensor [batch, n_in]

        Returns:
            spk_rec: list with one tensor [batch, n_out] (analog output membrane)
            mem_rec: list with one tensor [batch, n_out] (same as output mem)
        """
        # Ensure states on correct device
        if x.device != self.v.device:
            self.v = self.v.to(x.device)
            self.vo = self.vo.to(x.device)
            self.z = self.z.to(x.device)

        # Zero self-connections
        self.w_rec.data *= (1 - torch.eye(self.n_rec, device=x.device))

        # Expand states to batch
        if self.v.shape[0] != x.shape[0]:
            self.v = torch.zeros(x.shape[0], self.n_rec, device=x.device)
            self.vo = torch.zeros(x.shape[0], self.n_out, device=x.device)
            self.z = torch.zeros(x.shape[0], self.n_rec, device=x.device)

        # Recurrent membrane update
        v_next = (
            self.alpha * self.v
            + x @ self.w_in.t()
            + self.z @ self.w_rec.t()
            - self.z * self.threshold
        )
        z_next = (v_next > self.threshold).float()

        # Output membrane (analog)
        vo_next = self.kappa * self.vo + z_next @ self.w_out.t() + self.b_out

        # Commit state
        self.v = v_next
        self.z = z_next
        self.vo = vo_next

        # Return output membrane as "spike" for downstream aggregation
        return [vo_next], [vo_next]


