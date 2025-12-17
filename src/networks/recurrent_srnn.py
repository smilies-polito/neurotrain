import math

import snntorch as snn
import torch
import torch.nn as nn


class RecurrentSRNN(nn.Module):
    """
    Single-layer recurrent spiking network for e-prop training.

    Built on snnTorch's `RLeaky` neuron, while exposing explicit weight tensors
    (`w_in`, `w_rec`, `w_out`) expected by local-learning trainers.

    Notes:
    - The readout is an analog output membrane `vo` with decay `kappa`.
    - The recurrent neuron uses `reset_delay=False` (no one-step spike delay).
    - Recurrent self-connections are masked to zero (diagonal of `w_rec`).
    """

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        threshold: float = 0.6,
        tau_mem: float = 2.0,
        tau_out: float = 0.02,
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
        self.n_classes = n_out
        self.hidden_size = [n_rec]

        self.threshold = float(threshold)
        self.dt = float(dt)
        self.alpha = float(math.exp(-dt / tau_mem))
        self.kappa = float(math.exp(-dt / tau_out))
        self.gamma = float(gamma)
        self.is_recurrent = True

        self.fc_in = nn.Linear(n_in, n_rec, bias=False)
        self.lif = snn.RLeaky(
            beta=self.alpha,
            linear_features=n_rec,
            threshold=self.threshold,
            all_to_all=True,
            learn_recurrent=True,
            reset_mechanism="subtract",
            reset_delay=False,
        )
        self.register_buffer("rec_mask", 1 - torch.eye(n_rec), persistent=True)

        self.w_out = nn.Parameter(torch.empty(n_out, n_rec))
        self.register_buffer(
            "b_out",
            torch.full((n_out,), float(bias_out)),
            persistent=True,
        )

        self.register_buffer("spk", torch.zeros(0), persistent=False)
        self.register_buffer("mem", torch.zeros(0), persistent=False)
        self.register_buffer("vo", torch.zeros(0), persistent=False)

        self.reset_parameters(w_init_gain)
        self.reset(device=device if device is not None else torch.device("cpu"))

    @property
    def w_in(self) -> torch.nn.Parameter:
        return self.fc_in.weight

    @property
    def w_rec(self) -> torch.nn.Parameter:
        return self.lif.recurrent.weight

    def reset_parameters(self, gain: tuple):
        nn.init.kaiming_normal_(self.fc_in.weight)
        self.fc_in.weight.data.mul_(float(gain[0]))

        nn.init.kaiming_normal_(self.lif.recurrent.weight)
        self.lif.recurrent.weight.data.mul_(float(gain[1]))
        self._apply_recurrent_mask()

        nn.init.kaiming_normal_(self.w_out)
        self.w_out.data.mul_(float(gain[2]))

    def _apply_recurrent_mask(self):
        self.w_rec.data.mul_(self.rec_mask.to(self.w_rec.device))

    def reset(self, device: torch.device | None = None):
        dev = device if device is not None else (
            self.spk.device if self.spk.numel() else torch.device("cpu")
        )
        self.spk = torch.zeros(1, self.n_rec, device=dev)
        self.mem = torch.zeros(1, self.n_rec, device=dev)
        self.vo = torch.zeros(1, self.n_out, device=dev)

    def step(self, x: torch.Tensor, vo: torch.Tensor | None = None):
        if x.device != self.spk.device:
            self.spk = self.spk.to(x.device)
            self.mem = self.mem.to(x.device)
            self.vo = self.vo.to(x.device)

        if self.spk.shape[0] != x.shape[0]:
            self.spk = torch.zeros(x.shape[0], self.n_rec, device=x.device)
            self.mem = torch.zeros(x.shape[0], self.n_rec, device=x.device)
            self.vo = torch.zeros(x.shape[0], self.n_out, device=x.device)

        self._apply_recurrent_mask()

        cur = self.fc_in(x)
        z_t, v_t = self.lif(cur, self.spk, self.mem)
        self.spk, self.mem = z_t, v_t

        vo_prev = self.vo if vo is None else vo
        vo_t = self.kappa * vo_prev + z_t @ self.w_out.t() + self.b_out
        self.vo = vo_t
        return z_t, v_t, vo_t

    def forward(self, x: torch.Tensor):
        _, _, vo_t = self.step(x, vo=None)
        return [vo_t], [vo_t]
