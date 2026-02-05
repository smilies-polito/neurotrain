from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn


class SpikeFunction(torch.autograd.Function):
    """Surrogate gradient spike function used by ETLP."""

    scale = 0.3

    @staticmethod
    def pseudo_derivative(v_scaled: torch.Tensor) -> torch.Tensor:
        return (
            torch.maximum(
                1 - torch.abs(v_scaled), torch.tensor(0.0, device=v_scaled.device)
            )
            * SpikeFunction.scale
        )

    @staticmethod
    def forward(ctx, v_scaled: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v_scaled)
        return (v_scaled > 0).type(v_scaled.dtype)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        (v_scaled,) = ctx.saved_tensors
        dE_dz = dy
        dz_dv_scaled = SpikeFunction.pseudo_derivative(v_scaled)
        return dE_dz * dz_dv_scaled


activation = SpikeFunction.apply


class ETLPNetwork(nn.Module):
    """
    Event-based Three-factor Local Plasticity (ETLP) network.

    This implementation is adapted from the reference ETLP code and exposes
    a forward API compatible with the benchmarking framework.
    """

    NeuronState = namedtuple(
        "NeuronState",
        (
            "V_rec",
            "S_rec",
            "R_rec",
            "A_rec",
            "V_out",
            "S_out",
            "e_trace_in",
            "e_trace_rec",
            "epsilon_v_in",
            "epsilon_v_rec",
            "epsilon_v_out",
            "epsilon_a_in",
            "epsilon_a_rec",
        ),
    )

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        dt: float = 1.0,
        tau_v: float = 80.0,
        tau_a: float = 10.0,
        tau_o: float = 80.0,
        theta: float = 5.0,
        thr: float = 1.0,
        n_ref: int = 5,
        recurrent: bool = False,
        train_rec: bool = False,
        spike_scale: float = 0.3,
        keep_trace: bool = False,
    ) -> None:
        super().__init__()

        self.dt = float(dt)
        self.n_in = int(n_in)
        self.n_rec = int(n_rec)
        self.n_out = int(n_out)
        self.n_classes = self.n_out
        self.n_refractory = float(n_ref)
        self.recurrent = bool(recurrent)
        self.train_rec = bool(train_rec)
        self.keep_trace = bool(keep_trace)

        # Set surrogate gradient scaling
        SpikeFunction.scale = float(spike_scale) / max(float(thr), 1e-6)

        # Weight matrices for plasticity eq. 11/12 in the paper
        self.W_in = nn.Parameter(
            torch.tensor(
                0.2 * np.random.randn(self.n_in, self.n_rec) / np.sqrt(self.n_in)
            ).float(),
            requires_grad=True,
        )
        if self.recurrent:
            recurrent_weights = (
                0.2 * np.random.randn(self.n_rec, self.n_rec) / np.sqrt(self.n_rec)
            )
            self.W_rec = nn.Parameter(
                torch.tensor(
                    recurrent_weights
                    - recurrent_weights * np.eye(self.n_rec, self.n_rec)
                ).float(),
                requires_grad=self.train_rec,
            )
        else:
            self.W_rec = None
        self.W_out = nn.Parameter(
            torch.tensor(
                np.random.randn(self.n_rec, self.n_out) / np.sqrt(self.n_rec)
            ).float(),
            requires_grad=True,
        )

        # Fixed random projection for labels
        self.register_buffer(
            "b_out",
            torch.tensor(
                np.random.randn(self.n_rec, self.n_out) / np.sqrt(self.n_rec)
            ).float(),
        )

        # Identity for optional self-recurrency removal
        self.register_buffer("identity_diag_rec", torch.eye(self.n_rec, self.n_rec))

        # Parameters (decay factors)
        distribution = torch.distributions.gamma.Gamma(3, 3 / float(tau_v))
        tau_v_sample = distribution.rsample((1, self.n_rec)).clamp(3, 100)
        self.register_buffer("decay_v", torch.exp(-self.dt / tau_v_sample).float())
        self.register_buffer(
            "decay_o", torch.tensor(np.exp(-self.dt / float(tau_o))).float()
        )
        self.register_buffer(
            "decay_a", torch.tensor(np.exp(-self.dt / float(tau_a))).float()
        )
        self.register_buffer("thr", torch.tensor(float(thr)).float())
        self.register_buffer("theta", torch.tensor(float(theta)).float())

        self.state = None

    def initialize_state(self, input: torch.Tensor):
        # Initialize the neuron states to zeros
        batch_size = input.shape[0]
        device = input.device

        e_trace_rec = None
        epsilon_v_rec = None
        epsilon_a_rec = None
        if self.recurrent:
            e_trace_rec = torch.zeros(batch_size, self.n_rec, self.n_rec, device=device)
            epsilon_v_rec = torch.zeros(
                batch_size, self.n_rec, self.n_rec, device=device
            )
            epsilon_a_rec = torch.zeros(
                batch_size, self.n_rec, self.n_rec, device=device
            )

        state = self.NeuronState(
            V_rec=torch.zeros(batch_size, self.n_rec, device=device),
            S_rec=torch.zeros(batch_size, self.n_rec, device=device),
            R_rec=torch.zeros(batch_size, self.n_rec, device=device),
            A_rec=torch.zeros(batch_size, self.n_rec, device=device),
            V_out=torch.zeros(batch_size, self.n_out, device=device),
            S_out=torch.zeros(batch_size, self.n_out, device=device),
            e_trace_in=torch.zeros(batch_size, self.n_in, self.n_rec, device=device),
            e_trace_rec=e_trace_rec,
            epsilon_v_in=torch.zeros(batch_size, self.n_in, self.n_rec, device=device),
            epsilon_v_rec=epsilon_v_rec,
            epsilon_v_out=torch.zeros(batch_size, self.n_rec, device=device),
            epsilon_a_in=torch.zeros(batch_size, self.n_in, self.n_rec, device=device),
            epsilon_a_rec=epsilon_a_rec,
        )
        return state

    def reset(self):
        self.state = None

    def detach(self):
        if self.state is None:
            return
        detached = []
        for item in self.state:
            if item is None:
                detached.append(None)
            else:
                detached.append(item.detach())
        self.state = self.NeuronState(*detached)

    def forward(self, input: torch.Tensor):
        if self.state is None:
            self.state = self.initialize_state(input)

        V_rec = self.state.V_rec
        S_rec = self.state.S_rec
        R_rec = self.state.R_rec
        A_rec = self.state.A_rec

        V_out = self.state.V_out
        S_out = self.state.S_out

        e_trace_in = self.state.e_trace_in
        epsilon_a_in = self.state.epsilon_a_in
        epsilon_v_in = self.state.epsilon_v_in
        e_trace_rec = self.state.e_trace_rec
        epsilon_v_rec = self.state.epsilon_v_rec
        epsilon_a_rec = self.state.epsilon_a_rec
        epsilon_v_out = self.state.epsilon_v_out

        with torch.no_grad():
            A = self.thr + self.theta * A_rec
            psi = SpikeFunction.pseudo_derivative((V_rec - A) / self.thr)
            epsilon_a_in = (
                psi[:, None, :] * epsilon_v_in
                + (self.decay_a - psi[:, None, :] * self.theta) * epsilon_a_in
            )
            if (
                self.recurrent
                and epsilon_v_rec is not None
                and epsilon_a_rec is not None
            ):
                epsilon_a_rec = (
                    psi[:, None, :] * epsilon_v_rec
                    + (self.decay_a - psi[:, None, :] * self.theta) * epsilon_a_rec
                )

        # Threshold adaptation
        A_rec = self.decay_a * A_rec + S_rec
        A = self.thr + A_rec * self.theta

        # Detach previous spike for recurrency and reset
        S_rec = S_rec.detach()

        # Current calculation
        if self.recurrent and self.W_rec is not None:
            I_in = torch.mm(input, self.W_in) + torch.mm(S_rec, self.W_rec)
        else:
            I_in = torch.mm(input, self.W_in)

        # Recurrent neurons update
        V_rec_new = self.decay_v * V_rec + I_in - self.thr * S_rec

        # Spike generation
        is_refractory = R_rec > 0
        zeros_like_spikes = torch.zeros_like(S_rec)
        S_rec_new = torch.where(
            is_refractory,
            zeros_like_spikes,
            activation((V_rec_new - A) / self.thr),
        )
        R_rec_new = R_rec + self.n_refractory * S_rec_new - 1
        R_rec_new = torch.clip(R_rec_new, 0.0, self.n_refractory).detach()

        # Forward pass of the data to output weights
        I_out = torch.mm(S_rec_new, self.W_out)

        # Output neurons update
        V_out_new = self.decay_o * V_out + I_out - self.thr * S_out
        S_out_new = activation((V_out - self.thr) / self.thr)

        with torch.no_grad():
            if input.is_sparse:
                epsilon_v_in = (
                    self.decay_v[:, None, :] * epsilon_v_in
                    + input.to_dense()[:, :, None]
                )
            else:
                epsilon_v_in = (
                    self.decay_v[:, None, :] * epsilon_v_in + input[:, :, None]
                )

            if self.recurrent and epsilon_v_rec is not None:
                epsilon_v_rec = (
                    self.decay_v[:, None, :] * epsilon_v_rec + S_rec[:, :, None]
                )

            epsilon_v_out = self.decay_o * epsilon_v_out + S_rec_new

            v_scaled = (V_rec_new - A) / self.thr
            is_refractory = R_rec > 0
            psi_no_ref = SpikeFunction.pseudo_derivative(v_scaled)
            psi = torch.where(is_refractory, torch.zeros_like(psi_no_ref), psi_no_ref)

            if self.keep_trace:
                e_trace_in = e_trace_in * self.decay_o + (
                    psi[:, None, :] * (epsilon_v_in - self.theta * epsilon_a_in)
                )
                if self.recurrent and e_trace_rec is not None:
                    e_trace_rec = e_trace_rec * self.decay_o + (
                        psi[:, None, :] * (epsilon_v_rec - self.theta * epsilon_a_rec)
                    )
            else:
                e_trace_in = psi[:, None, :] * (
                    epsilon_v_in - self.theta * epsilon_a_in
                )
                if self.recurrent and e_trace_rec is not None:
                    e_trace_rec = psi[:, None, :] * (
                        epsilon_v_rec - self.theta * epsilon_a_rec
                    )

        new_state = self.NeuronState(
            V_rec=V_rec_new,
            S_rec=S_rec_new,
            R_rec=R_rec_new,
            A_rec=A_rec,
            V_out=V_out_new,
            S_out=S_out_new,
            e_trace_in=e_trace_in.detach(),
            e_trace_rec=None if e_trace_rec is None else e_trace_rec.detach(),
            epsilon_v_in=epsilon_v_in.detach(),
            epsilon_v_rec=None if epsilon_v_rec is None else epsilon_v_rec.detach(),
            epsilon_v_out=epsilon_v_out.detach(),
            epsilon_a_in=epsilon_a_in.detach(),
            epsilon_a_rec=None if epsilon_a_rec is None else epsilon_a_rec.detach(),
        )

        self.state = new_state

        spk_rec = [S_rec_new, S_out_new]
        mem_rec = [V_rec_new, V_out_new]
        return spk_rec, mem_rec
