import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


def _to_layer_list(
    value: float | Sequence[float], n_layers: int, name: str
) -> List[float]:
    """Expand scalar hyperparameters to per-layer lists."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != n_layers:
            raise ValueError(f"{name} must have length {n_layers}, got {len(value)}")
        return [float(v) for v in value]
    return [float(value) for _ in range(n_layers)]


class DecolleNetwork(nn.Module):
    """
    Feedforward network implementing the forward dynamics required by DECOLLE.

    The network keeps explicit synaptic (P, Q) and refractory (R) traces and
    exposes a per-timestep forward pass compatible with the framework's
    expectation of returning spike and membrane lists.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        dt: float = 1.0,
        tau_mem: float | Sequence[float] = 20.0,
        tau_syn: float | Sequence[float] = 5.0,
        tau_ref: float | Sequence[float] = 2.0,
        rho: float | Sequence[float] = 1.0,
        delta: float | Sequence[float] = 0.5,
        use_bias: bool = True,
    ):
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output size")

        self.input_size = layer_sizes[0]
        self.hidden_size = list(layer_sizes[1:-1])
        self.n_classes = layer_sizes[-1]
        self.n_layers = len(layer_sizes) - 1

        # Expand per-layer hyperparameters
        self.alpha = _to_layer_list(
            [math.exp(-dt / float(t)) for t in _to_layer_list(tau_mem, self.n_layers, "tau_mem")],
            self.n_layers,
            "tau_mem",
        )
        self.beta = _to_layer_list(
            [math.exp(-dt / float(t)) for t in _to_layer_list(tau_syn, self.n_layers, "tau_syn")],
            self.n_layers,
            "tau_syn",
        )
        self.gamma = _to_layer_list(
            [math.exp(-dt / float(t)) for t in _to_layer_list(tau_ref, self.n_layers, "tau_ref")],
            self.n_layers,
            "tau_ref",
        )
        self.rho = _to_layer_list(rho, self.n_layers, "rho")
        self.delta = _to_layer_list(delta, self.n_layers, "delta")

        # Trainable parameters (updated manually by DECOLLETrainer)
        self.weights = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False)
                for i in range(self.n_layers)
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(layer_sizes[i + 1]), requires_grad=False)
                for i in range(self.n_layers)
            ]
        )
        for lin in self.weights:
            nn.init.xavier_uniform_(lin.weight)

        # State buffers, initialized lazily based on batch size/device
        self.register_buffer("_dummy", torch.tensor(0.0), persistent=False)
        self._initialized = False
        self._batch_size = None  # type: ignore[assignment]
        self.P: List[torch.Tensor] = []
        self.Q: List[torch.Tensor] = []
        self.R: List[torch.Tensor] = []

    def _ensure_state(self, batch_size: int, device: torch.device) -> None:
        """Allocate state tensors if batch size or device changed."""
        if self._initialized and self._batch_size == batch_size:
            return
        self.P = [
            torch.zeros(batch_size, self.weights[i].in_features, device=device)
            for i in range(self.n_layers)
        ]
        self.Q = [
            torch.zeros_like(self.P[i], device=device) for i in range(self.n_layers)
        ]
        self.R = [
            torch.zeros(batch_size, self.weights[i].out_features, device=device)
            for i in range(self.n_layers)
        ]
        self._initialized = True
        self._batch_size = batch_size

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Run one timestep of DECOLLE dynamics.

        Args:
            x: Input spikes for the current timestep, shape [batch, input_size]

        Returns:
            spk_list: List of spikes per layer (post-synaptic), length n_layers
            mem_list: List of membrane potentials U per layer, length n_layers
            p_list:   List of presynaptic PSP traces P per layer (before update)
        """
        if x.dim() != 2 or x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input shape [batch, {self.input_size}], got {tuple(x.shape)}"
            )

        device = x.device
        batch_size = x.shape[0]
        self._ensure_state(batch_size, device)

        spk_list: List[torch.Tensor] = []
        mem_list: List[torch.Tensor] = []
        p_list: List[torch.Tensor] = []

        s_prev = x
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            p_prev = self.P[idx]
            q_prev = self.Q[idx]
            r_prev = self.R[idx]

            # State updates FIRST (matching original DECOLLE implementation)
            # Q integrates presynaptic spikes
            q_next = self.beta[idx] * q_prev + (1.0 - self.beta[idx]) * s_prev
            # P integrates Q (double-exponential synapse)
            p_next = self.alpha[idx] * p_prev + (1.0 - self.alpha[idx]) * q_next

            # Membrane potential using UPDATED P trace
            u = torch.matmul(p_next, weight.weight.t()) - self.rho[idx] * r_prev + bias
            s = (u >= 0).float()

            # Refractory state update after spike
            r_next = self.gamma[idx] * r_prev + (1.0 - self.gamma[idx]) * s

            spk_list.append(s)
            mem_list.append(u)
            p_list.append(p_next)  # Return updated P for learning rule

            self.Q[idx] = q_next
            self.P[idx] = p_next
            self.R[idx] = r_next
            s_prev = s

        return spk_list, mem_list, p_list

    def reset(self) -> None:
        """Reset all state variables; reallocated on next forward call."""
        self._initialized = False
        self._batch_size = None
        self.P = []
        self.Q = []
        self.R = []

