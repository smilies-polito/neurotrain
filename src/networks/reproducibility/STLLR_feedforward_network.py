"""
S-TLLR network: feedforward SNN built from LinearSTLLR layers.

Same interface as FCNetwork for NeuroBench compatibility.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base_snn import BaseSNN


class STLLRNetwork(BaseSNN):
    """
    Feedforward S-TLLR network with LIF + eligibility traces.

    Pattern analogous to FCNetwork's nn.Linear + snn.Leaky blocks.
    """
    net_tags = frozenset({"fully_connected"})

    class LinearSTLLR(nn.Linear):
        """
        S-TLLR linear layer: LIF neurons with eligibility traces (forward-only).

        Forward pass computes LIF dynamics, trace_in, trace_out, and Psi.
        Stores all tensors for the trainer to apply the three-factor weight update.
        No autograd.Function — weight updates happen in the trainer.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            threshold: float = 0.6,
            leak: float = 2.0,
            reset_mechanism: str = "soft",
            factors: Optional[Union[List[float], torch.Tensor]] = None,
            device=None,
            dtype=None,
            **kwargs,
        ):
            super().__init__(in_features, out_features, bias, device, dtype)
            self.reset_mechanism = reset_mechanism
            self.register_buffer("leak", torch.tensor(leak))
            self.register_buffer("threshold", torch.tensor(threshold))
            self.eps = 1e-4
            self.gain = nn.Parameter(torch.ones(out_features, 1))

            # State (initialized in _init_states)
            self.u: Optional[torch.Tensor] = None
            self.trace_in: Optional[torch.Tensor] = None
            self.trace_out: Optional[torch.Tensor] = None

            # Stored for trainer (updated each forward)
            self.last_input: Optional[torch.Tensor] = None
            self.last_trace_in: Optional[torch.Tensor] = None
            self.last_trace_out: Optional[torch.Tensor] = None
            self.last_psi: Optional[torch.Tensor] = None
            self.last_output: Optional[torch.Tensor] = None
            self.last_mem: Optional[torch.Tensor] = None

            if factors is None:
                factors = [0.5, 0.8, -0.2, 1.0]
            self.register_buffer(
                "factors", torch.tensor(factors, dtype=torch.float32)
            )

        def get_weight(self) -> torch.Tensor:
            """Layer-normalized weight (mean/var over in_features)."""
            fan_in = self.weight.shape[1]
            mean = self.weight.mean(dim=1, keepdim=True)
            var = self.weight.var(dim=1, keepdim=True) * fan_in + self.eps
            weight = (self.weight - mean) / var.sqrt()
            return weight * self.gain

        def reset_state(self) -> None:
            """Zero membrane and traces (snnTorch reset_mem convention)."""
            if self.u is not None:
                with torch.no_grad():
                    self.u.zero_()
                    self.trace_in.zero_()
                    self.trace_out.zero_()

        def _init_states(self, x: torch.Tensor) -> None:
            batch_size = x.size(0)
            if self.u is None or self.u.shape[0] != batch_size:
                with torch.no_grad():
                    out = F.linear(x, self.weight, None)
                    self.u = torch.zeros_like(out, device=x.device, dtype=x.dtype)
            self.trace_in = torch.zeros(
                batch_size, self.in_features, device=x.device, dtype=x.dtype
            )
            self.trace_out = torch.zeros(
                batch_size, self.out_features, device=x.device, dtype=x.dtype
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass: LIF + eligibility traces.

            Args:
                x: Input [B, in_features]

            Returns:
                Spikes [B, out_features]
            """
            self._init_states(x)
            weight = self.get_weight()
            leak = torch.sigmoid(self.leak)
            thresh = self.threshold.clamp(min=0.5)

            # Trace of pre-synaptic activity (new, for alpha_pre term)
            trace_in = self.factors[1] * self.trace_in + x

            # LIF
            cur = F.linear(x, weight, self.bias)
            mem = leak * self.u + cur
            u_thr = mem - thresh
            out = (u_thr > 0).float()

            # Psi and trace of post-synaptic activity
            psi = 1.0 / torch.pow(100 * torch.abs(u_thr) + 1, 2)
            trace_out_old = self.trace_out  # For alpha_post term in weight update
            trace_out = self.factors[0] * self.trace_out + psi

            # Reset membrane
            if self.reset_mechanism == "hard":
                self.u = mem * (1 - out)
            else:
                self.u = mem - thresh * out

            # Update state for next timestep
            self.trace_in = trace_in.detach()
            self.trace_out = trace_out.detach()

            # Store for trainer (trace_out_old = before adding psi, per reference backward)
            self.last_input = x.detach()
            self.last_trace_in = trace_in.detach()
            self.last_trace_out = trace_out_old.detach()
            self.last_psi = psi.detach()
            self.last_output = out.detach()
            self.last_mem = mem.detach()

            return out

    def __init__(
        self,
        layer_sizes: List[int],
        threshold: float = 0.6,
        leak: float = 2.0,
        factors: List[float] = None,
        reset_mechanism: str = "soft",
        **kwargs,
    ):
        super().__init__()
        self.input_size = layer_sizes[0]
        self.hidden_size = layer_sizes[1:-1]
        self._n_classes = layer_sizes[-1]

        if factors is None:
            factors = [0.5, 0.8, -0.2, 1.0]

        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                STLLRNetwork.LinearSTLLR(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    bias=True,
                    threshold=threshold,
                    leak=leak,
                    factors=factors,
                    reset_mechanism=reset_mechanism,
                )
            )

        self.reset_parameters()

        print(f"\n\nNetwork: {self.__class__.__name__}")
        print(f"Layers: {layer_sizes}")
        print(f"Input size: {self.input_size}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Output size: {self._n_classes}")

    def reset_parameters(self) -> None:
        """Initialize weights."""
        for layer in self.layers:
            if isinstance(layer, STLLRNetwork.LinearSTLLR):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Single timestep forward pass.

        Args:
            x: Input [B, F] (spikes or rate-coded)

        Returns:
            (spk_rec, mem_rec) where spk_rec[-1] has shape [B, n_classes]
        """
        spk_rec: List[torch.Tensor] = []
        mem_rec: List[torch.Tensor] = []
        spk = x
        for layer in self.layers:
            spk = layer(spk)
            spk_rec.append(spk)
            mem_rec.append(layer.last_mem)
        return spk_rec, mem_rec

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset(self) -> None:
        """Clear membrane and trace state (snnTorch convention)."""
        for layer in self.layers:
            if isinstance(layer, STLLRNetwork.LinearSTLLR):
                layer.reset_state()


# Alias for backward compatibility (e.g. from networks.stllr_network import LinearSTLLR)
LinearSTLLR = STLLRNetwork.LinearSTLLR
