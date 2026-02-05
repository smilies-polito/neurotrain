"""
S-TLLR network: feedforward SNN built from LinearSTLLR layers.

Same interface as FCNetwork for NeuroBench compatibility.
"""

from typing import List, Tuple

import torch

from networks.base_snn import BaseSNN
from networks.stllr_layers import LinearSTLLR


class STLLRNetwork(BaseSNN):
    """
    Feedforward S-TLLR network with LIF + eligibility traces.

    Pattern analogous to FCNetwork's nn.Linear + snn.Leaky blocks.
    """

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
                LinearSTLLR(
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
            if isinstance(layer, LinearSTLLR):
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
            if isinstance(layer, LinearSTLLR):
                layer.reset_state()
