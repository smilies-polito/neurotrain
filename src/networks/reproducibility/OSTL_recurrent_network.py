import snntorch as snn
import torch
import torch.nn as nn

from networks.base_snn import BaseSNN


class RecurrentFCNetwork(BaseSNN):
    """
    Multi-layer recurrent SNU/sSNU-style network for OSTL.

    Each layer has:
    - feed-forward weights W_l
    - recurrent weights R_l (same-layer, one-step delayed)
    - Leaky neuron state

    For `recurrent_type == "snu"`, layer outputs are binary spikes.
    For `recurrent_type == "ssnu"`, outputs are smooth activations derived from
    membrane state (sigmoid), matching a soft-SNU-style variant.
    """

    def __init__(
        self,
        layer_sizes,
        beta,
        quant: bool = False,
        threshold: float = 1.0,
        recurrent_type: str = "snu",
    ):
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least [input, output]")

        self.input_size = int(layer_sizes[0])
        self.hidden_size = list(layer_sizes[1:-1])
        self._n_classes = int(layer_sizes[-1])
        self.quant = bool(quant)
        self.threshold = float(threshold)
        self.recurrent_type = str(recurrent_type).lower()

        if self.recurrent_type not in ("snu", "ssnu"):
            raise ValueError(
                f"Unsupported recurrent_type '{recurrent_type}'. Use 'snu' or 'ssnu'."
            )

        ff_layers = []
        rec_layers = []
        lif_layers = []
        for i in range(len(layer_sizes) - 1):
            n_in = int(layer_sizes[i])
            n_out = int(layer_sizes[i + 1])
            ff_layers.append(nn.Linear(n_in, n_out, bias=False))
            rec_layers.append(nn.Linear(n_out, n_out, bias=False))
            lif_layers.append(snn.Leaky(beta=beta, threshold=self.threshold))

        self.feedforward_layers = nn.ModuleList(ff_layers)
        self.recurrent_layers = nn.ModuleList(rec_layers)
        self.lif_layers = nn.ModuleList(lif_layers)

        # Keep this compatibility view for trainers that inspect alternating layers.
        compat_layers = []
        for ff, lif in zip(self.feedforward_layers, self.lif_layers):
            compat_layers.append(ff)
            compat_layers.append(lif)
        self.layers = nn.ModuleList(compat_layers)

        self._prev_outputs = []
        self.reset_parameters()
        self.reset()

        print(f"\n\nNetwork: {self.__class__.__name__}")
        print(f"Layers: {layer_sizes}")
        print(f"Recurrent type: {self.recurrent_type}")
        print(f"Input size: {self.input_size}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Output size: {self._n_classes}")
        print(f"Beta: {beta}")
        print(f"Threshold: {self.threshold}")

    def reset_parameters(self):
        for ff in self.feedforward_layers:
            nn.init.xavier_uniform_(ff.weight)
        for rec in self.recurrent_layers:
            nn.init.orthogonal_(rec.weight)
            rec.weight.data.mul_(0.1)

    def _soft_output(self, mem: torch.Tensor) -> torch.Tensor:
        # sSNU-style soft activation from membrane state.
        return torch.sigmoid(mem / max(self.threshold, 1e-6))

    def forward(self, x: torch.Tensor):
        if not self._prev_outputs or self._prev_outputs[0].device != x.device:
            self.reset(device=x.device)

        if self._prev_outputs[0].shape[0] != x.shape[0]:
            self._prev_outputs = [
                torch.zeros(x.shape[0], ff.out_features, device=x.device)
                for ff in self.feedforward_layers
            ]

        out = x
        out_rec = []
        mem_rec = []
        next_prev_outputs = []

        for idx, (ff, rec, lif) in enumerate(
            zip(self.feedforward_layers, self.recurrent_layers, self.lif_layers)
        ):
            cur = ff(out) + rec(self._prev_outputs[idx])
            spk, mem = lif(cur)

            layer_out = self._soft_output(mem) if self.recurrent_type == "ssnu" else spk
            if self.quant:
                lif.mem.copy_(torch.trunc(lif.mem))

            out_rec.append(layer_out)
            mem_rec.append(mem)
            next_prev_outputs.append(layer_out.detach())
            out = layer_out

        self._prev_outputs = next_prev_outputs
        return out_rec, mem_rec

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset(self, device: torch.device | None = None):
        for lif in self.lif_layers:
            lif.reset_mem()

        dev = device
        if dev is None:
            for ff in self.feedforward_layers:
                dev = ff.weight.device
                break
            if dev is None:
                dev = torch.device("cpu")

        self._prev_outputs = [
            torch.zeros(1, ff.out_features, device=dev)
            for ff in self.feedforward_layers
        ]
