import snntorch as snn
import torch
import torch.nn as nn

from networks.base_snn import BaseSNN


#from ..utils.quantizer import fixed_point

class FCNetwork(BaseSNN):
    """
    Feedforward network with Leaky Integrate-and-Fire (LIF) neurons.
    layer_sizes: [in, hidden1, …, hiddenK, out]
    beta: leakiness parameter
    """
    def __init__(self, layer_sizes, beta, quant=False, threshold: float = 1.0):
        super().__init__()
        self.input_size = layer_sizes[0]
        self.hidden_size = layer_sizes[1:-1]
        self._n_classes = layer_sizes[-1]
        # I am including the quantization parameters but I don't plan to use them for now
        self.quant          = quant
        self.threshold      = float(threshold)

        layers = []
        for i in range(len(layer_sizes) - 1):
            #threshold_val = fixed_point(1.0, fp_dec=FP_DEC, bitwidth=BW) if self.quant else 1.0
            threshold_val = self.threshold
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))
            layers.append(snn.Leaky(beta=beta, threshold=threshold_val))
        # The network structure is now [Linear, LIF, Linear, LIF, ..., Linear, LIF] and saved in a PyTorch ModuleList
        self.layers = nn.ModuleList(layers)

        self.reset_parameters()

        # Print well formated infos about the network
        print(f"\n\nNetwork: {self.__class__.__name__}")
        print(f"Layers: {layer_sizes}")
        print(f"Modules: {self.layers}")
        print(f"Input size: {self.input_size}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Output size: {self._n_classes}")
        print(f"Beta: {beta}")
        print(f"Threshold: {self.threshold}")

    def reset_parameters(self):
        """
        Initialize weights of the network.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                #if (self.quant): layer.weight.data = fixed_point(layer.weight.data, FP_DEC, BW)

    def forward(self, x: torch.Tensor):
        spk = x
        spk_rec, mem_rec = [], []
        for fc, lif in zip(self.layers[0::2], self.layers[1::2]):
            cur = fc(spk)
            spk, mem = lif(cur)
            if self.quant:
                lif.mem.copy_(torch.trunc(lif.mem))
            spk_rec.append(spk)
            mem_rec.append(mem)
        return spk_rec, mem_rec

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def reset(self):
        for layer in self.layers:
            if isinstance(layer, snn.Leaky):
                layer.reset_mem()
