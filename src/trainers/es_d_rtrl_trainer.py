from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer


@dataclass
class StepLayer:
    """
    Container for one spiking layer at one timestep.

    Mapping to the paper:
    - syn_input ~ x^t
    - preact    ~ I^t = θ ⊛ x^t
    - mem       ~ h^t
    - spk       ~ spike output z^t
    """
    name: str
    synapse: nn.Module
    neuron: nn.Module
    syn_input: torch.Tensor
    preact: torch.Tensor
    spk: torch.Tensor
    mem: torch.Tensor
    is_recurrent: bool = False
    rec_weight: torch.Tensor | None = None


class ESDRTRLTrainer(BaseTrainer):
    """
    Practical ES-D-RTRL-style trainer supporting:
      - RSNN
      - FCSNN
      - ConvSNN
      - VGG9-style DVSGesture SNN

    Paper correspondence:
    - Eq. (2): online gradient accumulation over time
    - Eq. (6): eps^t ≈ eps_f^t ⊗ eps_x^t
    - Eq. (7): eps_x^t = alpha * eps_x^(t-1) + x^t
    - Eq. (8): eps_f^t = alpha * diag(D^t) ∘ eps_f^(t-1) + (1-alpha) * diag(D_f^t)
    """

    def __init__(
        self,
        network,
        lr: float,
        batch_size: int,
        etrace_decay: float = 0.9,
        gamma: float = 0.3,
        lr_layer_norm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_optimizer: bool = True,
        optimizer=None,
        detach_state_each_step: bool = True,
        debug_unused_learning_signals: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.etrace_decay = float(etrace_decay)
        self.gamma = float(gamma)
        self.lr_layer = tuple(float(x) for x in lr_layer_norm)
        self.use_optimizer = bool(use_optimizer)
        self.detach_state_each_step = bool(detach_state_each_step)
        self.debug_unused_learning_signals = bool(debug_unused_learning_signals)

        self.model_kind = self._detect_model_kind(network)

        self._external_optimizer = optimizer
        if use_optimizer:
            self.optimizer = (
                optimizer
                if optimizer is not None
                else torch.optim.Adam(network.parameters(), lr=self.lr)
            )
        else:
            self.optimizer = None

    # ============================================================
    # Model detection / reset / state detachment
    # ============================================================

    def _detect_model_kind(self, net) -> str:
        probe = getattr(net, "core", net)

        if hasattr(probe, "input_layers") and hasattr(probe, "recurrent_layers") and hasattr(probe, "fc_out"):
            return "rsnn"
        if hasattr(probe, "synapses") and hasattr(probe, "neurons"):
            return "fcsnn"
        if all(hasattr(probe, name) for name in ("conv1", "pool1", "lif1", "conv2", "pool2", "lif2", "fc", "lif_out")):
            return "convsnn"
        if (
            hasattr(probe, "conv1")
            and hasattr(probe, "lif1")
            and hasattr(probe, "head")
            and hasattr(probe, "VGG9_CFG")
        ):
            return "vgg9"

        raise TypeError(
            f"Unsupported network type: {type(net).__name__}. "
            "Expected RSNN, FCSNN, ConvSNN, or VGG9-style attributes."
        )

    def _reset_network(self, device: torch.device | None = None) -> None:
        target = self.network
        if hasattr(target, "reset"):
            try:
                target.reset(device=device)
            except TypeError:
                target.reset()
            return

        target = getattr(self.network, "core", None)
        if target is not None and hasattr(target, "reset"):
            try:
                target.reset(device=device)
            except TypeError:
                target.reset()
            return

        raise AttributeError(f"Network {type(self.network).__name__} has no reset() method.")

    def _detach_internal_state_(self) -> None:
        if not self.detach_state_each_step:
            return

        tensor_state_names = (
            "mem", "spk", "syn", "reset", "thresholded", "state",
            "syn_exc", "syn_inh", "cur", "u", "v"
        )

        for module in self.network.modules():
            for name in tensor_state_names:
                if hasattr(module, name):
                    val = getattr(module, name)
                    if isinstance(val, torch.Tensor):
                        setattr(module, name, val.detach())

        core = getattr(self.network, "core", None)
        if core is not None:
            for name in dir(core):
                if name.startswith("mem"):
                    val = getattr(core, name, None)
                    if isinstance(val, torch.Tensor):
                        setattr(core, name, val.detach())
            if hasattr(core, "head") and hasattr(core.head, "mem"):
                if isinstance(core.head.mem, torch.Tensor):
                    core.head.mem = core.head.mem.detach()

    # ============================================================
    # Surrogate / Jacobian diagonal
    # ============================================================

    def _get_threshold(self, neuron: nn.Module) -> float:
        thr = getattr(neuron, "threshold", 1.0)
        if isinstance(thr, torch.Tensor):
            return float(thr.detach().item())
        return float(thr)

    def _get_beta(self, neuron: nn.Module) -> float:
        beta = getattr(neuron, "beta", 1.0)
        if isinstance(beta, torch.Tensor):
            return float(beta.detach().item())
        return float(beta)

    def _surrogate_gradient(self, mem: torch.Tensor, neuron: nn.Module) -> torch.Tensor:
        threshold = self._get_threshold(neuron)
        return self.gamma * torch.clamp(
            1.0 - torch.abs((mem - threshold) / threshold),
            min=0.0,
        )

    def _hidden_jacobian_diag(self, z_prev: torch.Tensor, neuron: nn.Module) -> torch.Tensor:
        beta = self._get_beta(neuron)
        return beta * (1.0 - z_prev)

    # ============================================================
    # Forward step builders
    # ============================================================

    def _forward_step_layers_rsnn(self, x_t: torch.Tensor) -> list[StepLayer]:
        if x_t.dim() != len(self.network.in_shape) + 1:
            x_t = x_t.view(x_t.shape[0], *self.network.in_shape)

        spk_in = x_t.reshape(x_t.shape[0], -1)
        layers: list[StepLayer] = []

        for i, (fc, rlif) in enumerate(zip(self.network.input_layers, self.network.recurrent_layers)):
            cur = fc(spk_in)
            spk, mem = rlif(cur)
            layers.append(
                StepLayer(
                    name=f"hidden_{i}",
                    synapse=fc,
                    neuron=rlif,
                    syn_input=spk_in,
                    preact=cur,
                    spk=spk,
                    mem=mem,
                    is_recurrent=True,
                    rec_weight=rlif.recurrent.weight,
                )
            )
            spk_in = spk

        cur_out = self.network.fc_out(spk_in)
        spk_out, mem_out = self.network.lif_out(cur_out)
        layers.append(
            StepLayer(
                name="output",
                synapse=self.network.fc_out,
                neuron=self.network.lif_out,
                syn_input=spk_in,
                preact=cur_out,
                spk=spk_out,
                mem=mem_out,
                is_recurrent=False,
            )
        )
        return layers

    def _forward_step_layers_fcsnn(self, x_t: torch.Tensor) -> list[StepLayer]:
        if x_t.dim() != len(self.network.in_shape) + 1:
            x_t = x_t.view(x_t.shape[0], *self.network.in_shape)

        spk = x_t.reshape(x_t.shape[0], -1)
        layers: list[StepLayer] = []

        for i, (fc, lif) in enumerate(zip(self.network.synapses, self.network.neurons)):
            cur = fc(spk)
            spk_next, mem = lif(cur)
            layers.append(
                StepLayer(
                    name=f"layer_{i}",
                    synapse=fc,
                    neuron=lif,
                    syn_input=spk,
                    preact=cur,
                    spk=spk_next,
                    mem=mem,
                    is_recurrent=False,
                )
            )
            spk = spk_next

        return layers

    def _forward_step_layers_convsnn(self, x_t: torch.Tensor) -> list[StepLayer]:
        if x_t.dim() != 4:
            x_t = x_t.view(x_t.shape[0], *self.network.in_shape)

        layers: list[StepLayer] = []

        syn_in1 = x_t
        cur1 = self.network.pool1(self.network.conv1(syn_in1))
        spk1, mem1 = self.network.lif1(cur1)
        layers.append(
            StepLayer(
                name="conv1",
                synapse=self.network.conv1,
                neuron=self.network.lif1,
                syn_input=syn_in1,
                preact=cur1,
                spk=spk1,
                mem=mem1,
                is_recurrent=False,
            )
        )

        syn_in2 = spk1
        cur2 = self.network.pool2(self.network.conv2(syn_in2))
        spk2, mem2 = self.network.lif2(cur2)
        layers.append(
            StepLayer(
                name="conv2",
                synapse=self.network.conv2,
                neuron=self.network.lif2,
                syn_input=syn_in2,
                preact=cur2,
                spk=spk2,
                mem=mem2,
                is_recurrent=False,
            )
        )

        syn_in3 = spk2.flatten(1)
        cur3 = self.network.fc(syn_in3)
        spk3, mem3 = self.network.lif_out(cur3)
        layers.append(
            StepLayer(
                name="output",
                synapse=self.network.fc,
                neuron=self.network.lif_out,
                syn_input=syn_in3,
                preact=cur3,
                spk=spk3,
                mem=mem3,
                is_recurrent=False,
            )
        )
        return layers

    def _forward_step_layers_vgg9(self, x_t: torch.Tensor) -> list[StepLayer]:
        net = getattr(self.network, "core", self.network)

        if x_t.dim() != 4:
            raise ValueError(f"Expected VGG9 input [B, C, H, W], got shape {tuple(x_t.shape)}")

        layers: list[StepLayer] = []
        x = x_t

        for i, (_, _, pool_type) in enumerate(net.VGG9_CFG, start=1):
            syn_in = x
            conv = getattr(net, f"conv{i}")
            lif = getattr(net, f"lif{i}")
            mem_prev = getattr(net, f"mem{i}")

            preact = conv(syn_in)
            spk, mem = lif(preact, mem_prev)
            setattr(net, f"mem{i}", mem)

            # IMPORTANT:
            # - store spk BEFORE pooling in StepLayer so spk and mem match in shape
            # - use pooled output only for propagation to next layer
            x_next = spk
            if pool_type != "none":
                x_next = getattr(net, f"pool{i}")(x_next)

            layers.append(
                StepLayer(
                    name=f"conv{i}",
                    synapse=conv,
                    neuron=lif,
                    syn_input=syn_in,
                    preact=preact,
                    spk=spk,      # pre-pooling spike
                    mem=mem,      # same shape as spk
                    is_recurrent=False,
                )
            )

            x = x_next

        syn_in_out = x.flatten(1)
        preact_out = net.head(syn_in_out)

        layers.append(
            StepLayer(
                name="output",
                synapse=net.head.fc,
                neuron=net.lif8,
                syn_input=syn_in_out,
                preact=preact_out,
                spk=preact_out,
                mem=preact_out,
                is_recurrent=False,
            )
        )
        return layers

    def _forward_step_layers(self, x_t: torch.Tensor) -> list[StepLayer]:
        if self.model_kind == "rsnn":
            return self._forward_step_layers_rsnn(x_t)
        if self.model_kind == "fcsnn":
            return self._forward_step_layers_fcsnn(x_t)
        if self.model_kind == "convsnn":
            return self._forward_step_layers_convsnn(x_t)
        if self.model_kind == "vgg9":
            return self._forward_step_layers_vgg9(x_t)
        raise RuntimeError(f"Unknown model kind: {self.model_kind}")

    def _forward_step(self, x_t: torch.Tensor, vo: torch.Tensor | None = None):
        """
        Backward-compatible wrapper.

        New internal API:
            _forward_step_layers(x_t) -> list[StepLayer]

        Legacy API expected by some tests:
            _forward_step(x_t, vo) -> (z_t, v_t, vo)
        """
        layers = self._forward_step_layers(x_t)

        if vo is None:
            return layers

        if len(layers) >= 2:
            hidden_layer = layers[-2]
        else:
            hidden_layer = layers[-1]

        output_layer = layers[-1]
        return hidden_layer.spk, hidden_layer.mem, output_layer.mem

    # ============================================================
    # Gradient helpers
    # ============================================================

    def _group_lr_scale(self, layer: StepLayer) -> float:
        if layer.name == "output":
            return self.lr_layer[2]
        if layer.is_recurrent:
            return self.lr_layer[1]
        return self.lr_layer[0]

    def _accumulate_param_grad(self, param: torch.Tensor, grad: torch.Tensor | None) -> None:
        if grad is None:
            return
        if param.grad is None:
            param.grad = grad.detach().clone()
        else:
            param.grad.add_(grad.detach())

    def _compute_synapse_grad(
        self,
        layer: StepLayer,
        grad_output: torch.Tensor,
    ) -> torch.Tensor | None:
        weight = getattr(layer.synapse, "weight", None)
        if weight is None:
            return None

        grad_w = torch.autograd.grad(
            outputs=layer.preact,
            inputs=weight,
            grad_outputs=grad_output,
            retain_graph=True,
            allow_unused=True,
        )[0]
        return grad_w

    # ============================================================
    # Main training loop
    # ============================================================

    def train_sample(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if data.ndim < 3:
            raise ValueError(f"Expected [T, B, ...], got shape {tuple(data.shape)}")

        num_timesteps = data.shape[0]
        device = data.device
        alpha = self.etrace_decay

        self._reset_network(device=device)
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        eps_x: list[torch.Tensor] | None = None
        eps_f: list[torch.Tensor] | None = None
        z_prev: list[torch.Tensor] | None = None
        eps_x_rec: list[torch.Tensor] | None = None

        loss_accum = torch.zeros((), device=device)
        last_output_mem = None

        for t in range(num_timesteps):
            x_t = data[t]
            layers = self._forward_step_layers(x_t)

            if eps_x is None:
                eps_x = [torch.zeros_like(layer.syn_input, device=device) for layer in layers]
                eps_f = [torch.zeros_like(layer.mem, device=device) for layer in layers]
                z_prev = [torch.zeros_like(layer.spk, device=device) for layer in layers]
                eps_x_rec = [
                    torch.zeros_like(layer.spk, device=device) if layer.is_recurrent
                    else torch.tensor([], device=device)
                    for layer in layers
                ]

            assert eps_x is not None and eps_f is not None and z_prev is not None and eps_x_rec is not None

            output_mem = layers[-1].mem
            last_output_mem = output_mem

            loss_t = F.cross_entropy(output_mem, target, reduction="mean")
            loss_accum = loss_accum + loss_t.detach()

            mems = [layer.mem for layer in layers]
            learning_signals_raw = torch.autograd.grad(
                outputs=loss_t,
                inputs=mems,
                retain_graph=True,
                allow_unused=True,
            )

            if self.debug_unused_learning_signals:
                for i, (layer, sig) in enumerate(zip(layers, learning_signals_raw)):
                    if sig is None:
                        print(
                            f"[ES-D-RTRL][debug] timestep={t} layer={i} "
                            f"name={layer.name}: learning signal is None"
                        )

            learning_signals = [
                sig if sig is not None else torch.zeros_like(mem)
                for mem, sig in zip(mems, learning_signals_raw)
            ]

            for i, layer in enumerate(layers):
                # Output layer with threshold=1e9 (integrator head) would give D_f ≈ 0 from
                # the surrogate gradient; use identity (ones) so output weights can still learn.
                is_identity_out = (self.model_kind == "vgg9" and layer.name == "output") or (
                    layer.name == "output" and getattr(self.network, "out_integrator", False)
                )
                if is_identity_out:
                    D_f_diag = torch.ones_like(layer.mem)
                    D_diag = torch.ones_like(layer.mem)
                else:
                    D_f_diag = self._surrogate_gradient(layer.mem, layer.neuron).detach()
                    D_diag = self._hidden_jacobian_diag(z_prev[i], layer.neuron).detach()

                eps_x[i] = alpha * eps_x[i] + layer.syn_input.detach()
                eps_f[i] = alpha * D_diag * eps_f[i] + (1.0 - alpha) * D_f_diag

                Lf = learning_signals[i].detach() * eps_f[i]
                Lf = self._group_lr_scale(layer) * Lf

                grad_w = self._compute_synapse_grad(layer, grad_output=Lf)
                weight = getattr(layer.synapse, "weight", None)
                if weight is not None:
                    self._accumulate_param_grad(weight, grad_w)

                bias = getattr(layer.synapse, "bias", None)
                if bias is not None:
                    grad_b = torch.autograd.grad(
                        outputs=layer.preact,
                        inputs=bias,
                        grad_outputs=Lf,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    self._accumulate_param_grad(bias, grad_b)

                if layer.is_recurrent and layer.rec_weight is not None:
                    if Lf.ndim != 2 or z_prev[i].ndim != 2:
                        raise ValueError(
                            f"Expected 2D recurrent tensors, got Lf={tuple(Lf.shape)}, "
                            f"z_prev={tuple(z_prev[i].shape)}"
                        )

                    eps_x_rec[i] = alpha * eps_x_rec[i] + z_prev[i]
                    grad_rec = self.lr_layer[1] * (Lf.t() @ eps_x_rec[i])
                    self._accumulate_param_grad(layer.rec_weight, grad_rec)

            for i, layer in enumerate(layers):
                z_prev[i] = layer.spk.detach()

            if self.detach_state_each_step:
                self._detach_internal_state_()

        if self.optimizer is not None:
            self.optimizer.step()

        with torch.no_grad():
            assert last_output_mem is not None
            pred = last_output_mem.argmax(dim=1, keepdim=True)
            loss = loss_accum / num_timesteps

        return loss, pred

    # ============================================================
    # Utilities
    # ============================================================

    def normalize_sequence(
        self,
        data: torch.Tensor,
        timesteps: int | None = None,
    ) -> torch.Tensor:
        if data.ndim < 3:
            raise ValueError(f"Expected [T, B, ...], got shape {tuple(data.shape)}")
        return data

    def reset(self, device: torch.device | None = None) -> None:
        self._reset_network(device=device)

    def to(self, device) -> "ESDRTRLTrainer":
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self