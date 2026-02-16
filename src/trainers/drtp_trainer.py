"""
DRTP (Direct Random Target Projection) trainer for spiking networks.

Uses fixed random feedback matrices to project targets directly to hidden layers,
bypassing backpropagation through the network. Output layer is trained with a
local loss on spike counts, while hidden layers receive target projections.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from trainers.base_trainer import BaseTrainer


class DRTPTrainer(BaseTrainer):
    """
    Direct Random Target Projection trainer for spiking networks.

    Args:
        network: Network to train
        lr: Learning rate
        batch_size: Training batch size
        loss_type: Output loss type ("mse", "bce", "ce")
        feedback_distribution: Distribution for random feedback matrices
        feedback_scale: Multiplicative scale for feedback matrices
        fixed_feedback: If True, keep fixed feedback matrices for the run
        freeze_conv: If True, skip conv weight updates (fixed random front-end)
        quant: Quantization flag (unused; kept for interface compatibility)
        use_optimizer: If True, populate .grad and call optimizer.step()
        optimizer: Optional optimizer instance
        update_last: If True, update only at last timestep
        update_every: Update every N timesteps (default: 1)
    """

    _VALID_DISTS = ("kaiming_uniform", "uniform", "normal")

    def __init__(
        self,
        network: nn.Module,
        lr: float,
        batch_size: int,
        loss_type: str = "mse",
        feedback_distribution: str = "kaiming_uniform",
        feedback_scale: float = 1.0,
        fixed_feedback: bool = True,
        freeze_conv: bool = False,
        quant: bool = False,
        use_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        update_last: bool = False,
        update_every: int = 1,
        **kwargs,
    ):
        super().__init__()

        if feedback_distribution not in self._VALID_DISTS:
            raise ValueError(
                f"feedback_distribution must be one of {self._VALID_DISTS}, got {feedback_distribution}"
            )

        self.network = network
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.feedback_distribution = feedback_distribution
        self.feedback_scale = float(feedback_scale)
        self.fixed_feedback = bool(fixed_feedback)
        self.freeze_conv = bool(freeze_conv)
        self.quant = quant
        self.use_optimizer = use_optimizer
        self.update_last = update_last
        self.update_every = update_every

        self.n_classes = int(getattr(network, "n_classes", 0))
        self.loss_type = str(loss_type).lower()
        self.loss_value = 1.0
        self.loss_fn = nn.MSELoss()
        self._loss_target_kind = "onehot"
        self._output_error_fn = self._mse_error
        self._configure_loss(self.loss_type)

        self.trainable_layers, self.layer_types = self._resolve_trainable_layers()
        self.num_layers = len(self.trainable_layers)
        self.num_hidden = max(self.num_layers - 1, 0)
        if self.num_layers == 0:
            raise ValueError("DRTPTrainer requires a network with trainable layers.")

        self.layer_output_shapes = self._infer_layer_output_shapes()

        # Setup optimizer if requested
        self._external_optimizer = optimizer
        if self.use_optimizer:
            self.optimizer = optimizer or torch.optim.Adam(network.parameters(), lr=self.lr)
        else:
            self.optimizer = None

        # Fixed random feedback matrices (one per hidden layer)
        self.feedback = nn.ParameterList()
        if self.fixed_feedback:
            for shape in self.layer_output_shapes[:-1]:
                fb = torch.empty((self.n_classes, *shape))
                self._init_feedback_(fb)
                self.feedback.append(nn.Parameter(fb, requires_grad=False))

    def _init_feedback_(self, tensor: torch.Tensor) -> torch.Tensor:
        """Initialize feedback weights in-place."""
        if self.feedback_distribution == "kaiming_uniform":
            nn.init.kaiming_uniform_(tensor)
        elif self.feedback_distribution == "uniform":
            tensor.uniform_(-1.0, 1.0)
        elif self.feedback_distribution == "normal":
            tensor.normal_(mean=0.0, std=1.0)
        if self.feedback_scale != 1.0:
            tensor.mul_(self.feedback_scale)
        return tensor

    def _resolve_trainable_layers(self) -> tuple[list[nn.Module], list[str]]:
        if hasattr(self.network, "trainable_layers"):
            layers = list(getattr(self.network, "trainable_layers"))
            types = list(getattr(self.network, "trainable_types", []))
            if not types or len(types) != len(layers):
                types = [
                    "conv" if isinstance(layer, nn.Conv2d) else "linear"
                    for layer in layers
                ]
            return layers, types

        layers = [
            layer for layer in getattr(self.network, "layers", []) if isinstance(layer, nn.Linear)
        ]
        return layers, ["linear"] * len(layers)

    def _infer_layer_output_shapes(self) -> List[tuple[int, ...]]:
        if hasattr(self.network, "layer_output_shapes"):
            shapes = self.network.layer_output_shapes()
        else:
            shapes = []
            for layer, layer_type in zip(self.trainable_layers, self.layer_types):
                if layer_type == "linear":
                    shapes.append((int(layer.out_features),))
                else:
                    raise ValueError(
                        "Conv layer shapes require network.layer_output_shapes()."
                    )

        if len(shapes) != self.num_layers:
            raise ValueError(
                "Mismatch between trainable layers and inferred output shapes."
            )
        return [tuple(int(v) for v in shape) for shape in shapes]

    def _configure_loss(self, loss_type: str) -> None:
        if loss_type in ("mse", "mse_loss"):
            self.loss_fn = nn.MSELoss()
            self.loss_value = 2.0 / max(self.n_classes, 1)
            self._loss_target_kind = "onehot"
            self._output_error_fn = self._mse_error
        elif loss_type in ("bce", "binary_cross_entropy"):
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.loss_value = 1.0 / max(self.n_classes, 1)
            self._loss_target_kind = "onehot"
            self._output_error_fn = self._bce_error
        elif loss_type in ("ce", "cross_entropy"):
            self.loss_fn = nn.CrossEntropyLoss()
            self.loss_value = 1.0
            self._loss_target_kind = "index"
            self._output_error_fn = self._ce_error
        else:
            raise ValueError('loss_type must be one of {"mse", "bce", "ce"}')

    @staticmethod
    def _mse_error(output: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        return output - target_onehot

    @staticmethod
    def _bce_error(output: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(output) - target_onehot

    @staticmethod
    def _ce_error(output: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        return torch.softmax(output, dim=1) - target_onehot

    def _project_targets(
        self, targets_onehot: torch.Tensor, feedback: torch.Tensor
    ) -> torch.Tensor:
        if feedback.dim() == 2:
            return torch.matmul(targets_onehot, feedback)
        flat_fb = feedback.view(self.n_classes, -1)
        proj = torch.matmul(targets_onehot, flat_fb)
        return proj.view(targets_onehot.size(0), *feedback.shape[1:])

    def _sample_feedback(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        """Sample fresh feedback matrices for this batch."""
        mats = []
        for shape in self.layer_output_shapes[:-1]:
            fb = torch.empty((self.n_classes, *shape), device=device, dtype=dtype)
            self._init_feedback_(fb)
            mats.append(fb)
        return mats

    def _accumulate_grad(self, layer: nn.Module, grad_w: torch.Tensor) -> None:
        """Accumulate gradients into layer.weight.grad for optimizer usage."""
        if layer.weight.grad is None:
            layer.weight.grad = grad_w.clone()
        else:
            layer.weight.grad += grad_w

    def _apply_update(self, layer: nn.Module, grad_w: torch.Tensor) -> None:
        """Apply manual or optimizer-backed update."""
        if self.freeze_conv and isinstance(layer, nn.Conv2d):
            return
        if self.use_optimizer and self.optimizer is not None:
            self._accumulate_grad(layer, grad_w)
        else:
            layer.weight.data -= grad_w

    @torch.no_grad()
    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Train on a single batch using DRTP.

        Args:
            data: [timesteps, batch, in_features]
            target: [batch]

        Returns:
            loss: scalar tensor
            pred: [batch, 1] predictions from summed spikes
        """
        num_timesteps = data.shape[0]
        batch_size = data.shape[1]
        device = data.device

        tgt_onehot = torch.zeros(batch_size, self.n_classes, device=device)
        tgt_onehot.scatter_(1, target.unsqueeze(1), 1.0)

        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        feedback = self.feedback if self.fixed_feedback else self._sample_feedback(
            device=device, dtype=data.dtype
        )

        spk_sum = None

        for t in range(num_timesteps):
            spks, _ = self.network(data[t])
            spk_sum = spks[-1] if spk_sum is None else spk_sum + spks[-1]
            layer_inputs = getattr(self.network, "_last_layer_inputs", None)
            layer_spks = getattr(self.network, "_last_layer_spks", None)
            if (
                layer_inputs is None
                or layer_spks is None
                or len(layer_inputs) != self.num_layers
                or len(layer_spks) != self.num_layers
            ):
                layer_inputs = None
                layer_spks = None

            should_update = True
            if self.update_last:
                should_update = t == num_timesteps - 1
            elif self.update_every > 1:
                should_update = (t + 1) % self.update_every == 0

            if not should_update:
                continue

            # Hidden layer updates with target projection
            for layer_idx in range(self.num_hidden):
                if layer_inputs is None:
                    x_pre = data[t] if layer_idx == 0 else spks[layer_idx - 1]
                    x_post = spks[layer_idx]
                else:
                    x_pre = layer_inputs[layer_idx]
                    x_post = layer_spks[layer_idx]

                proj = self._project_targets(tgt_onehot, feedback[layer_idx])
                if not self.use_optimizer:
                    proj = proj * (self.lr / batch_size)
                delta = proj * x_post

                layer = self.trainable_layers[layer_idx]
                if self.layer_types[layer_idx] == "conv":
                    grad_w = torch.nn.grad.conv2d_weight(
                        x_pre,
                        layer.weight.shape,
                        delta,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                    )
                else:
                    grad_w = torch.matmul(delta.transpose(0, 1), x_pre)
                self._apply_update(layer, grad_w)

            # Output layer update (local loss on spikes)
            error = self._output_error_fn(spks[-1], tgt_onehot)
            if layer_inputs is None:
                x_pre_out = spks[-2] if self.num_hidden > 0 else data[t]
            else:
                x_pre_out = layer_inputs[-1]
            if self.use_optimizer:
                loss_grad = error * self.loss_value
            else:
                loss_grad = error * self.loss_value * (self.lr / batch_size)
            if self.layer_types[-1] == "conv":
                grad_out = torch.nn.grad.conv2d_weight(
                    x_pre_out,
                    self.trainable_layers[-1].weight.shape,
                    loss_grad,
                    stride=self.trainable_layers[-1].stride,
                    padding=self.trainable_layers[-1].padding,
                    dilation=self.trainable_layers[-1].dilation,
                    groups=self.trainable_layers[-1].groups,
                )
            else:
                grad_out = torch.matmul(loss_grad.transpose(0, 1), x_pre_out)
            self._apply_update(self.trainable_layers[-1], grad_out)
            if self.use_optimizer and self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        if self._loss_target_kind == "index":
            loss = self.loss_fn(spk_sum, target)
        else:
            loss = self.loss_fn(spk_sum, tgt_onehot)
        pred = spk_sum.argmax(dim=1, keepdim=True)
        return loss.detach(), pred

    def reset(self):
        """Reset network state and optimizer gradients."""
        self.network.reset()
        if self.use_optimizer and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def checkpoint_state(self) -> dict:
        """Return trainer-specific state for checkpointing."""
        if not self.fixed_feedback or len(self.feedback) == 0:
            return {}
        return {"feedback": self.feedback.state_dict()}

    def load_checkpoint_state(self, state: dict) -> None:
        """Restore trainer-specific state from checkpoint."""
        feedback_state = state.get("feedback")
        if feedback_state:
            self.feedback.load_state_dict(feedback_state)

    def to(self, device):
        """
        Move trainer and network to device, recreating optimizer if owned by this trainer.
        """
        super().to(device)
        if self.use_optimizer and self._external_optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        return self
