"""
ELL (Event-based Local Learning) Trainer.

Per-layer local classifiers, MSE to one-hot. Membrane/spike detached between
timesteps. Per-step, per-layer backward and optimizer.step().
"""

import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_trainer import BaseTrainer
from networks.local_classifier_network import LocalClassifierNetwork

_ell_debug_done = False
_debug_log_done = False
_DEBUG_LOG_PATH = "/home/ldapusers/bardini/snn-training-benchmarking/.cursor/debug.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {"hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": int(time.time() * 1000)}
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


class ELLTrainer(BaseTrainer):
    """
    ELL trainer: per-step, per-layer backward and update.

    Block uses mode='ell' (detach in recurrence). Trainer computes local MSE,
    backward, optimizer.step() each timestep for each layer.
    """

    def __init__(
        self,
        network: LocalClassifierNetwork,
        lr: float,
        batch_size: int,
        use_raw_input: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.use_raw_input = use_raw_input

        self.optimizers = [
            torch.optim.Adam(block.parameters(), lr=lr, weight_decay=0.0)
            for block in network.blocks
        ]

    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Train on one batch. data: [T, B, F], target: [B]."""
        global _ell_debug_done, _debug_log_done
        num_timesteps, batch_size, _ = data.shape
        device = data.device
        n_classes = self.network.n_classes

        target_onehot = torch.zeros(batch_size, n_classes, device=device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

        self.network.reset()

        # Paper-identical: same input every timestep (constant current into first layer)
        x_const = data.mean(dim=0)
        # When not using raw loader: denormalize MNIST so constant input matches [0,1] scale.
        if not self.use_raw_input and x_const.shape[1] == 784:
            x_const = (x_const * 0.3081 + 0.1307).clamp(0.0, 1.0)

        # #region agent log
        if not _debug_log_done:
            _debug_log("H5", "ell_trainer.py:train_sample", "x_const first batch", {"shape": list(x_const.shape), "min": float(x_const.min().item()), "max": float(x_const.max().item()), "mean": float(x_const.mean().item()), "std": float(x_const.std().item())})
        # #endregion

        spk_sum = torch.zeros(batch_size, n_classes, device=device)
        total_loss = 0.0
        grad_norms_t0 = None
        last_y_hat_spike = None

        for t in range(num_timesteps):
            layer_outputs = self.network.forward_step_all(x_const)

            # Backward in reverse order (last layer first) so graph is not freed
            losses = [
                F.mse_loss(y_hat_spike, target_onehot.detach())
                for _, y_hat_spike in layer_outputs
            ]
            for loss_sup in losses:
                total_loss = total_loss + loss_sup.item()

            for layer_idx in reversed(range(len(layer_outputs))):
                self.optimizers[layer_idx].zero_grad()
                losses[layer_idx].backward(retain_graph=(layer_idx > 0))
                # #region agent log
                if not _debug_log_done and t == 0 and layer_idx == 0:
                    block = self.network.blocks[0]
                    enc_grad = block.encoder.weight.grad
                    dec_grad = block.decoder_y.weight.grad
                    enc_norm = enc_grad.norm().item() if enc_grad is not None else float("nan")
                    dec_norm = dec_grad.norm().item() if dec_grad is not None else float("nan")
                    grad_norms_t0 = (enc_norm, dec_norm)
                # #endregion
                if os.environ.get("SNN_ELL_DEBUG") and not _ell_debug_done and layer_idx == 0:
                    block = self.network.blocks[0]
                    enc_grad = block.encoder.weight.grad
                    dec_grad = block.decoder_y.weight.grad
                    enc_norm = enc_grad.norm().item() if enc_grad is not None else float("nan")
                    dec_norm = dec_grad.norm().item() if dec_grad is not None else float("nan")
                    print("[ELL debug] t=0 grad norms: encoder={:.6f} decoder={:.6f}".format(enc_norm, dec_norm))
                self.optimizers[layer_idx].step()

            spk_sum = spk_sum + layer_outputs[-1][1].detach()
            if t == num_timesteps - 1:
                last_y_hat_spike = layer_outputs[-1][1].detach().clone()

        loss = torch.tensor(
            total_loss / (num_timesteps * len(self.network.blocks)), device=device
        )
        pred = spk_sum.argmax(dim=1)

        # #region agent log
        if not _debug_log_done:
            uniq, cnt = pred.cpu().unique(return_counts=True)
            _debug_log("H1", "ell_trainer.py:train_sample", "pred distribution first batch", {"pred_classes": uniq.tolist(), "pred_counts": cnt.tolist()})
            spk_mean_per_class = spk_sum.mean(dim=0).cpu().tolist()
            _debug_log("H2", "ell_trainer.py:train_sample", "spk_sum first batch", {"min": float(spk_sum.min().item()), "max": float(spk_sum.max().item()), "mean": float(spk_sum.mean().item()), "std": float(spk_sum.std().item()), "mean_per_class": spk_mean_per_class})
            if grad_norms_t0 is not None:
                _debug_log("H3", "ell_trainer.py:train_sample", "grad norms t=0 layer0", {"encoder_norm": grad_norms_t0[0], "decoder_norm": grad_norms_t0[1]})
            acc_spk_sum = pred.eq(target).float().mean().item()
            pred_last = last_y_hat_spike.argmax(dim=1) if last_y_hat_spike is not None else None
            acc_last = pred_last.eq(target).float().mean().item() if pred_last is not None else None
            _debug_log("H4", "ell_trainer.py:train_sample", "readout comparison", {"acc_spk_sum": acc_spk_sum, "acc_last_step": acc_last})
            _debug_log_done = True
        # #endregion

        # One-time diagnostics when SNN_ELL_DEBUG=1 (e.g. to detect single-class collapse)
        if os.environ.get("SNN_ELL_DEBUG") and not _ell_debug_done:
            _ell_debug_done = True
            uniq, cnt = pred.cpu().unique(return_counts=True)
            print("[ELL debug] first batch pred distribution: classes", uniq.tolist(), "counts", cnt.tolist())
            print("[ELL debug] spk_sum: min={:.4f} max={:.4f} mean={:.4f} std={:.4f}".format(
                spk_sum.min().item(), spk_sum.max().item(), spk_sum.mean().item(), spk_sum.std().item()))
            print("[ELL debug] spk_sum per class (mean over batch):", spk_sum.mean(dim=0).cpu().tolist())

        return loss, pred

    def reset(self) -> None:
        self.network.reset()

    def to(self, device):
        """Move trainer and network to device, recreating optimizers."""
        super().to(device)
        self.optimizers = [
            torch.optim.Adam(block.parameters(), lr=self.lr, weight_decay=0.0)
            for block in self.network.blocks
        ]
        return self
