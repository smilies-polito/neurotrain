"""
Training and evaluation loops shared across all experiment runs.

Lifted from src/LearningAlgorithms.py — no logic changes, just extracted
so experiment.py doesn't need to import the whole benchmark_runner.
"""

import torch
from torch.utils.data import DataLoader

from trainers.base_trainer import BaseTrainer


def train_one_epoch(
    trainer: BaseTrainer,
    train_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Train for one epoch.

    Args:
        trainer:      Trainer instance implementing BaseTrainer.
        train_loader: Training DataLoader (time-major [T, B, ...]).
        device:       Torch device.

    Returns:
        {"loss": float, "accuracy": float}
    """
    trainer.network.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    non_blocking = device.type == "cuda"

    for data, target in train_loader:
        # Data from loaders is already time-major [T, B, ...]
        data   = data.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        batch_size = data.size(1)

        trainer.reset()
        loss, pred = trainer.train_sample(data, target)

        total_loss    += loss.item() * batch_size
        total_correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += batch_size

    denom = total_samples or 1
    return {
        "loss":     total_loss    / denom,
        "accuracy": total_correct / denom,
    }


@torch.no_grad()
def evaluate(
    network: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate network accuracy on the test set.

    Args:
        network:     Trained SNN (must implement reset() and forward()).
        test_loader: Test DataLoader (time-major [T, B, ...]).
        device:      Torch device.

    Returns:
        Test accuracy as a float in [0, 1].
    """
    network.eval()
    correct = 0
    total   = 0

    non_blocking = device.type == "cuda"
    use_constant = getattr(network, "constant_input_per_timestep", False)

    for data, target in test_loader:
        data   = data.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)

        x_const = data.mean(dim=0) if use_constant else None

        network.reset()
        spk_sum = None
        for t in range(data.size(0)):
            x_t = x_const if x_const is not None else data[t]
            out = network(x_t)

            # Networks return (spk_rec, mem_rec); use last spike layer for readout
            if isinstance(out, (tuple, list)):
                spk = out[0]
                readout = spk[-1] if isinstance(spk, (list, tuple)) else spk
            else:
                readout = out

            spk_sum = readout if spk_sum is None else spk_sum + readout

        preds   = spk_sum.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total   += target.size(0)

    return correct / total if total > 0 else 0.0
