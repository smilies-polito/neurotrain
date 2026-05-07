"""
Training and evaluation loops shared across all experiment runs.

Lifted from src/LearningAlgorithms.py — no logic changes, just extracted
so experiment.py doesn't need to import the whole benchmark_runner.
"""

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from trainers.base_trainer import BaseTrainer


def _wrap(loader, desc: str, progress: bool):
    if progress and _tqdm is not None:
        return _tqdm(loader, desc=desc, leave=False, unit="batch")
    return loader


def train_one_epoch(
    trainer: BaseTrainer,
    train_loader: DataLoader,
    device: torch.device,
    progress: bool = False,
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

    for data, target in _wrap(train_loader, "train", progress):
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
    progress: bool = False,
) -> float:
    """
    Evaluate network accuracy on the test set.

    Args:
        network:     Trained SNN (must implement reset() and forward()).
        test_loader: Test DataLoader (time-major [T, B, ...]).
        device:      Torch device.

    Returns:
        Test accuracy as a float in [0, 1].

    Note:
        For networks with out_integrator=True (TP-style LI head), the readout is
        the membrane potential at the FINAL timestep only — the integrator already
        accumulates across time, so summing over T would over-count. For spike-output
        networks (out_integrator=False or absent), output spikes are summed across T.
    """
    network.eval()
    correct = 0
    total   = 0

    non_blocking   = device.type == "cuda"
    use_integrator = bool(getattr(network, "out_integrator", False))

    for data, target in _wrap(test_loader, "eval", progress):
        data   = data.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)

        network.reset()
        spk_sum  = None
        last_mem = None
        T = data.size(0)

        for t in range(T):
            x_t = data[t]
            out = network(x_t)

            # Networks return (spk_rec, mem_rec)
            if isinstance(out, (tuple, list)):
                spk_rec, mem_rec = out[0], out[1]
                spk_readout = spk_rec[-1] if isinstance(spk_rec, (list, tuple)) else spk_rec
                mem_readout = mem_rec[-1] if isinstance(mem_rec, (list, tuple)) else mem_rec
            else:
                spk_readout = out
                mem_readout = out

            if use_integrator:
                last_mem = mem_readout   # only the final timestep matters
            else:
                spk_sum = spk_readout if spk_sum is None else spk_sum + spk_readout

        readout = last_mem if use_integrator else spk_sum
        preds   = readout.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total   += target.size(0)

    return correct / total if total > 0 else 0.0
