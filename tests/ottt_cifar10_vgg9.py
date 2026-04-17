#!/usr/bin/env python3
"""
Single-file minimal integration test:
CIFAR-10 loader + CIFAR10_VGG9 + OTTTTrainer (+ optional Optuna).

Architecture notes:
  CIFAR10_VGG9 uses WSConv2d (Weight Standardization Conv) layers. These are
  nn.Conv2d subclasses, so OTTTTrainer's forward hooks fire on them correctly and
  trace-based gradient substitution is applied to all 8 conv blocks (except the
  first, which is the input layer — standard OTTT exclusion).

  The output head is a LeakyIntegrator (leak=1.0), which accumulates logits
  internally across timesteps via self.mem += F.linear(x, w_std). Its inner
  self.fc (nn.Linear) is bypassed so no hook fires on it. This is acceptable —
  the official OTTT implementation also skips the FC classifier head.

Eval note:
  Because LeakyIntegrator accumulates self.mem internally, spk_rec[-1] at the
  LAST timestep already contains the fully summed logits. We read it directly
  rather than accumulating spike_sum across timesteps.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import types
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Hardcoded Defaults
# -----------------------------------------------------------------------------
# Dataset defaults
BATCH_SIZE = 64         # Mini-batch size used for both training and evaluation.
TIMESTEPS = 25          # Rate-coded CIFAR-10 timesteps (1 rate frame per step).
NUM_WORKERS = 4         # DataLoader worker processes.

# Network defaults
BETA = 0.5              # LIF membrane decay. Paper default for VGG OTTT: tau=2 → beta=0.5.
THRESHOLD = 1.0         # LIF spiking threshold.

# Trainer defaults
EPOCHS = 10             # Training epochs for the default non-Optuna run.
LR = 1e-3               # SGD learning rate (OTTT paper default).
SEED = 42               # Global random seed for Python, NumPy, and PyTorch.
DEVICE = "auto"         # Runtime device selection: auto, cpu, or cuda.
HPC_PRINTS = False      # If True, suppress per-batch progress bar updates.

# Optuna defaults
OPTUNA_TRIALS = 0       # Number of Optuna trials; 0 disables hyperparameter search.
OPTUNA_EPOCHS = 5       # Epochs executed inside each Optuna trial.
STUDY_NAME = "ottt_cifar10_vgg9"  # Optuna study name.
OPTUNA_STORAGE = ""     # Optuna storage URL; empty string keeps the study in memory.

# -----------------------------------------------------------------------------
# Minimal repo bootstrap: make imports work when running from tests/
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

if "networks" not in sys.modules:
    networks_pkg = types.ModuleType("networks")
    networks_pkg.__path__ = [str(SRC_DIR / "networks")]
    sys.modules["networks"] = networks_pkg

from datasets.cifar10_loader import CIFAR10Loader
from networks.benchmarking.vgg9 import vgg9_cifar10 as CIFAR10_VGG9
from trainers.ottt_trainer import OTTTTrainer


# -----------------------------------------------------------------------------
# Tiny utilities (inlined to keep this file self-contained)
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal CIFAR-10+VGG9 OTTT test.")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    p.add_argument("--timesteps", type=int, default=TIMESTEPS, help="Rate-coded CIFAR-10 timesteps.")
    p.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    p.add_argument("--beta", type=float, default=BETA, help="LIF beta.")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="LIF threshold.")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE, help="Execution device.")
    p.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS, help="Number of Optuna trials (0 disables).")
    p.add_argument("--optuna-epochs", type=int, default=OPTUNA_EPOCHS, help="Epochs per Optuna trial.")
    p.add_argument("--study-name", type=str, default=STUDY_NAME, help="Optuna study name.")
    p.add_argument("--optuna-storage", type=str, default=OPTUNA_STORAGE, help="Optuna storage URL (empty=in-memory).")
    p.add_argument("--hpc-prints", dest="hpc_prints", action="store_true", default=HPC_PRINTS, help="Disable incremental batch progress prints.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(requested)


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
def run_training(
    *,
    # Dataset parameters
    batch_size: int,
    timesteps: int,
    # Network parameters
    threshold: float,
    beta: float,
    # General training parameters
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    hpc_prints: bool = False,
    # Optuna parameters
    log_prefix: str = "",
    trial: "optuna.trial.Trial | None" = None,
) -> Dict[str, float]:
    set_seed(seed)

    train_loader, test_loader = CIFAR10Loader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
    )
    network = CIFAR10_VGG9(
        in_channels=3,
        num_classes=10,
        beta=beta,
        threshold=threshold,
        verbose=True,
    ).to(device)
    trainer = OTTTTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        # OTTTTrainer derives trace_decay from network.tau (default tau=2 → trace_decay=0.5).
        # CIFAR10_VGG9 does not expose .tau, so for beta=0.5 the default is already correct
        # (tau=2 ↔ beta=0.5). For other beta values pass trace_decay=beta explicitly.
        online_updates=True,
    ).to(device)
    # CosineAnnealingLR mirrors OTTT paper (T_max=300 epochs in original; scaled to run length).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=epochs)

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_loss = 0.0

    non_blocking = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # [TRAIN] one full epoch on training batches
        trainer.network.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = len(train_loader)

        for i, (data, target) in enumerate(train_loader, 1):
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)

            loss, pred = trainer.train_sample(data, target)
            batch_size_cur = target.size(0)
            total_loss += loss.item() * batch_size_cur
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += batch_size_cur

            if not hpc_prints:
                f = int(28 * i / n_batches)
                print(f"\r  [{'#' * f}{'-' * (28 - f)}] {int(100 * i / n_batches):3d}%  ", end="", flush=True)

        if not hpc_prints:
            print("\r" + " " * 40 + "\r", end="", flush=True)

        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        # [EVAL] one full pass on test batches.
        # CIFAR10_VGG9 uses a LeakyIntegrator head (leak=1.0): self.mem accumulates
        # logits internally across all timesteps. Read the final timestep's spk_rec[-1]
        # directly — do NOT accumulate spike_sum across timesteps (that would double-count).
        network.eval()
        total = 0
        correct = 0
        # Accumulate per-layer average spike rates for monitoring.
        layer_rate_sum = [0.0] * network._num_blocks
        n_eval_batches = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                network.reset()
                T = data.size(0)
                batch_layer_rates = [0.0] * network._num_blocks
                for t in range(T):
                    spk_rec, mem_rec = network(data[t])
                    # Accumulate spike rates for conv blocks (not the head).
                    for li in range(network._num_blocks):
                        batch_layer_rates[li] += spk_rec[li].mean().item()
                # LeakyIntegrator head has already accumulated logits — read last timestep.
                preds = mem_rec[-1].argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += target.size(0)
                for li in range(network._num_blocks):
                    layer_rate_sum[li] += batch_layer_rates[li] / T
                n_eval_batches += 1

        avg_rates = [layer_rate_sum[li] / n_eval_batches for li in range(network._num_blocks)]
        rates_str = " ".join(f"L{li+1}={r:.3f}" for li, r in enumerate(avg_rates))

        test_acc = correct / total if total > 0 else 0.0

        final_train_loss = train_loss
        final_test_acc = test_acc
        best_test_acc = max(best_test_acc, test_acc)

        scheduler.step()
        epoch_time_s = time.perf_counter() - epoch_start

        print(
            f"{log_prefix}epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} "
            f"epoch_time_s={epoch_time_s:.2f} "
            f"spike_rates=[{rates_str}]"
        )

    return {
        "best_test_acc": best_test_acc,
        "final_test_acc": final_test_acc,
        "final_train_loss": final_train_loss,
    }


def run_optuna(args: argparse.Namespace, device: torch.device) -> None:
    try:
        import optuna
    except ImportError as err:
        raise RuntimeError("Optuna is not installed. Install it with `pip install optuna`.") from err

    storage = args.optuna_storage or None
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=storage is not None,
        sampler=sampler,
    )

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        beta = trial.suggest_float("beta", 0.40, 0.70)
        threshold = trial.suggest_float("threshold", 0.5, 1.5)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        result = run_training(
            epochs=args.optuna_epochs,
            batch_size=batch_size,
            timesteps=args.timesteps,
            lr=lr,
            beta=beta,
            threshold=threshold,
            seed=args.seed + trial.number,
            device=device,
            hpc_prints=args.hpc_prints,
            log_prefix=f"[trial {trial.number}] ",
            trial=trial,
        )
        trial.set_user_attr("final_test_acc", result["final_test_acc"])
        return result["best_test_acc"]

    print(f"[Optuna] trials={args.optuna_trials} epochs_per_trial={args.optuna_epochs} study={args.study_name}")
    study.optimize(objective, n_trials=args.optuna_trials)

    print("\n[Optuna] Best trial")
    print(f"value={study.best_value:.4f}")
    print(f"params={study.best_params}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"[Run] device={device.type}")

    if args.optuna_trials > 0:
        run_optuna(args, device)
        return

    result = run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        lr=args.lr,
        beta=args.beta,
        threshold=args.threshold,
        seed=args.seed,
        device=device,
        hpc_prints=args.hpc_prints,
    )
    print(f"\n[Done] final_test_acc={result['final_test_acc']:.4f} best_test_acc={result['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()
