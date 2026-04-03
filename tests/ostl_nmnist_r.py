#!/usr/bin/env python3
"""
Single-file minimal integration test:
N-MNIST loader + RSNN + OSTLTrainer (+ optional Optuna).
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
TIMESTEPS = 10          # Number of temporal bins produced by the N-MNIST loader.
NUM_WORKERS = 8        # DataLoader worker processes for N-MNIST loading.

# Network defaults
BETA = 0.9              # LIF membrane decay.
THRESHOLD = 1.0         # LIF spiking threshold.

# General training defaults
EPOCHS = 10             # Training epochs for the default non-Optuna run.
LR = 1e-2               # OSTL learning rate.
SEED = 42               # Global random seed for Python, NumPy, and PyTorch.
DEVICE = "auto"         # Runtime device selection: auto, cpu, or cuda.
HPC_PRINTS = False      # If True, suppress per-batch progress bar updates.

# OSTL-specific defaults
GRAD_CLIP = 0.0         # Gradient clipping (0 = disabled).
DEFERRED = True         # If True, apply weight updates only at the end of the sequence.

# Optuna defaults
OPTUNA_TRIALS = 0       # Number of Optuna trials; 0 disables hyperparameter search.
OPTUNA_EPOCHS = 1       # Epochs executed inside each Optuna trial.
STUDY_NAME = "ostl_nmnist_r"  # Optuna study name.
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

from datasets.nmnist_loader import NMNISTLoader
from networks.benchmarking.r_snn import RSNN
from trainers.ostl_trainer import OSTLTrainer


# -----------------------------------------------------------------------------
# Tiny utilities (inlined to keep this file self-contained)
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal N-MNIST+RSNN OSTL test.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    p.add_argument("--deferred", action="store_true", default=DEFERRED)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    p.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS)
    p.add_argument("--optuna-epochs", type=int, default=OPTUNA_EPOCHS)
    p.add_argument("--study-name", type=str, default=STUDY_NAME)
    p.add_argument("--optuna-storage", type=str, default=OPTUNA_STORAGE)
    p.add_argument("--hpc-prints", dest="hpc_prints", action="store_true", default=HPC_PRINTS)
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
    batch_size: int,
    timesteps: int,
    beta: float,
    threshold: float,
    epochs: int,
    lr: float,
    grad_clip: float,
    deferred: bool,
    seed: int,
    device: torch.device,
    hpc_prints: bool = False,
    log_prefix: str = "",
    trial: "optuna.trial.Trial | None" = None,
) -> Dict[str, float]:
    set_seed(seed)

    train_loader, test_loader = NMNISTLoader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
    )
    network = RSNN(in_shape=(2, 34, 34), num_classes=10, beta=beta, threshold=threshold, reset_mechanism="zero").to(device)
    trainer = OSTLTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        grad_clip=grad_clip,
        deferred=deferred,
    ).to(device)

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_loss = 0.0

    non_blocking = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # [TRAIN]
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

        # [EVAL]
        network.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader, 1):
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                network.reset()
                spike_sum = None
                for t in range(data.size(0)):
                    spk_rec, _ = network(data[t])
                    out_t = spk_rec[-1]
                    spike_sum = out_t if spike_sum is None else spike_sum + out_t
                preds = spike_sum.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += target.size(0)

                if not hpc_prints:
                    f = int(28 * i / len(test_loader))
                    print(f"\r  [{'#' * f}{'-' * (28 - f)}] {int(100 * i / len(test_loader)):3d}% eval", end="", flush=True)

        if not hpc_prints:
            print("\r" + " " * 45 + "\r", end="", flush=True)

        test_acc = correct / total if total > 0 else 0.0

        final_train_loss = train_loss
        final_test_acc = test_acc
        best_test_acc = max(best_test_acc, test_acc)

        epoch_time_s = time.perf_counter() - epoch_start
        print(
            f"{log_prefix}epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} "
            f"epoch_time_s={epoch_time_s:.2f}"
        )

        if trial is not None:
            try:
                import optuna
                trial.report(test_acc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            except ImportError:
                pass

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
        lr = trial.suggest_float("lr", 1e-3, 5e-2, log=True)
        beta = trial.suggest_float("beta", 0.85, 0.99)
        threshold = trial.suggest_float("threshold", 0.5, 1.5)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        result = run_training(
            epochs=args.optuna_epochs,
            batch_size=batch_size,
            timesteps=args.timesteps,
            lr=lr,
            beta=beta,
            threshold=threshold,
            grad_clip=args.grad_clip,
            deferred=args.deferred,
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
        grad_clip=args.grad_clip,
        deferred=args.deferred,
        seed=args.seed,
        device=device,
        hpc_prints=args.hpc_prints,
    )
    print(f"\n[Done] final_test_acc={result['final_test_acc']:.4f} best_test_acc={result['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()
