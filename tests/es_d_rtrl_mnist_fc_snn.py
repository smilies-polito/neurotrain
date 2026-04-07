#!/usr/bin/env python3
"""
Unit-test for RSNN + ESDRTRLTrainer on MNIST.

Tests the ES-D-RTRL (BrainScale) online learning algorithm on the MNIST
digit classification task using a single-layer recurrent SNN (RSNN) with
snnTorch RLeaky neurons.

Usage:
    # Default training (10 epochs):
    python test_esdrtrl_rsnn_mnist.py

    # Quick sanity check:
    python test_esdrtrl_rsnn_mnist.py --epochs 2

    # Optuna hyperparameter search:
    python test_esdrtrl_rsnn_mnist.py --optuna-trials 30 --optuna-epochs 15

    # HPC mode (no progress bars):
    python test_esdrtrl_rsnn_mnist.py --epochs 20 --hpc-prints
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
TIMESTEPS = 10          # Number of rate-coding steps produced by the MNIST loader.
NUM_WORKERS = 4         # DataLoader worker processes for MNIST loading.
DATA_ROOT = ""          # Optional MNIST root override; empty string uses the loader default.

# Network defaults
BETA = 0.95             # Recurrent hidden-layer leak/decay.
THRESHOLD = 1.0         # Hidden spiking threshold.
HIDDEN_SIZE = 100       # Number of recurrent hidden units for the RSNN.

# Trainer defaults
# General training defaults
EPOCHS = 10             # Training epochs for the default non-Optuna run.
LR = 2e-4               # Adam optimizer learning rate.
SEED = 42               # Global random seed for Python, NumPy, and PyTorch.
DEVICE = "auto"         # Runtime device selection: auto, cpu, or cuda.
HPC_PRINTS = False      # If True, suppress per-batch progress bar updates.

# ES-D-RTRL specific defaults (Wang et al. 2025, Eqs. 7-8)
ETRACE_DECAY = 0.9      # Smoothing factor α for eligibility traces.
GAMMA = 0.3             # Surrogate gradient magnitude γ.
KAPPA = 0.9             # Output readout decay factor κ.

# Optuna defaults
OPTUNA_TRIALS = 0       # Number of Optuna trials; 0 disables hyperparameter search.
OPTUNA_EPOCHS = 20      # Epochs executed inside each Optuna trial.
STUDY_NAME = "esdrtrl_mnist_study"  # Optuna study name.
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

# [MODIFY] Import dataset, trainer and network
from datasets.mnist_loader import MNISTLoader
from networks.benchmarking.fc_snn import FCSNN
from trainers.es_d_rtrl_trainer import ESDRTRLTrainer

# -----------------------------------------------------------------------------
# Tiny utilities (inlined to keep this file self-contained)
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal MNIST+RSNN ES-D-RTRL test.")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    p.add_argument("--timesteps", type=int, default=TIMESTEPS, help="Number of MNIST rate-coding steps.")
    p.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    p.add_argument("--beta", type=float, default=BETA, help="Hidden recurrent decay.")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="Hidden firing threshold for the RSNN hidden neurons.")
    p.add_argument("--etrace-decay", type=float, default=ETRACE_DECAY, help="ES-D-RTRL smoothing factor α (Eqs. 7-8).")
    p.add_argument("--gamma", type=float, default=GAMMA, help="Surrogate gradient magnitude γ.")
    p.add_argument("--kappa", type=float, default=KAPPA, help="Output readout decay factor κ.")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE, help="Execution device.")
    p.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS, help="Number of Optuna trials (0 disables).")
    p.add_argument("--optuna-epochs", type=int, default=OPTUNA_EPOCHS, help="Epochs per Optuna trial.")
    p.add_argument("--study-name", type=str, default=STUDY_NAME, help="Optuna study name.")
    p.add_argument("--optuna-storage", type=str, default=OPTUNA_STORAGE, help="Optuna storage URL (empty=in-memory).")
    p.add_argument("--hpc-prints", dest="hpc_prints", action="store_true", default=HPC_PRINTS, help="Disable incremental batch progress prints.",)
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


def run_training(
    *,
    # Dataset parameters
    batch_size: int,
    timesteps: int,
    # Network parameters
    threshold: float,
    beta: float,
    # ES-D-RTRL trainer parameters
    etrace_decay: float,
    gamma: float,
    kappa: float,
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

    # [MODIFY] Initialize dataset, trainer and network
    train_loader, test_loader = MNISTLoader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
        data_root=DATA_ROOT or None,
    )
    network = FCSNN(
        in_shape=(1, 28, 28),
        num_classes=10,
        beta=beta,
        threshold=threshold,
    ).to(device)
    trainer = ESDRTRLTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        etrace_decay=etrace_decay,
        gamma=gamma,
        kappa=kappa,
        use_optimizer=True,
    ).to(device)

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_monitor_loss = 0.0

    non_blocking = device.type == "cuda"

    # Loop on epochs
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
                print(f"\r  [{'#' * f}{'-' * (28 - f)}] {int(100 * i / n_batches):3d}%", end="", flush=True)

        if not hpc_prints:
            print("\r" + " " * 40 + "\r", end="", flush=True)

        monitor_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        # [EVAL] one full pass on test batches
        trainer.network.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                data = trainer.normalize_sequence(data, timesteps=timesteps)
                trainer.reset(device=device)
                vo = torch.zeros(target.size(0), trainer.network.n_classes, device=device)
                vo_sum = None
                for t in range(data.size(0)):
                    frame = data[t]
                    z_t, v_t, vo = trainer._forward_step(frame, vo)
                    vo_sum = vo.clone() if vo_sum is None else (vo_sum + vo)
                preds = vo_sum.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += target.size(0)

        test_acc = correct / total if total > 0 else 0.0

        # [METRICS] track epoch outputs and best score
        final_train_monitor_loss = monitor_loss
        final_test_acc = test_acc
        best_test_acc = max(best_test_acc, test_acc)

        epoch_time_s = time.perf_counter() - epoch_start

        print(
            f"{log_prefix}epoch={epoch}/{epochs} "
            f"monitor_loss={monitor_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} "
            f"epoch_time_s={epoch_time_s:.2f}"
        )

    return {
        "best_test_acc": best_test_acc,
        "final_test_acc": final_test_acc,
        "final_train_monitor_loss": final_train_monitor_loss,
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
        beta = trial.suggest_float("beta", 0.85, 0.99)
        threshold = trial.suggest_float("threshold", 0.5, 2.0)
        etrace_decay = trial.suggest_float("etrace_decay", 0.7, 0.99)
        gamma = trial.suggest_float("gamma", 0.1, 0.5)
        kappa = trial.suggest_float("kappa", 0.8, 0.99)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        result = run_training(
            epochs=args.optuna_epochs,
            batch_size=batch_size,
            timesteps=args.timesteps,
            lr=lr,
            beta=beta,
            threshold=threshold,
            etrace_decay=etrace_decay,
            gamma=gamma,
            kappa=kappa,
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
        etrace_decay=args.etrace_decay,
        gamma=args.gamma,
        kappa=args.kappa,
        seed=args.seed,
        device=device,
        hpc_prints=args.hpc_prints,
    )
    print(f"\n[Done] final_test_acc={result['final_test_acc']:.4f} best_test_acc={result['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()