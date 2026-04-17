#!/usr/bin/env python3
"""
Test for ELLTrainer + FCSNN on CIFAR-10.

This script evaluates the current ELL-style local learning setup on the
CIFAR-10 classification task using an snnTorch fully connected SNN.

Notes:
    - The current ELLTrainer is primarily aligned with rate-coded
      classification and reuses data[0] across timesteps.
    - This script therefore tests the trainer "as is" on CIFAR-10,
      without changing trainer behavior.
    - For CIFAR-10, the main path uses FCSNN with input shape 3×32×32
      and 10 output classes.

Usage:
    python test_ell_fc_snn_cifar10.py
    python test_ell_fc_snn_cifar10.py --epochs 20 --hidden-sizes 512
    python test_ell_fc_snn_cifar10.py --optuna-trials 20 --optuna-epochs 8
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
BATCH_SIZE = 64
TIMESTEPS = 10
NUM_WORKERS = 4
DATA_ROOT = ""

HIDDEN_SIZES = "1024"
BETA = 0.95
THRESHOLD = 1.0

EPOCHS = 10
LR = 5e-4
LR_DECAY_FACTOR = 5.0
LR_DECAY_EVERY = 15
SEED = 42
DEVICE = "auto"
HPC_PRINTS = False

OPTUNA_TRIALS = 0
OPTUNA_EPOCHS = 10
STUDY_NAME = "ell_fc_cifar10_study"
OPTUNA_STORAGE = ""

# -----------------------------------------------------------------------------
# Repo bootstrap
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
from networks.fc_snn import FCSNN
from trainers.ell_trainer import ELLTrainer


# -----------------------------------------------------------------------------
# CLI / utils
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CIFAR-10 + FCSNN + ELL-style local learning test."
    )
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--lr-decay-factor", type=float, default=LR_DECAY_FACTOR)
    p.add_argument("--lr-decay-every", type=int, default=LR_DECAY_EVERY)

    p.add_argument(
        "--hidden-sizes", type=str, default=HIDDEN_SIZES,
        help="Comma-separated hidden layer sizes, e.g. '100' or '512,256'."
    )
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)

    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    p.add_argument("--hpc-prints", dest="hpc_prints", action="store_true",
                   default=HPC_PRINTS)

    p.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS)
    p.add_argument("--optuna-epochs", type=int, default=OPTUNA_EPOCHS)
    p.add_argument("--study-name", type=str, default=STUDY_NAME)
    p.add_argument("--optuna-storage", type=str, default=OPTUNA_STORAGE)
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


def parse_hidden_sizes(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def run_training(
    *,
    batch_size: int,
    timesteps: int,
    hidden_sizes: List[int],
    beta: float,
    threshold: float,
    epochs: int,
    lr: float,
    lr_decay_factor: float,
    lr_decay_every: int,
    seed: int,
    device: torch.device,
    hpc_prints: bool = False,
    log_prefix: str = "",
    trial=None,
) -> Dict[str, float]:
    set_seed(seed)

    train_loader, test_loader = CIFAR10Loader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
        data_root=DATA_ROOT or None,
    )

    network = FCSNN(
        in_shape=(3, 32, 32),
        num_classes=10,
        hidden_sizes=hidden_sizes,
        beta=beta,
        threshold=threshold,
    ).to(device)

    trainer = ELLTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
    ).to(device)

    schedulers = [
        torch.optim.lr_scheduler.StepLR(
            opt, step_size=lr_decay_every, gamma=1.0 / lr_decay_factor
        )
        for opt in trainer.optimizers
    ]

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_loss = 0.0
    non_blocking = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        network.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = len(train_loader)

        for i, (data, target) in enumerate(train_loader, 1):
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)

            loss, pred = trainer.train_sample(data, target)

            bs_cur = target.size(0)
            total_loss += loss.item() * bs_cur
            total_correct += pred.eq(target).sum().item()
            total_samples += bs_cur

            if not hpc_prints:
                f = int(28 * i / n_batches)
                print(
                    f"\r  [{'#' * f}{'-' * (28 - f)}] {int(100 * i / n_batches):3d}%",
                    end="", flush=True,
                )

        if not hpc_prints:
            print("\r" + " " * 40 + "\r", end="", flush=True)

        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        for sched in schedulers:
            sched.step()

        network.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)

                pred = trainer.predict(data)
                correct += pred.squeeze(1).eq(target).sum().item()
                total += target.size(0)

        test_acc = correct / total if total > 0 else 0.0
        final_train_loss = train_loss
        final_test_acc = test_acc
        best_test_acc = max(best_test_acc, test_acc)

        epoch_time_s = time.perf_counter() - epoch_start
        current_lr = schedulers[0].get_last_lr()[0]
        print(
            f"{log_prefix}epoch={epoch}/{epochs} "
            f"loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} "
            f"lr={current_lr:.2e} "
            f"time={epoch_time_s:.2f}s"
        )

        if trial is not None:
            import optuna
            trial.report(test_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return {
        "best_test_acc": best_test_acc,
        "final_test_acc": final_test_acc,
        "final_train_loss": final_train_loss,
    }


# -----------------------------------------------------------------------------
# Optuna
# -----------------------------------------------------------------------------
def run_optuna(args: argparse.Namespace, device: torch.device) -> None:
    try:
        import optuna
    except ImportError as err:
        raise RuntimeError(
            "Optuna is not installed. Install it with `pip install optuna`."
        ) from err

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
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        lr_decay_factor = trial.suggest_float("lr_decay_factor", 2.0, 10.0)
        lr_decay_every = trial.suggest_int("lr_decay_every", 5, 25)
        hidden = trial.suggest_categorical("hidden_size", [100, 256, 512, 1024])

        result = run_training(
            epochs=args.optuna_epochs,
            batch_size=batch_size,
            timesteps=args.timesteps,
            hidden_sizes=[hidden],
            beta=beta,
            threshold=threshold,
            lr=lr,
            lr_decay_factor=lr_decay_factor,
            lr_decay_every=lr_decay_every,
            seed=args.seed + trial.number,
            device=device,
            hpc_prints=args.hpc_prints,
            log_prefix=f"[trial {trial.number}] ",
            trial=trial,
        )
        trial.set_user_attr("final_test_acc", result["final_test_acc"])
        return result["best_test_acc"]

    print(
        f"[Optuna] trials={args.optuna_trials} "
        f"epochs_per_trial={args.optuna_epochs} "
        f"study={args.study_name}"
    )
    study.optimize(objective, n_trials=args.optuna_trials)

    print("\n[Optuna] Best trial")
    print(f"  value={study.best_value:.4f}")
    print(f"  params={study.best_params}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    print(f"[ELL FC CIFAR10 test] device={device.type}")
    print(f"[ELL FC CIFAR10 test] hidden_sizes={hidden_sizes} beta={args.beta} threshold={args.threshold}")
    print(f"[ELL FC CIFAR10 test] lr={args.lr} decay=÷{args.lr_decay_factor} every {args.lr_decay_every} ep")
    print(f"[ELL FC CIFAR10 test] timesteps={args.timesteps} batch_size={args.batch_size} epochs={args.epochs}")

    if args.optuna_trials > 0:
        run_optuna(args, device)
        return

    result = run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        hidden_sizes=hidden_sizes,
        beta=args.beta,
        threshold=args.threshold,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_every=args.lr_decay_every,
        seed=args.seed,
        device=device,
        hpc_prints=args.hpc_prints,
    )

    print(
        f"\n[Done] final_test_acc={result['final_test_acc']:.4f} "
        f"best_test_acc={result['best_test_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
