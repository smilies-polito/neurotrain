#!/usr/bin/env python3
"""
Unit-test: RSNN + EpropTrainer on SVHN.

Dataset:  SVHN (Street View House Numbers), 10 classes, 3×32×32 images.
Network:  Single-layer recurrent SNN (RSNN) with snnTorch RLeaky.
Trainer:  E-prop (Bellec et al. 2020) for classification (Eq. 29).
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
BATCH_SIZE = 64
TIMESTEPS = 10
NUM_WORKERS = 4
DATA_ROOT = ""

# Network defaults
BETA = 0.95
THRESHOLD = 1.0
HIDDEN_SIZE = 100

# General training defaults
EPOCHS = 10
LR = 2e-4
SEED = 42
DEVICE = "auto"
HPC_PRINTS = False

# Trainer e-prop defaults
EPROP_GAMMA = 0.3
EPROP_TAU_MEM = 20.0
EPROP_TAU_OUT = 30.0
EPROP_THRESHOLD = 0.03

# Optuna defaults
OPTUNA_TRIALS = 0
OPTUNA_EPOCHS = 20
STUDY_NAME = "optuna_study"
OPTUNA_STORAGE = ""

# -----------------------------------------------------------------------------
# Minimal repo bootstrap
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

from datasets.svhn_loader import SVHNLoader
from networks.r_snn import RSNN
from trainers.eprop_trainer import EpropTrainer

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVHN + RSNN e-prop test.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--eprop-gamma", type=float, default=EPROP_GAMMA)
    p.add_argument("--eprop-tau-mem", type=float, default=EPROP_TAU_MEM)
    p.add_argument("--eprop-tau-out", type=float, default=EPROP_TAU_OUT)
    p.add_argument("--eprop-threshold", type=float, default=EPROP_THRESHOLD)
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
# Training
# -----------------------------------------------------------------------------
def run_training(
    *,
    batch_size: int,
    timesteps: int,
    threshold: float,
    beta: float,
    eprop_gamma: float,
    eprop_tau_mem: float,
    eprop_tau_out: float,
    eprop_threshold: float,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    hpc_prints: bool = False,
    log_prefix: str = "",
    trial: "optuna.trial.Trial | None" = None,
) -> Dict[str, float]:
    set_seed(seed)

    # ── Dataset: SVHN, 3×32×32, 10 classes ──
    train_loader, test_loader = SVHNLoader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
    )

    # ── Network: in_shape must match SVHN ──
    network = RSNN(
        in_shape=(3, 32, 32),   # SVHN: 3 channels, 32×32 → flattened = 3072
        num_classes=10,
        hidden_sizes=(HIDDEN_SIZE,),
        beta=beta,
        threshold=threshold,
    ).to(device)

    # ── Trainer: e-prop parameters from paper ──
    trainer = EpropTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        gamma=eprop_gamma,
        tau_mem=eprop_tau_mem,
        tau_out=eprop_tau_out,
        threshold=eprop_threshold,
        use_optimizer=True,
    ).to(device)

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_monitor_loss = 0.0
    non_blocking = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # ── TRAIN ──
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

        # ── EVAL ──
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
                    z_t, v_t, vo = trainer._eprop_step(frame, vo)
                    vo_sum = vo.clone() if vo_sum is None else (vo_sum + vo)
                preds = vo_sum.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += target.size(0)

        test_acc = correct / total if total > 0 else 0.0

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



# -----------------------------------------------------------------------------
# Optuna
# -----------------------------------------------------------------------------
def run_optuna(args: argparse.Namespace, device: torch.device) -> None:
    try:
        import optuna
    except ImportError as err:
        raise RuntimeError("Optuna is not installed.") from err

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
        beta = trial.suggest_float("beta", 0.90, 0.99)
        threshold = trial.suggest_float("threshold", 0.5, 1.5)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        result = run_training(
            epochs=args.optuna_epochs,
            batch_size=batch_size,
            timesteps=args.timesteps,
            lr=lr,
            beta=beta,
            threshold=threshold,
            eprop_gamma=args.eprop_gamma,
            eprop_tau_mem=args.eprop_tau_mem,
            eprop_tau_out=args.eprop_tau_out,
            eprop_threshold=args.eprop_threshold,
            seed=args.seed + trial.number,
            device=device,
            hpc_prints=args.hpc_prints,
            log_prefix=f"[trial {trial.number}] ",
            trial=trial,
        )
        trial.set_user_attr("final_test_acc", result["final_test_acc"])
        return result["best_test_acc"]

    print(f"[Optuna] trials={args.optuna_trials} epochs_per_trial={args.optuna_epochs}")
    study.optimize(objective, n_trials=args.optuna_trials)
    print(f"\n[Optuna] Best: value={study.best_value:.4f} params={study.best_params}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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
        eprop_gamma=args.eprop_gamma,
        eprop_tau_mem=args.eprop_tau_mem,
        eprop_tau_out=args.eprop_tau_out,
        eprop_threshold=args.eprop_threshold,
        seed=args.seed,
        device=device,
        hpc_prints=args.hpc_prints,
    )
    print(f"\n[Done] final_test_acc={result['final_test_acc']:.4f} best_test_acc={result['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()