#!/usr/bin/env python3
"""
Unit-test for ConvSNN + DECOLLETrainer on MNIST.

Tests the DECOLLE online local-learning algorithm on the MNIST digit
classification task using a convolutional spiking network (ConvSNN).

Usage:
    python test_decolle_conv_snn_mnist.py
    python test_decolle_conv_snn_mnist.py --epochs 2
    python test_decolle_conv_snn_mnist.py --optuna-trials 30 --optuna-epochs 15
    python test_decolle_conv_snn_mnist.py --epochs 20 --hpc-prints
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

BATCH_SIZE = 64
TIMESTEPS = 10
NUM_WORKERS = 4
DATA_ROOT = ""

BETA = 0.95
THRESHOLD = 1.0

EPOCHS = 10
LR = 1e-4
SEED = 42
DEVICE = "auto"
HPC_PRINTS = False

G_SCALE = 0.5
BURN_IN = 0
SURROGATE = "boxcar"
SURROGATE_SCALE = 5.0
DELTA = 0.5
H_WITH_NOISE = False
OMEGA_STD = 0.5
LAMBDA_U_UPPER = 0.0
LAMBDA_U_LOWER = 0.0
LR_SCALE_PER_LAYER = False

OPTUNA_TRIALS = 0
OPTUNA_EPOCHS = 20
STUDY_NAME = "decolle_conv_mnist_study"
OPTUNA_STORAGE = ""

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

from datasets.mnist_loader import MNISTLoader
from networks.benchmarking.conv_snn import ConvSNN
from trainers.decolle_trainer import DECOLLETrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal MNIST+ConvSNN DECOLLE test.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)

    p.add_argument("--g-scale", type=float, default=G_SCALE)
    p.add_argument("--burn-in", type=int, default=BURN_IN)
    p.add_argument("--surrogate", choices=("sigmoid", "boxcar"), default=SURROGATE)
    p.add_argument("--surrogate-scale", type=float, default=SURROGATE_SCALE)
    p.add_argument("--delta", type=float, default=DELTA)
    p.add_argument("--h-with-noise", action="store_true", default=H_WITH_NOISE)
    p.add_argument("--omega-std", type=float, default=OMEGA_STD)
    p.add_argument("--lambda-u-upper", type=float, default=LAMBDA_U_UPPER)
    p.add_argument("--lambda-u-lower", type=float, default=LAMBDA_U_LOWER)
    p.add_argument("--lr-scale-per-layer", action="store_true", default=LR_SCALE_PER_LAYER)

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


def run_training(
    *,
    batch_size: int,
    timesteps: int,
    threshold: float,
    beta: float,
    g_scale: float,
    burn_in: int,
    surrogate: str,
    surrogate_scale: float,
    delta: float,
    h_with_noise: bool,
    omega_std: float,
    lambda_u_upper: float,
    lambda_u_lower: float,
    lr_scale_per_layer: bool,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    hpc_prints: bool = False,
    log_prefix: str = "",
    trial: "optuna.trial.Trial | None" = None,
) -> Dict[str, float]:
    set_seed(seed)

    train_loader, test_loader = MNISTLoader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
        data_root=DATA_ROOT or None,
    )

    network = ConvSNN(
        in_shape=(1, 28, 28),
        num_classes=10,
        beta=beta,
        threshold=threshold,
    ).to(device)

    trainer = DECOLLETrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        g_scale=g_scale,
        burn_in=burn_in,
        surrogate=surrogate,
        surrogate_scale=surrogate_scale,
        delta=delta,
        h_with_noise=h_with_noise,
        omega_std=omega_std,
        lambda_u_upper=lambda_u_upper,
        lambda_u_lower=lambda_u_lower,
        lr_scale_per_layer=lr_scale_per_layer,
    ).to(device)

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_monitor_loss = 0.0
    non_blocking = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        trainer.network.train()
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
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += bs_cur

            if not hpc_prints:
                f = int(28 * i / n_batches)
                print(f"\r  [{'#' * f}{'-' * (28 - f)}] {int(100 * i / n_batches):3d}%", end="", flush=True)

        if not hpc_prints:
            print("\r" + " " * 40 + "\r", end="", flush=True)

        monitor_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        trainer.network.eval()
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
        g_scale = trial.suggest_float("g_scale", 0.1, 1.0)
        burn_in = trial.suggest_int("burn_in", 0, max(0, args.timesteps // 2))
        delta = trial.suggest_float("delta", 0.1, 1.0)
        lambda_u_upper = trial.suggest_float("lambda_u_upper", 1e-8, 1e-2, log=True)
        lambda_u_lower = trial.suggest_float("lambda_u_lower", 1e-8, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        result = run_training(
            epochs=args.optuna_epochs,
            batch_size=batch_size,
            timesteps=args.timesteps,
            lr=lr,
            beta=beta,
            threshold=threshold,
            g_scale=g_scale,
            burn_in=burn_in,
            surrogate=args.surrogate,
            surrogate_scale=args.surrogate_scale,
            delta=delta,
            h_with_noise=args.h_with_noise,
            omega_std=args.omega_std,
            lambda_u_upper=lambda_u_upper,
            lambda_u_lower=lambda_u_lower,
            lr_scale_per_layer=args.lr_scale_per_layer,
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
        g_scale=args.g_scale,
        burn_in=args.burn_in,
        surrogate=args.surrogate,
        surrogate_scale=args.surrogate_scale,
        delta=args.delta,
        h_with_noise=args.h_with_noise,
        omega_std=args.omega_std,
        lambda_u_upper=args.lambda_u_upper,
        lambda_u_lower=args.lambda_u_lower,
        lr_scale_per_layer=args.lr_scale_per_layer,
        seed=args.seed,
        device=device,
        hpc_prints=args.hpc_prints,
    )
    print(f"\n[Done] final_test_acc={result['final_test_acc']:.4f} best_test_acc={result['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()