#!/usr/bin/env python3
"""
BPTT training test: CIFAR-10 + CIFAR10_VGG9.

Adapted from bptt_dvsgest_vgg9_tp.py.
Dataset: CIFAR-10 with rate-coded spike trains — shape [T, B, 3, 32, 32].
Network: CIFAR10_VGG9 (VGG-9 SNN, in_channels=3, num_classes=10).
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
BATCH_SIZE = 32         # Mini-batch size used for both training and evaluation.
TIMESTEPS = 20          # Number of rate-coded time steps per sample.
NUM_WORKERS = 8         # DataLoader worker processes.

BETA = 0.53             # LIF membrane decay.
THRESHOLD = 1.0         # Spiking threshold.

EPOCHS = 100            # Training epochs for the default non-Optuna run.
LR = 1e-3               # Adam learning rate.
SEED = 42               # Global random seed for Python, NumPy, and PyTorch.
DEVICE = "auto"         # Runtime device selection: auto, cpu, or cuda.
HPC_PRINTS = False      # If True, suppress per-batch progress bar updates.

OPTUNA_TRIALS = 0       # Number of Optuna trials; 0 disables hyperparameter search.
OPTUNA_EPOCHS = 20      # Epochs executed inside each Optuna trial.
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

from datasets.cifar10_loader import CIFAR10Loader
from networks.benchmarking.vgg9_cifar10 import CIFAR10_VGG9
from trainers.bptt_trainer import BPTTTrainer

# -----------------------------------------------------------------------------
# Tiny utilities
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal CIFAR-10+VGG9 BPTT test.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
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
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    hpc_prints: bool = False,
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
        beta=beta,
        threshold=threshold,
        verbose=True,
    ).to(device)
    trainer = BPTTTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        grad_clip=1.0,
    ).to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=epochs)

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
        layer_rate_sum = [0.0] * network._num_blocks
        n_eval_batches = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader, 1):
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                network.reset()
                spike_sum = None
                batch_layer_rates = [0.0] * network._num_blocks
                T = data.size(0)
                for t in range(T):
                    spk_rec, _ = network(data[t])
                    for li in range(network._num_blocks):
                        batch_layer_rates[li] += spk_rec[li].mean().item()
                    out_t = spk_rec[-1]
                    spike_sum = out_t if spike_sum is None else spike_sum + out_t
                for li in range(network._num_blocks):
                    layer_rate_sum[li] += batch_layer_rates[li] / T
                n_eval_batches += 1
                preds = spike_sum.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += target.size(0)

                if not hpc_prints:
                    f = int(28 * i / len(test_loader))
                    print(f"\r  [{'#' * f}{'-' * (28 - f)}] {int(100 * i / len(test_loader)):3d}% eval",
                          end="", flush=True)

        if not hpc_prints:
            print("\r" + " " * 45 + "\r", end="", flush=True)

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
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        beta = trial.suggest_float("beta", 0.40, 0.70)
        threshold = trial.suggest_float("threshold", 0.5, 2.0)
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
