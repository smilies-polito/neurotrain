#!/usr/bin/env python3
"""
Integration test: Fashion-MNIST loader + OTTT_VGG9_FashionMNIST + OTTTTrainer.

This test verifies that the OTTT-SNN variant works correctly on Fashion-MNIST.
The network uses ScaledWSConv2d layers and Scale post-spike scaling, matching
the official OTTT-SNN architecture from (Xiao et al., NeurIPS 2022).

Architecture notes:
  OTTT_VGG9_FashionMNIST uses ScaledWSConv2d (Weight Standardization Conv) layers.
  These are nn.Conv2d subclasses, so OTTTTrainer's forward hooks fire on them
  correctly and trace-based gradient substitution is applied to all 8 conv blocks.

Eval note:
  The classifier is a plain Linear layer (no LIF), so we accumulate its output
  logits across all timesteps and take the argmax.
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

# Dataset defaults
BATCH_SIZE = 64
TIMESTEPS = 10
NUM_WORKERS = 4

# Network defaults
BETA = 0.5
THRESHOLD = 1.0

# Trainer defaults
EPOCHS = 10
LR = 1e-3
SEED = 42
DEVICE = "auto"
HPC_PRINTS = False

# Optuna defaults
OPTUNA_TRIALS = 0
OPTUNA_EPOCHS = 5
STUDY_NAME = "ottt_fashionmnist_vgg9_ottt"
OPTUNA_STORAGE = ""

# Minimal repo bootstrap
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

from datasets.fashionmnist_loader import FashionMNISTLoader
from networks.benchmarking.vgg9 import vgg9_ottt_fashionmnist as OTTT_VGG9_FashionMNIST
from trainers.ottt_trainer import OTTTTrainer


# Utilities
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fashion-MNIST + OTTT_VGG9_FashionMNIST test.")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    p.add_argument("--timesteps", type=int, default=TIMESTEPS, help="Rate-coded timesteps.")
    p.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    p.add_argument("--beta", type=float, default=BETA, help="LIF beta.")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="LIF threshold.")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE, help="Execution device.")
    p.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS, help="Number of Optuna trials.")
    p.add_argument("--optuna-epochs", type=int, default=OPTUNA_EPOCHS, help="Epochs per trial.")
    p.add_argument("--study-name", type=str, default=STUDY_NAME, help="Optuna study name.")
    p.add_argument("--optuna-storage", type=str, default=OPTUNA_STORAGE, help="Optuna storage URL.")
    p.add_argument("--hpc-prints", dest="hpc_prints", action="store_true", default=HPC_PRINTS,
                   help="Disable incremental batch progress prints.")
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


# Train / eval
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

    train_loader, test_loader = FashionMNISTLoader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
    )

    network = OTTT_VGG9_FashionMNIST(
        in_channels=1,
        num_classes=10,
        beta=beta,
        threshold=threshold,
        verbose=True,
    ).to(device)

    trainer = OTTTTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        online_updates=True,
    ).to(device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=epochs)

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_loss = 0.0

    non_blocking = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # [TRAIN] one full epoch
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

        # [EVAL] one full pass on test batches
        # Plain Linear head: accumulate logits across all timesteps
        network.eval()
        total = 0
        correct = 0
        layer_rate_sum = [0.0] * network._num_blocks
        n_eval_batches = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                network.reset()
                T = data.size(0)
                batch_layer_rates = [0.0] * network._num_blocks
                logit_sum = torch.zeros(target.size(0), 10, device=device)

                for t in range(T):
                    spk_rec, mem_rec = network(data[t])
                    # Accumulate logits from the Linear head
                    logit_sum += mem_rec[-1]
                    # Accumulate spike rates for conv blocks only (not the head)
                    for li in range(network._num_blocks):
                        batch_layer_rates[li] += spk_rec[li].mean().item()

                # Prediction from accumulated logits
                preds = logit_sum.argmax(dim=1)
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
