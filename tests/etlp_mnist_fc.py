#!/usr/bin/env python3
"""
Validation test: ETLP trainer on MNIST with FCSNN.

Quintana et al., "ETLP: Event-based Three-factor Local Plasticity for
Online Learning with Neuromorphic Hardware", NCE 2024.
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
TIMESTEPS = 25
NUM_WORKERS = 8

# Network defaults
BETA = 0.9
THRESHOLD = 1.0
HIDDEN_SIZES = [256, 256]

# General training defaults
EPOCHS = 100
LR = 1e-3               # ETLP with Adam (use_optimizer=True) or direct SGD
SEED = 42
DEVICE = "auto"
HPC_PRINTS = False

# ETLP-specific defaults
SURROGATE_SCALE = 0.3
UPDATE_RATE_HZ = 1000.0  # p = 1.0 with dt=1ms → deterministic per-step updates
DT_MS = 1.0
FEEDBACK_SCALE = 1.0
GRAD_CLIP = 0.0

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

from datasets.mnist_loader import MNISTLoader
from networks.fc_snn import FCSNN
from trainers.etlp_trainer import ETLPTrainer


# -----------------------------------------------------------------------------
# Tiny utilities (inlined to keep this file self-contained)
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal MNIST+FCSNN ETLP test.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    p.add_argument("--surrogate-scale", type=float, default=SURROGATE_SCALE)
    p.add_argument("--update-rate-hz", type=float, default=UPDATE_RATE_HZ)
    p.add_argument("--dt-ms", type=float, default=DT_MS)
    p.add_argument("--feedback-scale", type=float, default=FEEDBACK_SCALE)
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
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
    seed: int,
    device: torch.device,
    surrogate_scale: float = SURROGATE_SCALE,
    update_rate_hz: float = UPDATE_RATE_HZ,
    dt_ms: float = DT_MS,
    feedback_scale: float = FEEDBACK_SCALE,
    grad_clip: float = GRAD_CLIP,
    hpc_prints: bool = False,
    log_prefix: str = "",
) -> Dict[str, float]:
    set_seed(seed)

    train_loader, test_loader = MNISTLoader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
    )
    network = FCSNN(
        in_shape=(1, 28, 28),
        num_classes=10,
        beta=beta,
        threshold=threshold,
        reset_mechanism="subtract",
        hidden_sizes=HIDDEN_SIZES,
    ).to(device)
    trainer = ETLPTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        surrogate_scale=surrogate_scale,
        update_rate_hz=update_rate_hz,
        dt_ms=dt_ms,
        feedback_scale=feedback_scale,
        grad_clip=grad_clip,
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

    return {
        "best_test_acc": best_test_acc,
        "final_test_acc": final_test_acc,
        "final_train_loss": final_train_loss,
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"[Run] device={device.type}")

    result = run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        lr=args.lr,
        beta=args.beta,
        threshold=args.threshold,
        seed=args.seed,
        device=device,
        surrogate_scale=args.surrogate_scale,
        update_rate_hz=args.update_rate_hz,
        dt_ms=args.dt_ms,
        feedback_scale=args.feedback_scale,
        grad_clip=args.grad_clip,
        hpc_prints=args.hpc_prints,
    )
    print(f"\n[Done] final_test_acc={result['final_test_acc']:.4f} best_test_acc={result['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()
