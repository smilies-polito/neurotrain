#!/usr/bin/env python3
"""
Unit-test: DVSGEST_VGG9 + ESDRTRLTrainer on DVS Gesture.

Dataset:  DVS Gesture, 11 classes, 2×128×128 event frames.
Network:  Spiking VGG-9 (DVSGEST_VGG9).
Trainer:  ES-D-RTRL for classification.
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
import torch.nn as nn

# -----------------------------------------------------------------------------
# Hardcoded Defaults
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
TIMESTEPS = 10
NUM_WORKERS = 4
DATA_ROOT = ""

BETA = 0.53
THRESHOLD = 1.0

EPOCHS = 10
LR = 2e-4
SEED = 42
DEVICE = "auto"
HPC_PRINTS = False

ETRACE_DECAY = 0.9
GAMMA = 0.3

OPTUNA_TRIALS = 0
OPTUNA_EPOCHS = 20
STUDY_NAME = "esdrtrl_dvsgesture_vgg9_study"
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

from datasets.dvsgesture_loader import DVSGestureLoader
from trainers.es_d_rtrl_trainer import ESDRTRLTrainer

# Adjust this import to wherever you saved the uploaded VGG9 model file.
from networks.benchmarking.vgg9_dvsgest import DVSGEST_VGG9


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------
class DVSGESTVGG9ForESDRTRL(nn.Module):
    """
    Small adapter so the trainer can:
    - detect the real VGG9 model through .core
    - access n_classes
    - reset network state
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 11,
        beta: float = 0.53,
        threshold: float = 1.0,
        verbose: bool = False,
    ):
        super().__init__()
        self.core = DVSGEST_VGG9(
            in_channels=in_channels,
            num_classes=num_classes,
            beta=beta,
            threshold=threshold,
            verbose=verbose,
        )
        self.n_classes = num_classes
        self.in_shape = (in_channels, 128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spk_list, _ = self.core(x)
        return spk_list[-1]

    def init_states(self) -> None:
        self.core.init_states()

    def reset(self, device: torch.device | None = None) -> None:
        # device arg accepted for trainer compatibility
        self.core.reset()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DVS Gesture + DVSGEST_VGG9 ES-D-RTRL test.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--etrace-decay", type=float, default=ETRACE_DECAY)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    p.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS)
    p.add_argument("--optuna-epochs", type=int, default=OPTUNA_EPOCHS)
    p.add_argument("--study-name", type=str, default=STUDY_NAME)
    p.add_argument("--optuna-storage", type=str, default=OPTUNA_STORAGE)
    p.add_argument("--hpc-prints", dest="hpc_prints", action="store_true", default=HPC_PRINTS)
    p.add_argument("--verbose-network", action="store_true", default=False)
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
    etrace_decay: float,
    gamma: float,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    hpc_prints: bool = False,
    verbose_network: bool = False,
    log_prefix: str = "",
    trial: "optuna.trial.Trial | None" = None,
) -> Dict[str, float]:
    set_seed(seed)

    train_loader, test_loader = DVSGestureLoader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
        data_root=DATA_ROOT or None,
    )

    network = DVSGESTVGG9ForESDRTRL(
        in_channels=2,
        num_classes=11,
        beta=beta,
        threshold=threshold,
        verbose=verbose_network,
    ).to(device)

    trainer = ESDRTRLTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        etrace_decay=etrace_decay,
        gamma=gamma,
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
                print(
                    f"\r  [{'#' * f}{'-' * (28 - f)}] {int(100 * i / n_batches):3d}%",
                    end="",
                    flush=True,
                )

        if not hpc_prints:
            print("\r" + " " * 48 + "\r", end="", flush=True)

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
                    z_t, v_t, vo = trainer._forward_step(frame, vo)
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

        if trial is not None:
            trial.report(test_acc, step=epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

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
        beta = trial.suggest_float("beta", 0.45, 0.70)
        threshold = trial.suggest_float("threshold", 0.5, 2.0)
        etrace_decay = trial.suggest_float("etrace_decay", 0.7, 0.99)
        gamma = trial.suggest_float("gamma", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        result = run_training(
            epochs=args.optuna_epochs,
            batch_size=batch_size,
            timesteps=args.timesteps,
            lr=lr,
            beta=beta,
            threshold=threshold,
            etrace_decay=etrace_decay,
            gamma=gamma,
            seed=args.seed + trial.number,
            device=device,
            hpc_prints=args.hpc_prints,
            verbose_network=False,
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
        etrace_decay=args.etrace_decay,
        gamma=args.gamma,
        seed=args.seed,
        device=device,
        hpc_prints=args.hpc_prints,
        verbose_network=args.verbose_network,
    )
    print(
        f"\n[Done] final_test_acc={result['final_test_acc']:.4f} "
        f"best_test_acc={result['best_test_acc']:.4f}"
    )


if __name__ == "__main__":
    main()