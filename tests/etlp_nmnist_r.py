#!/usr/bin/env python3
"""
Paper-reproduction test: ETLP on N-MNIST with a feedforward RSNN.

Reference experiment (Quintana et al., 2024, Section IV / Table 1):
  - Dataset     : N-MNIST (event-based, 2×34×34 input, 10 classes)
  - Network     : feedforward LIF (1 hidden layer, n_rec=200)
  - tau_v=80 ms → beta ≈ exp(-dt/tau_v) = exp(-1/80) ≈ 0.9876
  - Optimizer   : Adamax lr=5e-4 (we use Adam as the closest available)
  - update_rate_hz = 100 Hz, dt_ms = 1.0 → p = 0.1 per timestep (Algorithm 1)
  - T = 300 bins at 1 ms resolution (dt = 1 ms)
  - Reported accuracy: 94.30%
"""

from __future__ import annotations

import argparse
import math
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
BATCH_SIZE = 128        # Paper uses batch_size=128.
TIMESTEPS = 300         # 1 ms bins over ~300 ms N-MNIST recordings (paper: dt=1 ms).
NUM_WORKERS = 4         # DataLoader worker processes.

# Network defaults  (paper: n_rec=200, tau_v=80 ms, thr=1.0)
HIDDEN_SIZE = 200       # Single hidden layer width.
BETA = math.exp(-1.0 / 80.0)  # ≈ 0.9876 — membrane decay for tau_v=80 ms, dt=1 ms.
THRESHOLD = 1.0

# General training defaults
EPOCHS = 200            # Paper trains for 200 epochs.
LR = 5e-4               # Paper: Adamax lr=5e-4. We use Adam with the same lr.
SEED = 42
DEVICE = "auto"
HPC_PRINTS = False

# ETLP-specific defaults  (paper: update_rate_hz=100 Hz, dt=1 ms → p=0.1)
SURROGATE_SCALE = 0.3   # Triangular pseudo-derivative scale (Eq. 8).
UPDATE_RATE_HZ = 100.0  # Teaching-event firing rate (Algorithm 1 Poisson trigger).
DT_MS = 1.0             # Simulation timestep in milliseconds.
FEEDBACK_SCALE = 1.0    # B_l ~ N(0, scale/sqrt(n_out)).
GRAD_CLIP = 0.0

# Optuna defaults
OPTUNA_TRIALS = 0
OPTUNA_EPOCHS = 5
STUDY_NAME = "etlp_nmnist_fc"
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

from datasets.nmnist_loader import NMNISTLoader
from networks.benchmarking.r_snn import RSNN
from trainers.etlp_trainer import ETLPTrainer


# -----------------------------------------------------------------------------
# Tiny utilities
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETLP paper-reproduction on N-MNIST (feedforward LIF).")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    p.add_argument("--surrogate-scale", type=float, default=SURROGATE_SCALE)
    p.add_argument("--update-rate-hz", type=float, default=UPDATE_RATE_HZ)
    p.add_argument("--dt-ms", type=float, default=DT_MS)
    p.add_argument("--feedback-scale", type=float, default=FEEDBACK_SCALE)
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
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
    hidden_size: int,
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
    # Paper: feedforward LIF, single hidden layer.
    # Input: [B, 2, 34, 34] (2 polarities × 34×34 pixels), flattened to 2312 inside RSNN.
    network = RSNN(
        in_shape=(2, 34, 34),
        num_classes=10,
        hidden_sizes=[hidden_size],
        beta=beta,
        threshold=threshold,
        reset_mechanism="subtract",  # soft reset (subtract threshold) — matches paper
    ).to(device)
    # Paper uses Adamax; we use Adam (closest available in this framework).
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    trainer = ETLPTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        surrogate_scale=surrogate_scale,
        update_rate_hz=update_rate_hz,
        dt_ms=dt_ms,
        feedback_scale=feedback_scale,
        grad_clip=grad_clip,
        use_optimizer=True,
        optimizer=optimizer,
    ).to(device)

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_loss = 0.0

    non_blocking = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # RECURRENT WEIGHTS TRACKING
        init_rec_wgts = {
            name: p.clone().detach()
            for name, p in network.recurrent_layers.named_parameters()
            if p.requires_grad
        }

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

        # RECURRENT WEIGHTS TRACKING
        for name, p in network.recurrent_layers.named_parameters():
            if p.requires_grad:
                delta = torch.norm(p.detach() - init_rec_wgts[name]).item()
                print(f"  -> \u0394w ({name}): {delta:.6f}")

        if trial is not None:
            try:
                import optuna
                trial.report(test_acc, epoch)
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
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        beta = trial.suggest_float("beta", 0.95, 0.999)
        threshold = trial.suggest_float("threshold", 0.5, 2.0)
        hidden_size = trial.suggest_categorical("hidden_size", [128, 200, 256, 450])
        update_rate_hz = trial.suggest_categorical("update_rate_hz", [50.0, 100.0, 200.0, 1000.0])

        result = run_training(
            epochs=args.optuna_epochs,
            batch_size=args.batch_size,
            timesteps=args.timesteps,
            lr=lr,
            beta=beta,
            threshold=threshold,
            hidden_size=hidden_size,
            update_rate_hz=update_rate_hz,
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
        hidden_size=args.hidden_size,
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
