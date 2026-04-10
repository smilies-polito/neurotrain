#!/usr/bin/env python3
"""
Validation test: TP trainer on N-MNIST with RSNN.

Pes et al. (2026) - "Traces propagation: memory-efficient and scalable
forward-only learning in spiking neural networks"

Hyperparameters follow Table 4 (N-MNIST: α=0.98, β=0.98, vth=1.0, batch=128).
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
BATCH_SIZE = 128
TIMESTEPS = 300
NUM_WORKERS = 8

BETA = 0.98         # snntorch membrane decay (for eval dynamics)
THRESHOLD = 1.0     # snntorch spike threshold (for eval dynamics)
HIDDEN_SIZE = 200   # Paper: single hidden layer of 200 neurons for N-MNIST

EPOCHS = 100
LR = 1e-4
SEED = 42
DEVICE = "auto"
HPC_PRINTS = False

# TP-specific (Table 4, N-MNIST)
ALPHA = 0.98            # membrane decay α (Eq 1)
BETA_TRACE = 0.98       # eligibility trace decay β (Eq 11-12)
VTH = 1.0               # spike threshold
SURROGATE_SCALE = 1.0   # ArcTan surrogate scale

OPTUNA_TRIALS = 0
OPTUNA_EPOCHS = 20
STUDY_NAME = "tp_nmnist_r"
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

from datasets.nmnist_loader import NMNISTLoader
from networks.benchmarking.r_snn import RSNN
from trainers.tp_trainer import TPTrainer


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TP trainer on N-MNIST with RSNN.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--beta-trace", type=float, default=BETA_TRACE)
    p.add_argument("--vth", type=float, default=VTH)
    p.add_argument("--surrogate-scale", type=float, default=SURROGATE_SCALE)
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
    beta: float,
    threshold: float,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    alpha: float = ALPHA,
    beta_trace: float = BETA_TRACE,
    vth: float = VTH,
    surrogate_scale: float = SURROGATE_SCALE,
    hpc_prints: bool = False,
    log_prefix: str = "",
    trial=None,
) -> Dict[str, float]:
    set_seed(seed)

    train_loader, test_loader = NMNISTLoader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
    )
    network = RSNN(
        in_shape=(2, 34, 34),
        num_classes=10,
        hidden_sizes=[HIDDEN_SIZE],
        beta=beta,
        threshold=threshold,
        reset_mechanism="subtract",
        out_integrator=True,   # TP requires LI at output (paper Sec 3.1)
    ).to(device)
    trainer = TPTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        alpha=alpha,
        beta=beta_trace,
        vth=vth,
        surrogate_scale=surrogate_scale,
        train_target_propagator=True,
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

        # [EVAL] — use mem_rec[-1] at the final timestep (TP Sec 3.1).
        # With out_integrator=True: lif_out has beta=1.0 / threshold=1e9 — never fires.
        # mem_rec[-1] after T steps = Σ_t W_out·spk_hidden^t = training integrator.
        # Do NOT accumulate over timesteps; the integrator accumulates internally.
        network.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader, 1):
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                network.reset()
                mem_final = None
                for t in range(data.size(0)):
                    _, mem_rec = network(data[t])
                    mem_final = mem_rec[-1]   # overwrite each step; LI accumulates internally
                preds = mem_final.argmax(dim=1)
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
        raise RuntimeError("Install optuna: pip install optuna") from err

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
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        alpha = trial.suggest_float("alpha", 0.9, 0.99)
        beta_trace = trial.suggest_float("beta_trace", 0.9, 0.99)
        vth = trial.suggest_float("vth", 0.5, 1.5)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        result = run_training(
            epochs=args.optuna_epochs,
            batch_size=batch_size,
            timesteps=args.timesteps,
            lr=lr,
            beta=args.beta,
            threshold=args.threshold,
            seed=args.seed + trial.number,
            device=device,
            alpha=alpha,
            beta_trace=beta_trace,
            vth=vth,
            hpc_prints=args.hpc_prints,
            log_prefix=f"[trial {trial.number}] ",
            trial=trial,
        )
        trial.set_user_attr("final_test_acc", result["final_test_acc"])
        return result["best_test_acc"]

    print(f"[Optuna] trials={args.optuna_trials} epochs_per_trial={args.optuna_epochs}")
    study.optimize(objective, n_trials=args.optuna_trials)
    print(f"\n[Optuna] Best trial: value={study.best_value:.4f} params={study.best_params}")


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
        alpha=args.alpha,
        beta_trace=args.beta_trace,
        vth=args.vth,
        surrogate_scale=args.surrogate_scale,
        hpc_prints=args.hpc_prints,
    )
    print(f"\n[Done] final_test_acc={result['final_test_acc']:.4f} best_test_acc={result['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()
