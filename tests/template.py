#!/usr/bin/env python3
"""
Template for unit-test of NETWORK and TRAINER.
To adapt to a new network and trainer:
- [ ] Modify the imports to include the needed dataset, network, and trainer classes.
- [ ] Adjust the run_training function to initialize the dataset, network, and trainer.
- [ ] Hardcode parameters at the top.
- [ ] (Optional) If using Optuna se the ragnes to the parameters you are interested in tuning.
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
TIMESTEPS = 100         # Number of temporal bins produced by the SHD loader.
NUM_WORKERS = 4         # DataLoader worker processes for SHD loading.
DATA_ROOT = ""          # Optional SHD root override; empty string uses the loader default.

# Network defaults
BETA = 0.95             # Recurrent hidden-layer leak/decay.
THRESHOLD = 1.0         # Hidden spiking threshold.
# Network specific defaults
# eg. HIDDEN_SIZE = 450       # Paper-style recurrent hidden layer width

# Trainer defaults
# General training defaults
EPOCHS = 10             # Training epochs for the default non-Optuna run.
LR = 2e-4               # OSTTP optimizer learning rate.
SEED = 42               # Global random seed for Python, NumPy, and PyTorch.
DEVICE = "auto"         # Runtime device selection: auto, cpu, or cuda.
HPC_PRINTS = False      # If True, suppress per-batch progress bar updates.
# [MODIFY] Import dataset, trainer and network ##################################################################
# eg. PSEUDO_DERIVATIVE = "fast_sigmoid"  # Surrogate used inside OSTTP eligibility updates.

# Optuna defaults
OPTUNA_TRIALS = 0       # Number of Optuna trials; 0 disables hyperparameter search.
OPTUNA_EPOCHS = 20      # Epochs executed inside each Optuna trial.
STUDY_NAME = "optuna_study"  # Optuna study name.
OPTUNA_STORAGE = ""     # Optuna storage URL; empty string keeps the study in memory.

# -----------------------------------------------------------------------------
# Minimal repo bootstrap: make imports work when running from tests/
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"

# Ensure src/ and tests/ are importable.
# (tests/ first so local test helpers would win if they exist)
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# Work around src/networks/__init__.py eager imports by creating a lightweight
# namespace package so we can import only what we need.
if "networks" not in sys.modules:
    networks_pkg = types.ModuleType("networks")
    networks_pkg.__path__ = [str(SRC_DIR / "networks")]
    sys.modules["networks"] = networks_pkg

# [MODIFY] Import dataset, trainer and network ################################################################## 
# from datasets.shd_loader import SHDLoader
# from networks.reproducibility.osttp_shd_rec import OSTTPSHDRec
# from trainers.osttp_trainer import OSTTPTrainer

# -----------------------------------------------------------------------------
# Tiny utilities (inlined to keep this file self-contained)
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal SHD+OSTTP reproduction test.")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    p.add_argument("--timesteps", type=int, default=TIMESTEPS, help="Number of SHD time bins.")
    p.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    p.add_argument("--beta", type=float, default=BETA, help="Hidden recurrent decay.")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="Hidden firing threshold.")
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
    # General training parameters
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    hpc_prints: bool = False,
    # Training specific parameters
    # eg. pseudo_derivative: str,
    # Optuna parameters
    log_prefix: str = "",
    trial: "optuna.trial.Trial | None" = None,
) -> Dict[str, float]:
    set_seed(seed)

    # [MODIFY] Initialize dataset, trainer and network ##################################################################
    # train_loader, test_loader = SHDLoader(
    #     batch_size=batch_size,
    #     T=timesteps,
    #     pin_memory=(device.type == "cuda"),
    #     seed=seed,
    #     num_workers=NUM_WORKERS,
    # )
    # network = OSTTPSHDRec(
    #     hidden_size=HIDDEN_SIZE,
    #     beta=beta,
    #     output_beta=OUTPUT_BETA,
    #     threshold=threshold,
    # ).to(device)
    # trainer = OSTTPTrainer(
    #     network=network,
    #     lr=lr,
    #     batch_size=batch_size,
    #     pseudo_derivative=PSEUDO_DERIVATIVE,
    #     output_loss="ce",
    #     output_readout="mem",
    #     feedback_scale=FEEDBACK_SCALE,
    #     feedback_seed=FEEDBACK_SEED,
    #     use_optimizer=True,
    # ).to(device)

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_loss = 0.0

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
                print(f"\r  [{'#' * f}{'-' * (28 - f)}] {int(100 * i / n_batches):3d}%  ", end="", flush=True)

        if not hpc_prints:
            print("\r" + " " * 40 + "\r", end="", flush=True)

        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        # [EVAL] one full pass on test batches
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

        # [METRICS] track epoch outputs and best score
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


def run_optuna(args: argparse.Namespace, device: torch.device) -> None:
    try:
        import optuna
    except ImportError as err:
        raise RuntimeError("Optuna is not installed. Install it with `pip install optuna`.") from err

    storage = args.optuna_storage or None
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",               # We want to maximize test accuracy.
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
            seed=args.seed + trial.number,
            device=device,
            hpc_prints=args.hpc_prints,
            log_prefix=f"[trial {trial.number}] ",
            trial=trial,
        )
        trial.set_user_attr("final_test_acc", result["final_test_acc"])
        # The objective for our exploration is the best test accuracy
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
