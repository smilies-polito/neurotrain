#!/usr/bin/env python3
"""
Integration test: CIFAR-10 static loader + OTTT_VGG9_CIFAR10 + OTTTTrainer.

This test uses the OTTT-SNN static-image training recipe:
  - Images are NOT rate-coded. Each batch is [B, 3, 32, 32].
  - The same frame is fed to the network at every timestep (OTTT-SNN's approach).
  - The trainer loops T times over the same input, relying on stateful LIF neurons
    to generate temporal dynamics rather than pre-encoded spike trains.

How the static frame reaches the trainer:
  - Each batch [B, 3, 32, 32] is expanded to [T, B, 3, 32, 32] (T copies of the same frame).
  - network.constant_input_per_timestep = True signals OTTTTrainer to:
      (a) use data.mean(dim=0) as the per-step input (which equals the original frame
          when all T slices are identical), and
      (b) enable grad clipping (0.2) and NaN/Inf sanitization — stable-training
          settings recommended by the OTTT paper for static inputs.

Differences from ottt_cifar10_vgg9_ottt.py (rate-coded version):
  - Imports CIFAR10StaticLoader instead of CIFAR10Loader.
  - Batches are [B, 3, 32, 32] not [T, B, 3, 32, 32].
  - Frame is expanded to [T, B, 3, 32, 32] before passing to trainer.
  - network.constant_input_per_timestep = True activates the static-input recipe.
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
TIMESTEPS = 6       # OTTT-SNN paper uses T=6 for static CIFAR
NUM_WORKERS = 4

# Network defaults
BETA = 0.5
THRESHOLD = 1.0

# Trainer defaults
EPOCHS = 10
LR = 1e-1           # OTTT-SNN paper uses SGD with lr=0.1 for static CIFAR
SEED = 42
DEVICE = "auto"
HPC_PRINTS = False

# Optuna defaults
OPTUNA_TRIALS = 0
OPTUNA_EPOCHS = 5
STUDY_NAME = "ottt_cifar10_vgg9_ottt_static"
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

from datasets.cifar10_static_loader import CIFAR10StaticLoader
from networks.benchmarking.vgg9 import vgg9_ottt_cifar10 as OTTT_VGG9_CIFAR10
from trainers.ottt_trainer import OTTTTrainer


# Utilities
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10 static + OTTT_VGG9_CIFAR10 test.")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    p.add_argument("--timesteps", type=int, default=TIMESTEPS, help="Timesteps per sample.")
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

    train_loader, test_loader = CIFAR10StaticLoader(
        batch_size=batch_size,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
    )

    network = OTTT_VGG9_CIFAR10(
        in_channels=3,
        num_classes=10,
        beta=beta,
        threshold=threshold,
        verbose=True,
    ).to(device)

    # Flag the network as static-input so OTTTTrainer auto-enables:
    #   - loss_lambda = 0.05  (CE + MSE mix from the OTTT-SNN paper for static CIFAR)
    #   - grad_clip = 0.2     (element-wise gradient clipping for stability)
    #   - sanitize_grads = True (replace NaN/Inf gradients before optimizer step)
    network.constant_input_per_timestep = True

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
            data = data.to(device, non_blocking=non_blocking)    # [B, 3, 32, 32]
            target = target.to(device, non_blocking=non_blocking)

            # Expand to [T, B, 3, 32, 32]: T identical copies of the same static frame.
            # OTTTTrainer reads data.shape[0]=T and, since constant_input_per_timestep=True,
            # uses data.mean(dim=0)=[B, 3, 32, 32] as the per-step input — equal to the
            # original frame when all T slices are identical.
            data_expanded = data.unsqueeze(0).expand(timesteps, -1, -1, -1, -1)

            loss, pred = trainer.train_sample(data_expanded, target)
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
        # Same-frame-per-timestep: loop T times over the static frame, accumulate logits.
        network.eval()
        total = 0
        correct = 0
        layer_rate_sum = [0.0] * network._num_blocks
        n_eval_batches = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device, non_blocking=non_blocking)    # [B, 3, 32, 32]
                target = target.to(device, non_blocking=non_blocking)
                network.reset()

                batch_layer_rates = [0.0] * network._num_blocks
                logit_sum = torch.zeros(target.size(0), 10, device=device)

                # Feed the same frame T times; LIF membrane state evolves across steps.
                for t in range(timesteps):
                    spk_rec, mem_rec = network(data)
                    logit_sum += mem_rec[-1]
                    for li in range(network._num_blocks):
                        batch_layer_rates[li] += spk_rec[li].mean().item()

                preds = logit_sum.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += target.size(0)
                for li in range(network._num_blocks):
                    layer_rate_sum[li] += batch_layer_rates[li] / timesteps
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
        lr = trial.suggest_float("lr", 1e-2, 5e-1, log=True)
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
