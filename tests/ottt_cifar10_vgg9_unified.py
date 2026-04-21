#!/usr/bin/env python3
"""
Validation test: OTTT trainer on CIFAR-10 with the UNIFIED VGG9 factory.

Counterpart: ottt_cifar10_vgg9_legacy.py uses the old OTTT_VGG9_CIFAR10 per-dataset
class.  Running both with identical args should produce identical epoch metrics —
this validates that the unified class is a correct drop-in replacement.

Usage:
    python3 tests/ottt_cifar10_vgg9_legacy.py  --epochs 1 --seed 42 --hpc-prints
    python3 tests/ottt_cifar10_vgg9_unified.py --epochs 1 --seed 42 --hpc-prints
    # Expect: train_loss / train_acc / test_acc match to >=4 decimal places.

Dataset note:
  Uses CIFAR10StaticLoader (normalized static images). Each sample is expanded to
  [T, B, 3, 32, 32] by repeating the same frame T times, matching the OTTT paper.

Architecture note:
  vgg9_ottt_cifar10() has a plain Linear head (global_pool + head).  During eval,
  logits are accumulated over all T timesteps before argmax (same as legacy).
  constant_input_per_timestep=True enables the OTTT paper's static-input recipe:
  loss_lambda=0.05, grad_clip=0.2, sanitize_grads=True.
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
# Defaults  (identical to ottt_cifar10_vgg9_legacy.py)
# -----------------------------------------------------------------------------
BATCH_SIZE  = 64
TIMESTEPS   = 6        # OTTT paper uses T=6 for static CIFAR-10
NUM_WORKERS = 4

BETA      = 0.5
THRESHOLD = 1.0

EPOCHS = 10
LR     = 1e-1          # OTTT paper uses SGD lr=0.1 for static CIFAR-10
SEED   = 42
DEVICE = "auto"
HPC_PRINTS = False

OPTUNA_TRIALS  = 0
OPTUNA_EPOCHS  = 5
STUDY_NAME     = "ottt_cifar10_vgg9_unified"
OPTUNA_STORAGE = ""

# -----------------------------------------------------------------------------
# Path bootstrap
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR      = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if "networks" not in sys.modules:
    networks_pkg = types.ModuleType("networks")
    networks_pkg.__path__ = [str(SRC_DIR / "networks")]
    sys.modules["networks"] = networks_pkg

from datasets.cifar10_static_loader import CIFAR10StaticLoader
from networks.vgg9 import vgg9_ottt_cifar10               # ← unified factory
from trainers.ottt_trainer import OTTTTrainer


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OTTT trainer on CIFAR-10 — unified VGG9.")
    p.add_argument("--epochs",         type=int,   default=EPOCHS)
    p.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    p.add_argument("--timesteps",      type=int,   default=TIMESTEPS)
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--beta",           type=float, default=BETA)
    p.add_argument("--threshold",      type=float, default=THRESHOLD)
    p.add_argument("--seed",           type=int,   default=SEED)
    p.add_argument("--device",         choices=("auto", "cpu", "cuda"), default=DEVICE)
    p.add_argument("--optuna-trials",  type=int,   default=OPTUNA_TRIALS)
    p.add_argument("--optuna-epochs",  type=int,   default=OPTUNA_EPOCHS)
    p.add_argument("--study-name",     type=str,   default=STUDY_NAME)
    p.add_argument("--optuna-storage", type=str,   default=OPTUNA_STORAGE)
    p.add_argument("--hpc-prints",     dest="hpc_prints", action="store_true", default=HPC_PRINTS)
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
    batch_size:  int,
    timesteps:   int,
    beta:        float,
    threshold:   float,
    epochs:      int,
    lr:          float,
    seed:        int,
    device:      torch.device,
    hpc_prints:  bool = False,
    log_prefix:  str  = "",
    trial=None,
) -> Dict[str, float]:
    set_seed(seed)

    # OTTT CIFAR-10 uses static images: the same frame is fed at each timestep.
    # CIFAR10StaticLoader returns [B, 3, 32, 32]; we expand to [T, B, 3, 32, 32] below.
    train_loader, test_loader = CIFAR10StaticLoader(
        batch_size=batch_size,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
    )

    # constant_input_per_timestep=True enables the full OTTT paper recipe for static
    # CIFAR-10: loss_lambda=0.05, grad_clip=0.2, sanitize_grads=True.
    network = vgg9_ottt_cifar10(
        in_channels=3,
        num_classes=10,
        beta=beta,
        threshold=threshold,
        constant_input_per_timestep=True,
    ).to(device)

    # Dump conv1 and head weights — compare with legacy to confirm bit-identical init.
    print(f"[init-check] conv1.weight[:2,0,0,0] = {network.conv1.weight[:2,0,0,0].tolist()}")
    print(f"[init-check]  head.weight[0,:4]     = {network.head.weight[0,:4].tolist()}")

    trainer = OTTTTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        online_updates=True,
    ).to(device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=epochs)

    best_test_acc    = 0.0
    final_test_acc   = 0.0
    final_train_loss = 0.0
    non_blocking     = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # [TRAIN]
        trainer.network.train()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = len(train_loader)

        for i, (data, target) in enumerate(train_loader, 1):
            data   = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            # Expand static frame to [T, B, C, H, W] (same frame at each timestep).
            data   = data.unsqueeze(0).expand(timesteps, -1, -1, -1, -1)

            loss, pred     = trainer.train_sample(data, target)
            batch_size_cur = target.size(0)
            total_loss    += loss.item() * batch_size_cur
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += batch_size_cur

            if not hpc_prints:
                f = int(28 * i / n_batches)
                print(f"\r  [{'#' * f}{'-' * (28 - f)}] {int(100 * i / n_batches):3d}%  ", end="", flush=True)

        if not hpc_prints:
            print("\r" + " " * 40 + "\r", end="", flush=True)

        train_loss = total_loss    / total_samples if total_samples else 0.0
        train_acc  = total_correct / total_samples if total_samples else 0.0

        # [EVAL] — plain Linear head (global_pool + head): accumulate logits over T
        # timesteps, then argmax. Mirrors the legacy eval loop exactly.
        network.eval()
        total   = 0
        correct = 0
        layer_rate_sum = [0.0] * network._num_blocks
        n_eval_batches = 0
        with torch.no_grad():
            for data, target in test_loader:
                data   = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                network.reset()
                logit_sum         = None
                batch_layer_rates = [0.0] * network._num_blocks
                for t in range(timesteps):
                    spk_rec, _ = network(data)  # same static frame at each timestep
                    if logit_sum is None:
                        logit_sum = spk_rec[-1].clone()
                    else:
                        logit_sum = logit_sum + spk_rec[-1]
                    for li in range(network._num_blocks):
                        batch_layer_rates[li] += spk_rec[li].mean().item()
                preds   = logit_sum.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total   += target.size(0)
                for li in range(network._num_blocks):
                    layer_rate_sum[li] += batch_layer_rates[li] / timesteps
                n_eval_batches += 1

        avg_rates = [layer_rate_sum[li] / n_eval_batches for li in range(network._num_blocks)]
        rates_str = " ".join(f"L{li+1}={r:.3f}" for li, r in enumerate(avg_rates))

        test_acc         = correct / total if total else 0.0
        final_train_loss = train_loss
        final_test_acc   = test_acc
        best_test_acc    = max(best_test_acc, test_acc)

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
            except ImportError:
                pass

    return {
        "best_test_acc":    best_test_acc,
        "final_test_acc":   final_test_acc,
        "final_train_loss": final_train_loss,
    }


def run_optuna(args: argparse.Namespace, device: torch.device) -> None:
    try:
        import optuna
    except ImportError as err:
        raise RuntimeError("Install optuna: pip install optuna") from err

    storage = args.optuna_storage or None
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study   = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=storage is not None,
        sampler=sampler,
    )

    def objective(trial):
        lr         = trial.suggest_float("lr",   1e-4, 1e-2, log=True)
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
    args   = parse_args()
    device = get_device(args.device)
    print(f"[Run] device={device.type}  network=vgg9_ottt_cifar10 (unified)")

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
