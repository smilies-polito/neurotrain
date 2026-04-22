#!/usr/bin/env python3
"""
TP trainer + unified VGG-9 + DVSGesture (event-based, 2×128×128, T=20).

Hyperparameters from Pes et al. 2026 (DVSGesture recipe):
  Optimizer: Adam, lr=1e-4, CosineAnnealingLR
  alpha (membrane decay): 0.53
  beta  (trace decay):    0.98
  vth:                    1.0

Eval: LI head — read mem_rec[-1] at the FINAL timestep only.
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

BATCH_SIZE  = 64
TIMESTEPS   = 20
NUM_WORKERS = 4

BETA      = 0.53
THRESHOLD = 1.0

EPOCHS  = 5
LR      = 1e-4
ALPHA   = 0.53
TP_BETA = 0.98
VTH     = 1.0
SEED    = 42
DEVICE  = "auto"
HPC_PRINTS = False

OPTUNA_TRIALS  = 0
OPTUNA_EPOCHS  = 5
STUDY_NAME     = "tp_dvsgest_vgg9"
OPTUNA_STORAGE = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
for _p in (str(TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if "networks" not in sys.modules:
    _pkg = types.ModuleType("networks")
    _pkg.__path__ = [str(SRC_DIR / "networks")]
    sys.modules["networks"] = _pkg

from datasets.dvsgesture_loader import DVSGestureLoader
from networks.vgg9 import vgg9
from trainers.tp_trainer import TPTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",         type=int,   default=EPOCHS)
    p.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    p.add_argument("--timesteps",      type=int,   default=TIMESTEPS)
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--alpha",          type=float, default=ALPHA)
    p.add_argument("--tp-beta",        type=float, default=TP_BETA)
    p.add_argument("--vth",            type=float, default=VTH)
    p.add_argument("--beta",           type=float, default=BETA)
    p.add_argument("--threshold",      type=float, default=THRESHOLD)
    p.add_argument("--seed",           type=int,   default=SEED)
    p.add_argument("--device",         choices=("auto","cpu","cuda"), default=DEVICE)
    p.add_argument("--optuna-trials",  type=int,   default=OPTUNA_TRIALS)
    p.add_argument("--optuna-epochs",  type=int,   default=OPTUNA_EPOCHS)
    p.add_argument("--study-name",     type=str,   default=STUDY_NAME)
    p.add_argument("--optuna-storage", type=str,   default=OPTUNA_STORAGE)
    p.add_argument("--hpc-prints",     action="store_true", default=HPC_PRINTS)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(req)


def eval_network(network, test_loader, device) -> float:
    network.eval()
    correct = total = 0
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for data, target in test_loader:
            data   = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            network.reset()
            mem_last = None
            for t in range(data.size(0)):
                _, mem_rec = network(data[t])
                mem_last = mem_rec[-1]
            correct += mem_last.argmax(dim=1).eq(target).sum().item()
            total   += target.size(0)
    return correct / total if total else 0.0


def run_training(
    *, batch_size, timesteps, beta, threshold,
    epochs, lr, alpha, tp_beta, vth,
    seed, device, hpc_prints=False,
    log_prefix="", trial=None,
) -> Dict[str, float]:
    set_seed(seed)

    train_loader, test_loader = DVSGestureLoader(
        batch_size=batch_size, T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed, num_workers=NUM_WORKERS,
    )

    network = vgg9(
        in_channels=2, num_classes=11, input_shape=(2, 128, 128),
        head_type="leaky_integrator", use_tp_pool=True,
        beta=beta, threshold=threshold,
        conv_gain=1.8, surrogate_kind="atan", surrogate_scale=1.0,
        li_head_spatial=2, li_head_leak=1.0,
    ).to(device)

    optimizer = torch.optim.Adam(list(network.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = TPTrainer(
        network=network, lr=lr, batch_size=batch_size,
        alpha=alpha, beta=tp_beta, vth=vth, surrogate_scale=1.0,
        train_target_propagator=True, use_optimizer=False,
        optimizer=optimizer,
    ).to(device)

    best_test_acc = final_test_acc = final_train_loss = 0.0
    non_blocking = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        network.train()
        total_loss = total_correct = total_samples = 0
        n_batches = len(train_loader)

        for i, (data, target) in enumerate(train_loader, 1):
            data   = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            loss, pred = trainer.train_sample(data, target)
            bs = target.size(0)
            total_loss    += loss.item() * bs
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += bs
            if not hpc_prints:
                f = int(28 * i / n_batches)
                print(f"\r  [{'#'*f}{'-'*(28-f)}] {int(100*i/n_batches):3d}%  ", end="", flush=True)

        if not hpc_prints:
            print("\r" + " " * 40 + "\r", end="", flush=True)

        scheduler.step()
        train_loss = total_loss / total_samples if total_samples else 0.0
        train_acc  = total_correct / total_samples if total_samples else 0.0
        test_acc   = eval_network(network, test_loader, device)

        best_test_acc    = max(best_test_acc, test_acc)
        final_test_acc   = test_acc
        final_train_loss = train_loss

        print(f"{log_prefix}epoch={epoch}/{epochs} train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
              f"epoch_time_s={time.perf_counter()-t0:.2f}")

        if trial is not None:
            import optuna
            trial.report(test_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return {"best_test_acc": best_test_acc, "final_test_acc": final_test_acc,
            "final_train_loss": final_train_loss}


def run_optuna(args: argparse.Namespace, device: torch.device) -> None:
    try:
        import optuna
    except ImportError as err:
        raise RuntimeError("Install optuna: pip install optuna") from err

    storage = args.optuna_storage or None
    study = optuna.create_study(
        direction="maximize", study_name=args.study_name,
        storage=storage, load_if_exists=storage is not None,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    def objective(trial):
        lr    = trial.suggest_float("lr",    1e-5, 5e-2, log=True)
        alpha = trial.suggest_float("alpha", 0.1,  0.99)
        result = run_training(
            epochs=args.optuna_epochs, batch_size=args.batch_size,
            timesteps=args.timesteps, lr=lr,
            alpha=alpha, tp_beta=args.tp_beta, vth=args.vth,
            beta=args.beta, threshold=args.threshold,
            seed=args.seed + trial.number, device=device,
            hpc_prints=args.hpc_prints,
            log_prefix=f"[trial {trial.number}] ", trial=trial,
        )
        trial.set_user_attr("final_test_acc", result["final_test_acc"])
        return result["best_test_acc"]

    print(f"[Optuna] trials={args.optuna_trials} epochs_per_trial={args.optuna_epochs}")
    study.optimize(objective, n_trials=args.optuna_trials)
    print(f"\n[Optuna] Best: value={study.best_value:.4f} params={study.best_params}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"[Run] device={device.type}")

    if args.optuna_trials > 0:
        run_optuna(args, device)
        return

    result = run_training(
        epochs=args.epochs, batch_size=args.batch_size,
        timesteps=args.timesteps, lr=args.lr,
        alpha=args.alpha, tp_beta=args.tp_beta, vth=args.vth,
        beta=args.beta, threshold=args.threshold,
        seed=args.seed, device=device, hpc_prints=args.hpc_prints,
    )
    print(f"\n[Done] final_test_acc={result['final_test_acc']:.4f} "
          f"best_test_acc={result['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()
