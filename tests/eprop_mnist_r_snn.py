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
TIMESTEPS = 10          # Number of rate-coding steps produced by the MNIST loader.
NUM_WORKERS = 4         # DataLoader worker processes for MNIST loading.
DATA_ROOT = ""          # Optional MNIST root override; empty string uses the loader default.

# Network defaults
BETA = 0.95             # Recurrent hidden-layer leak/decay.
THRESHOLD = 1.0         # Hidden spiking threshold.
HIDDEN_SIZE = 100       # Number of recurrent hidden units for the RSNN.
# Network specific defaults

# Trainer defaults
# General training defaults
EPOCHS = 10             # Training epochs for the default non-Optuna run.
LR = 2e-4               # OSTTP optimizer learning rate.
SEED = 42               # Global random seed for Python, NumPy, and PyTorch.
DEVICE = "auto"         # Runtime device selection: auto, cpu, or cuda.
HPC_PRINTS = False      # If True, suppress per-batch progress bar updates.

# Trainer e-prop defaults
EPROP_GAMMA = 0.3       # Pseudo-derivative dampening γ_pd.
EPROP_TAU_MEM = 20.0    # e-prop membrane decay time constant τ_m.
EPROP_TAU_OUT = 30.0    # e-prop readout decay time constant τ_out.
EPROP_THRESHOLD = 0.03  # e-prop surrogate threshold v_th.
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
from datasets.mnist_loader import MNISTLoader
from networks.benchmarking.r_snn import RSNN
from trainers.eprop_trainer import EpropTrainer

# -----------------------------------------------------------------------------
# Tiny utilities (inlined to keep this file self-contained)
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal MNIST+RSNN e-prop test.")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    p.add_argument("--timesteps", type=int, default=TIMESTEPS, help="Number of MNIST rate-coding steps.")
    p.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    p.add_argument("--beta", type=float, default=BETA, help="Hidden recurrent decay.")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="Hidden firing threshold for the RSNN hidden neurons.")
    p.add_argument("--eprop-gamma", type=float, default=EPROP_GAMMA, help="e-prop pseudo-derivative dampening γ_pd.")
    p.add_argument("--eprop-tau-mem", type=float, default=EPROP_TAU_MEM, help="e-prop membrane decay time constant τ_m.")
    p.add_argument("--eprop-tau-out", type=float, default=EPROP_TAU_OUT, help="e-prop readout decay time constant τ_out.")
    p.add_argument("--eprop-threshold", type=float, default=EPROP_THRESHOLD, help="e-prop surrogate threshold v_th.")
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
    # Trainer e-prop parameters
    eprop_gamma: float,
    eprop_tau_mem: float,
    eprop_tau_out: float,
    eprop_threshold: float,
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
    train_loader, test_loader = MNISTLoader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
        data_root=DATA_ROOT or None,
    )
    network = RSNN(
        in_shape=(1, 28, 28),
        num_classes=10,
        hidden_sizes=(HIDDEN_SIZE,),
        beta=beta,
        threshold=threshold,
    ).to(device)
    trainer = EpropTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
        gamma=eprop_gamma,
        tau_mem=eprop_tau_mem,
        tau_out=eprop_tau_out,
        threshold=eprop_threshold,
        use_optimizer=True,
    ).to(device)

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_monitor_loss = 0.0

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
                print(f"\r  [{'#' * f}{'-' * (28 - f)}] {int(100 * i / n_batches):3d}%", end="", flush=True)

        if not hpc_prints:
            print("\r" + " " * 40 + "\r", end="", flush=True)

        monitor_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        # [EVAL] one full pass on test batches
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
                    z_t, v_t, vo = trainer._eprop_step(frame, vo)
                    vo_sum = vo.clone() if vo_sum is None else (vo_sum + vo)
                preds = vo_sum.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += target.size(0)

        test_acc = correct / total if total > 0 else 0.0

        # [METRICS] track epoch outputs and best score
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

    

    return {
        "best_test_acc": best_test_acc,
        "final_test_acc": final_test_acc,
        "final_train_monitor_loss": final_train_monitor_loss,
    }

def run_overfit_test(
    *,
    batch_size: int,
    timesteps: int,
    beta: float,
    threshold: float,
    eprop_gamma: float,
    eprop_tau_mem: float,
    eprop_tau_out: float,
    eprop_threshold: float,
    lr: float,
    seed: int,
    device: torch.device,
    subset_size: int = 64,
    epochs: int = 20,
    target_acc: float = 0.95,
) -> Dict[str, float]:
    """
    Sanity check isolata: verifica se una coppia fresh RSNN + EpropTrainer
    riesce a overfittare un piccolo subset fisso del training set.

    Il test ricrea da zero:
      - DataLoader
      - network
      - trainer

    così non altera lo stato del training principale.

    Args:
        batch_size:        Batch size usata per costruire il dataset loader.
        timesteps:         Numero di timestep del dataset.
        beta:              Leak/decay del network RSNN.
        threshold:         Threshold dei neuroni hidden del network.
        eprop_gamma:       Gamma della pseudo-derivata e-prop.
        eprop_tau_mem:     Tau_mem usata dal trainer e-prop.
        eprop_tau_out:     Tau_out usata dal trainer e-prop.
        eprop_threshold:   Threshold usata dal trainer per la pseudo-derivata.
        lr:                Learning rate.
        seed:              Seed globale.
        device:            Torch device.
        subset_size:       Numero di esempi da memorizzare.
        epochs:            Numero massimo di epoche di overfit.
        target_acc:        Accuratezza target per early stop.

    Returns:
        dict con accuratezza finale e flag di successo.
    """
    print(f"\n[Overfit Test] subset_size={subset_size} epochs={epochs}")

    # -------------------------------------------------------------------------
    # Reproducibilità
    # -------------------------------------------------------------------------
    set_seed(seed)
    non_blocking = device.type == "cuda"

    # -------------------------------------------------------------------------
    # Dataset fresh
    # -------------------------------------------------------------------------
   

    
    train_loader, test_loader = MNISTLoader(
        batch_size=batch_size,
        T=timesteps,
        pin_memory=(device.type == "cuda"),
        seed=seed,
        num_workers=NUM_WORKERS,
        data_root=DATA_ROOT or None,
    )

    dataset = train_loader.dataset
    subset_size = min(subset_size, len(dataset))
    indices = list(range(subset_size))
    subset = torch.utils.data.Subset(dataset, indices)

    subset_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=subset_size,   # singolo batch: test più rapido e diagnostico
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        collate_fn=getattr(train_loader, "collate_fn", None),
    )

    # -------------------------------------------------------------------------
    # Network fresh
    # -------------------------------------------------------------------------
    network = RSNN(
        in_shape=(1, 28, 28),
        num_classes=10,
        beta=beta,
        threshold=threshold,
    ).to(device)

    # -------------------------------------------------------------------------
    # Trainer fresh
    # -------------------------------------------------------------------------
    trainer = EpropTrainer(
        network=network,
        lr=lr,
        batch_size=subset_size,
        gamma=eprop_gamma,
        tau_mem=eprop_tau_mem,
        tau_out=eprop_tau_out,
        threshold=eprop_threshold,
        use_optimizer=True,
    ).to(device)

    # -------------------------------------------------------------------------
    # Loop di overfit
    # -------------------------------------------------------------------------
    final_acc = 0.0
    final_monitor_loss = 0.0

    for epoch in range(1, epochs + 1):
        trainer.network.train()

        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        for data, target in subset_loader:
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)

            # Se hai un metodo comune di normalizzazione, usalo qui.
            # Altrimenti sostituisci questa riga con il preprocessing corretto.
            if hasattr(trainer, "normalize_sequence"):
                data = trainer.normalize_sequence(data, timesteps=timesteps)

            loss, pred = trainer.train_sample(data, target)

            batch_size_cur = target.size(0)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += batch_size_cur
            total_loss += loss.item() * batch_size_cur

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        monitor_loss = total_loss / total_samples if total_samples > 0 else 0.0

        final_acc = acc
        final_monitor_loss = monitor_loss

        print(
            f"[Overfit] epoch={epoch}/{epochs} "
            f"monitor_loss={monitor_loss:.4f} "
            f"acc={acc:.4f}"
        )

        if acc >= target_acc:
            print(f"[Overfit] reached target accuracy ({target_acc:.2f}) early.")
            break

    success = final_acc >= target_acc

    print(
        f"[Overfit Result] final_acc={final_acc:.4f} "
        f"final_monitor_loss={final_monitor_loss:.4f} "
        f"success={'YES' if success else 'NO'}"
    )

    return {
        "overfit_acc": final_acc,
        "overfit_monitor_loss": final_monitor_loss,
        "overfit_success": success,
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
            eprop_gamma=args.eprop_gamma,
            eprop_tau_mem=args.eprop_tau_mem,
            eprop_tau_out=args.eprop_tau_out,
            eprop_threshold=args.eprop_threshold,
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
        eprop_gamma=args.eprop_gamma,
        eprop_tau_mem=args.eprop_tau_mem,
        eprop_tau_out=args.eprop_tau_out,
        eprop_threshold=args.eprop_threshold,
        seed=args.seed,
        device=device,
        hpc_prints=args.hpc_prints,
    )
    print(f"\n[Done] final_test_acc={result['final_test_acc']:.4f} best_test_acc={result['best_test_acc']:.4f}")

    print("\n[Running Overfit Test]")
    run_overfit_test(
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        beta=args.beta,
        threshold=args.threshold,
        eprop_gamma=args.eprop_gamma,
        eprop_tau_mem=args.eprop_tau_mem,
        eprop_tau_out=args.eprop_tau_out,
        eprop_threshold=args.eprop_threshold,
        lr=args.lr,
        seed=args.seed + 9999,  # Different seed for overfit test
        device=device,
    )

    

if __name__ == "__main__":
    main()
