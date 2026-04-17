#!/usr/bin/env python3
"""
Test for ELL-style local learning on ConvSNN + DVS Gesture.

The production ELLTrainer is written around flat synapse/neuron stacks and is
not directly suitable for convolutional feature maps. This test keeps the
trainer unchanged and defines a small conv-specific adapter trainer locally.

Usage:
    python test_ell_conv_snn_dvsgesture.py
    python test_ell_conv_snn_dvsgesture.py --epochs 20 --beta 0.95
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
import torch.nn.functional as F

BATCH_SIZE = 64
TIMESTEPS = 10
NUM_WORKERS = 4
DATA_ROOT = ""

BETA = 0.95
THRESHOLD = 1.0

EPOCHS = 10
LR = 5e-4
LR_DECAY_FACTOR = 5.0
LR_DECAY_EVERY = 15
SEED = 42
DEVICE = "auto"
HPC_PRINTS = False

OPTUNA_TRIALS = 0
OPTUNA_EPOCHS = 10
STUDY_NAME = "ell_conv_dvsgesture_study"
OPTUNA_STORAGE = ""

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
from networks.conv_snn import ConvSNN
from trainers.ell_trainer import AuxClassifier


class FlatAuxClassifier(AuxClassifier):
    """Aux classifier that flattens convolutional spikes before decoding."""

    def forward(self, spike_in: torch.Tensor) -> torch.Tensor:
        flat = spike_in.flatten(1)
        return super().forward(flat)


class ELLConvAdapterTrainer(nn.Module):
    """
    Conv-specific local-learning adapter mirroring the current ELLTrainer logic
    while preserving spatial activations through convolutional layers.
    """

    def __init__(self, network: ConvSNN, lr: float = 5e-4, batch_size: int = 64):
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size

        ref_neuron = network.lif1
        neuron_beta = float(ref_neuron.beta)
        neuron_thresh = float(ref_neuron.threshold)
        neuron_reset = ref_neuron.reset_mechanism
        neuron_spike_grad = ref_neuron.spike_grad

        with torch.no_grad():
            device = next(network.parameters()).device
            dummy = torch.zeros(1, *network.in_shape, device=device)
            cur1 = network.pool1(network.conv1(dummy))
            spk1, _ = network.lif1(cur1)
            network.lif1.reset_mem()

            cur2 = network.pool2(network.conv2(spk1))
            spk2, _ = network.lif2(cur2)
            network.lif2.reset_mem()
            network.lif_out.reset_mem()

            aux1_in = int(spk1.flatten(1).shape[1])
            aux2_in = int(spk2.flatten(1).shape[1])

        self.aux_classifiers = nn.ModuleList([
            FlatAuxClassifier(
                num_in=aux1_in,
                num_classes=network.n_classes,
                beta=neuron_beta,
                threshold=neuron_thresh,
                spike_grad=neuron_spike_grad,
                reset_mechanism=neuron_reset,
                bias=False,
            ),
            FlatAuxClassifier(
                num_in=aux2_in,
                num_classes=network.n_classes,
                beta=neuron_beta,
                threshold=neuron_thresh,
                spike_grad=neuron_spike_grad,
                reset_mechanism=neuron_reset,
                bias=False,
            ),
        ]).to(next(network.parameters()).device)

        self.optimizers = [
            torch.optim.Adam(
                list(network.conv1.parameters()) +
                list(network.lif1.parameters()) +
                list(self.aux_classifiers[0].parameters()),
                lr=lr,
                weight_decay=0.0,
            ),
            torch.optim.Adam(
                list(network.conv2.parameters()) +
                list(network.lif2.parameters()) +
                list(self.aux_classifiers[1].parameters()),
                lr=lr,
                weight_decay=0.0,
            ),
            torch.optim.Adam(
                list(network.fc.parameters()) +
                list(network.lif_out.parameters()),
                lr=lr,
                weight_decay=0.0,
            ),
        ]

    def reset(self) -> None:
        self.network.reset()
        for aux in self.aux_classifiers:
            aux.reset()

    def to(self, device):
        self.network = self.network.to(device)
        self.aux_classifiers = self.aux_classifiers.to(device)
        self.optimizers = [
            torch.optim.Adam(
                list(self.network.conv1.parameters()) +
                list(self.network.lif1.parameters()) +
                list(self.aux_classifiers[0].parameters()),
                lr=self.lr,
                weight_decay=0.0,
            ),
            torch.optim.Adam(
                list(self.network.conv2.parameters()) +
                list(self.network.lif2.parameters()) +
                list(self.aux_classifiers[1].parameters()),
                lr=self.lr,
                weight_decay=0.0,
            ),
            torch.optim.Adam(
                list(self.network.fc.parameters()) +
                list(self.network.lif_out.parameters()),
                lr=self.lr,
                weight_decay=0.0,
            ),
        ]
        return self

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        num_timesteps = data.shape[0]
        batch_size = data.shape[1]
        device = data.device
        n_classes = self.network.n_classes

        target_onehot = torch.zeros(batch_size, n_classes, device=device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

        self.reset()
        x_const = data[0]
        spk_sum = torch.zeros(batch_size, n_classes, device=device)
        total_loss = 0.0

        for _ in range(num_timesteps):
            cur1 = self.network.pool1(self.network.conv1(x_const))
            spk1, _ = self.network.lif1(cur1)
            self.network.lif1.mem = self.network.lif1.mem.detach()

            cur2 = self.network.pool2(self.network.conv2(spk1.detach()))
            spk2, _ = self.network.lif2(cur2)
            self.network.lif2.mem = self.network.lif2.mem.detach()

            flat = spk2.detach().flatten(1)
            spk_out, _ = self.network.lif_out(self.network.fc(flat))
            self.network.lif_out.mem = self.network.lif_out.mem.detach()

            for aux in self.aux_classifiers:
                aux.lif.mem = aux.lif.mem.detach()

            aux1 = self.aux_classifiers[0](spk1)
            aux2 = self.aux_classifiers[1](spk2)

            losses = [
                F.mse_loss(aux1, target_onehot),
                F.mse_loss(aux2, target_onehot),
                F.mse_loss(spk_out, target_onehot),
            ]

            for loss_l in losses:
                total_loss += loss_l.item()

            for idx in reversed(range(len(losses))):
                self.optimizers[idx].zero_grad()
                losses[idx].backward(retain_graph=(idx > 0))
                self.optimizers[idx].step()

            spk_sum += spk_out.detach()

        avg_loss = torch.tensor(total_loss / (num_timesteps * 3), device=device)
        pred = spk_sum.argmax(dim=1)
        return avg_loss, pred

    @torch.no_grad()
    def predict(self, data: torch.Tensor):
        num_timesteps = data.shape[0]
        batch_size = data.shape[1]
        device = data.device
        n_classes = self.network.n_classes

        self.network.reset()
        x_const = data[0]
        spk_sum = torch.zeros(batch_size, n_classes, device=device)

        for _ in range(num_timesteps):
            spk_rec, _ = self.network(x_const)
            spk_sum += spk_rec[-1].detach()

        return spk_sum.argmax(dim=1, keepdim=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DVS Gesture + ConvSNN + ELL-style local learning test."
    )
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--lr-decay-factor", type=float, default=LR_DECAY_FACTOR)
    p.add_argument("--lr-decay-every", type=int, default=LR_DECAY_EVERY)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    p.add_argument("--hpc-prints", dest="hpc_prints", action="store_true",
                   default=HPC_PRINTS)
    p.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS)
    p.add_argument("--optuna-epochs", type=int, default=OPTUNA_EPOCHS)
    p.add_argument("--study-name", type=str, default=STUDY_NAME)
    p.add_argument("--optuna-storage", type=str, default=OPTUNA_STORAGE)
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
    batch_size: int,
    timesteps: int,
    beta: float,
    threshold: float,
    epochs: int,
    lr: float,
    lr_decay_factor: float,
    lr_decay_every: int,
    seed: int,
    device: torch.device,
    hpc_prints: bool = False,
    log_prefix: str = "",
    trial=None,
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

    network = ConvSNN(
        in_shape=(2, 128, 128),
        num_classes=11,
        beta=beta,
        threshold=threshold,
    ).to(device)

    trainer = ELLConvAdapterTrainer(
        network=network,
        lr=lr,
        batch_size=batch_size,
    ).to(device)

    schedulers = [
        torch.optim.lr_scheduler.StepLR(
            opt, step_size=lr_decay_every, gamma=1.0 / lr_decay_factor
        )
        for opt in trainer.optimizers
    ]

    best_test_acc = 0.0
    final_test_acc = 0.0
    final_train_loss = 0.0
    non_blocking = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        network.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n_batches = len(train_loader)

        for i, (data, target) in enumerate(train_loader, 1):
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)

            loss, pred = trainer.train_sample(data, target)

            bs_cur = target.size(0)
            total_loss += loss.item() * bs_cur
            total_correct += pred.eq(target).sum().item()
            total_samples += bs_cur

            if not hpc_prints:
                f = int(28 * i / n_batches)
                print(
                    f"\r  [{'#' * f}{'-' * (28 - f)}] {int(100 * i / n_batches):3d}%",
                    end="", flush=True,
                )

        if not hpc_prints:
            print("\r" + " " * 40 + "\r", end="", flush=True)

        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        for sched in schedulers:
            sched.step()

        network.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)

                pred = trainer.predict(data)
                correct += pred.squeeze(1).eq(target).sum().item()
                total += target.size(0)

        test_acc = correct / total if total > 0 else 0.0
        final_train_loss = train_loss
        final_test_acc = test_acc
        best_test_acc = max(best_test_acc, test_acc)

        epoch_time_s = time.perf_counter() - epoch_start
        current_lr = schedulers[0].get_last_lr()[0]
        print(
            f"{log_prefix}epoch={epoch}/{epochs} "
            f"loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} "
            f"lr={current_lr:.2e} "
            f"time={epoch_time_s:.2f}s"
        )

        if trial is not None:
            import optuna
            trial.report(test_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return {
        "best_test_acc": best_test_acc,
        "final_test_acc": final_test_acc,
        "final_train_loss": final_train_loss,
    }


def run_optuna(args: argparse.Namespace, device: torch.device) -> None:
    try:
        import optuna
    except ImportError as err:
        raise RuntimeError(
            "Optuna is not installed. Install it with `pip install optuna`."
        ) from err

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
        beta = trial.suggest_float("beta", 0.85, 0.99)
        threshold = trial.suggest_float("threshold", 0.5, 2.0)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        lr_decay_factor = trial.suggest_float("lr_decay_factor", 2.0, 10.0)
        lr_decay_every = trial.suggest_int("lr_decay_every", 5, 25)

        result = run_training(
            epochs=args.optuna_epochs,
            batch_size=batch_size,
            timesteps=args.timesteps,
            beta=beta,
            threshold=threshold,
            lr=lr,
            lr_decay_factor=lr_decay_factor,
            lr_decay_every=lr_decay_every,
            seed=args.seed + trial.number,
            device=device,
            hpc_prints=args.hpc_prints,
            log_prefix=f"[trial {trial.number}] ",
            trial=trial,
        )
        trial.set_user_attr("final_test_acc", result["final_test_acc"])
        return result["best_test_acc"]

    print(
        f"[Optuna] trials={args.optuna_trials} "
        f"epochs_per_trial={args.optuna_epochs} "
        f"study={args.study_name}"
    )
    study.optimize(objective, n_trials=args.optuna_trials)

    print("\n[Optuna] Best trial")
    print(f"  value={study.best_value:.4f}")
    print(f"  params={study.best_params}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    print(f"[ELL Conv DVSGesture test] device={device.type}")
    print(f"[ELL Conv DVSGesture test] beta={args.beta} threshold={args.threshold}")
    print(f"[ELL Conv DVSGesture test] lr={args.lr} decay=÷{args.lr_decay_factor} every {args.lr_decay_every} ep")
    print(f"[ELL Conv DVSGesture test] timesteps={args.timesteps} batch_size={args.batch_size} epochs={args.epochs}")

    if args.optuna_trials > 0:
        run_optuna(args, device)
        return

    result = run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        beta=args.beta,
        threshold=args.threshold,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_every=args.lr_decay_every,
        seed=args.seed,
        device=device,
        hpc_prints=args.hpc_prints,
    )

    print(
        f"\n[Done] final_test_acc={result['final_test_acc']:.4f} "
        f"best_test_acc={result['best_test_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
