#!/usr/bin/env python3
"""
Single-experiment runner.

Receives a fully resolved ExperimentSpec (as a JSON file path), trains the
specified (trainer × model × dataset) combination, runs NeuroBench evaluation,
and writes results to the output directory.

Usage:
    python3 experiment.py <spec_json_path> <output_dir>

Called by run_exp_campaign.py for each experiment in the campaign.
You can also call it directly for debugging a single experiment.
"""

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Path setup ─────────────────────────────────────────────────────────────
# Allow imports from src/ without installing the package
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from campaign.experiment_spec import ExperimentSpec
from campaign.results import (
    experiment_dir,
    save_experiment_config,
    save_experiment_metrics,
)
from campaign.training_loop import evaluate, train_one_epoch
from datasets import LOADER_REGISTRY
from networks import NETWORK_REGISTRY
from trainers import TRAINER_REGISTRY
from campaign.neurobench_eval import run_neurobench


def main(spec_path: str, output_dir: str) -> None:
    spec = ExperimentSpec.load(spec_path)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Logging ────────────────────────────────────────────────────────────
    log_level = spec.runtime.get("log_level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out / "log.txt"),
        ],
    )
    log = logging.getLogger(__name__)
    log.info("Starting experiment: %s", spec.name)

    # ── Reproducibility ────────────────────────────────────────────────────
    seed = int(spec.runtime.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Device ─────────────────────────────────────────────────────────────
    device_str = spec.runtime.get("device", "cuda")
    device = torch.device(
        device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    log.info("Using device: %s", device)

    # ── Dataset ────────────────────────────────────────────────────────────
    ds_cfg = dict(spec.dataset)
    dataset_name = ds_cfg.pop("name")
    batch_size   = ds_cfg.get("batch_size", 256)
    T            = ds_cfg.get("T", 25)

    if dataset_name not in LOADER_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {sorted(LOADER_REGISTRY)}"
        )
    log.info("Loading dataset: %s (T=%d, batch=%d)", dataset_name, T, batch_size)
    ds_cfg["seed"] = seed
    ds_cfg["pin_memory"] = (device.type == "cuda")
    train_loader, test_loader = LOADER_REGISTRY[dataset_name](
        **{k: v for k, v in ds_cfg.items() if v is not None}
    )

    # ── Network ────────────────────────────────────────────────────────────
    m_cfg = dict(spec.model)
    model_name = m_cfg.pop("name")
    m_cfg.pop("algorithm_name", None)  # unused after removing algorithm-model override logic

    if model_name not in NETWORK_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(NETWORK_REGISTRY)}"
        )
    log.info("Building network: %s", model_name)
    network = NETWORK_REGISTRY[model_name](
        **{k: v for k, v in m_cfg.items() if v is not None}
    )
    network = network.to(device)

    # ── Trainer ────────────────────────────────────────────────────────────
    t_cfg = dict(spec.trainer)
    trainer_name = t_cfg.pop("name")

    if trainer_name not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown trainer '{trainer_name}'. "
            f"Available: {sorted(TRAINER_REGISTRY.keys())}"
        )
    TrainerClass = TRAINER_REGISTRY[trainer_name]

    log.info("Building trainer: %s", trainer_name)
    t_cfg.pop("batch_size", None)
    trainer = TrainerClass(
        network=network,
        lr=t_cfg.pop("lr", 1e-3),
        batch_size=batch_size,
        **{k: v for k, v in t_cfg.items() if v is not None},
    )
    trainer = trainer.to(device)

    # ── Training loop ──────────────────────────────────────────────────────
    epochs = int(spec.runtime.get("epochs", 10))
    epoch_metrics = []

    log.info("Training for %d epochs...", epochs)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(trainer, train_loader, device)
        test_acc      = evaluate(network, test_loader, device)

        epoch_metrics.append({
            "epoch":         epoch,
            "train_loss":    train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy":  test_acc,
        })
        log.info(
            "Epoch %d/%d — loss: %.4f  train_acc: %.4f  test_acc: %.4f",
            epoch, epochs,
            train_metrics["loss"], train_metrics["accuracy"], test_acc,
        )

    elapsed = time.time() - t0
    final_test_acc = epoch_metrics[-1]["test_accuracy"] if epoch_metrics else 0.0
    log.info("Training done in %.1f s. Final test accuracy: %.4f", elapsed, final_test_acc)

    # ── NeuroBench evaluation ──────────────────────────────────────────────
    nb_results = {}
    if spec.runtime.get("neurobench", True):
        log.info("Running NeuroBench evaluation...")
        try:
            nb_results = run_neurobench(
                network=network,
                test_loader=test_loader,
                device=str(device),
                num_timesteps=T,
            )
        except Exception as e:
            log.warning("NeuroBench evaluation failed: %s", e)
            nb_results = {"error": str(e)}

    # ── Save results ───────────────────────────────────────────────────────
    save_experiment_config(out, spec)

    metrics = {
        "name":          spec.name,
        "trainer":       trainer_name,
        "model":         model_name,
        "dataset":       dataset_name,
        "test_accuracy": final_test_acc,
        "train_loss":    epoch_metrics[-1]["train_loss"] if epoch_metrics else 0.0,
        "elapsed_s":     elapsed,
        "epochs":        epochs,
        "epoch_metrics": epoch_metrics,
        "neurobench":    nb_results,
    }
    save_experiment_metrics(out, metrics)

    log.info("Results written to: %s", out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 experiment.py <spec_json_path> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
