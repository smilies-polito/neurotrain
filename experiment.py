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

When spec.opt is True, an Optuna study is run instead of a single training run.
Each trial samples new hyper-parameters from the search space defined in the
YAML config (tunable blocks: {value, type, min, max, list}).  Study artefacts
(trials.csv, best_params.yaml) are written to <output_dir>/optuna/.
"""

import gc
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Path setup ─────────────────────────────────────────────────────────────
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


# Keys in YAML configs used only by the compatibility system (campaign/compatibility.py).
# They must be stripped before passing config dicts to constructors.
_CONFIG_METADATA_KEYS = {"supported_net_types", "net_type"}

def _strip_metadata(cfg: dict) -> dict:
    """Remove campaign-level metadata keys that are not constructor arguments."""
    return {k: v for k, v in cfg.items() if k not in _CONFIG_METADATA_KEYS}


def _train_and_evaluate(spec: ExperimentSpec, out: Path, log: logging.Logger) -> dict:
    """
    Core training loop: build dataset/model/trainer, train for spec.runtime
    epochs, optionally run NeuroBench, and return a metrics dict.

    The spec must already have plain (non-tunable-block) values — either
    normalised by normalize_optuna_attrs or resolved by suggest_from_cfg.
    """
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
    batch_size   = spec.runtime.get("batch_size", 256)
    T            = ds_cfg.get("T", 25)
    # Whether the dataset repeats the same static frame every timestep.
    # Derived from direct_coding in the dataset YAML; used to configure OTTT
    # and the evaluator — not stored on the network.
    constant_input_per_timestep = bool(ds_cfg.get("direct_coding", False))

    if dataset_name not in LOADER_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {sorted(LOADER_REGISTRY)}"
        )
    log.info("Loading dataset: %s (T=%d, batch=%d)", dataset_name, T, batch_size)
    ds_cfg["seed"] = seed
    ds_cfg.setdefault("pin_memory", False)
    train_loader, test_loader = LOADER_REGISTRY[dataset_name](
        batch_size=batch_size,
        **{k: v for k, v in _strip_metadata(ds_cfg).items() if v is not None}
    )

    # ── Network ────────────────────────────────────────────────────────────
    m_cfg = dict(spec.model)
    model_name = m_cfg.pop("name")
    m_cfg.pop("algorithm_name", None)

    if model_name not in NETWORK_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(NETWORK_REGISTRY)}"
        )
    log.info("Building network: %s", model_name)
    network = NETWORK_REGISTRY[model_name](
        **{k: v for k, v in _strip_metadata(m_cfg).items() if v is not None}
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
    # Inject constant_input_per_timestep for trainers that accept it (OTTT).
    # Passing it as a kwarg to trainers that don't accept it is harmless — they
    # will forward it via **kwargs or ignore it if their signature doesn't include it.
    import inspect as _inspect
    _trainer_sig = _inspect.signature(TrainerClass.__init__).parameters
    _extra: dict = {}
    if "constant_input_per_timestep" in _trainer_sig:
        _extra["constant_input_per_timestep"] = constant_input_per_timestep
    trainer = TrainerClass(
        network=network,
        lr=t_cfg.pop("lr", 1e-3),
        batch_size=batch_size,
        **_extra,
        **{k: v for k, v in _strip_metadata(t_cfg).items() if v is not None and k not in _extra},
    )
    trainer = trainer.to(device)

    # ── Training loop ──────────────────────────────────────────────────────
    epochs   = int(spec.runtime.get("epochs", 10))
    progress = bool(spec.runtime.get("progress", False))
    epoch_metrics = []

    log.info("Training for %d epochs...", epochs)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(trainer, train_loader, device, progress=progress)
        test_acc      = evaluate(network, test_loader, device,
                                 constant_input_per_timestep=constant_input_per_timestep,
                                 progress=progress)

        epoch_metrics.append({
            "epoch":          epoch,
            "train_loss":     train_metrics["loss"],
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

    # Free GPU memory so successive Optuna trials don't accumulate allocations.
    # gc.collect() must run before empty_cache(): it resolves Python reference cycles
    # (e.g. trainer ↔ network via hooks/synapse_layers) so their GPU tensors are
    # finalised first; empty_cache() then returns all freed CUDA memory to the allocator.
    del trainer, train_loader, test_loader, network
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name":           spec.name,
        "trainer":        trainer_name,
        "model":          model_name,
        "dataset":        dataset_name,
        "test_accuracy":  final_test_acc,
        "train_loss":     epoch_metrics[-1]["train_loss"] if epoch_metrics else 0.0,
        "elapsed_s":      elapsed,
        "epochs":         epochs,
        "epoch_metrics":  epoch_metrics,
        "neurobench":     nb_results,
    }


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

    # ── Normal run (opt=False) ──────────────────────────────────────────────
    if not spec.opt:
        metrics = _train_and_evaluate(spec, out, log)
        save_experiment_config(out, spec)
        save_experiment_metrics(out, metrics)
        log.info("Results written to: %s", out)
        return

    # ── Optuna run (opt=True) ───────────────────────────────────────────────
    # Each trial resolves tunable blocks to concrete values, then trains.
    # All trials run in the same subprocess; GPU memory is freed between trials.
    from campaign.optuna_opt import run_study, suggest_from_cfg

    def objective(trial) -> float:
        trial_dir = out / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_spec = ExperimentSpec(
            name    = f"{spec.name}_t{trial.number}",
            opt     = False,
            trainer = suggest_from_cfg(spec.trainer, trial, "trainer"),
            model   = suggest_from_cfg(spec.model,   trial, "model"),
            dataset = suggest_from_cfg(spec.dataset,  trial, "dataset"),
            runtime = spec.runtime,
            optuna  = {},
        )

        metrics = _train_and_evaluate(trial_spec, trial_dir, log)
        save_experiment_config(trial_dir, trial_spec)
        save_experiment_metrics(trial_dir, metrics)

        # Free GPU memory before the next trial (gc before empty_cache — same reason as above)
        gc.collect()
        torch.cuda.empty_cache()
        return float(metrics["test_accuracy"])

    seed = int(spec.runtime.get("seed", 42))
    optuna_cfg = {**spec.optuna, "seed": seed}

    study = run_study(
        spec_name  = spec.name,
        optuna_cfg = optuna_cfg,
        out_dir    = out,
        objective  = objective,
    )

    # Save the best trial's config and metrics at the experiment root
    best_trial_dir = out / "trials" / f"trial_{study.best_trial.number:04d}"
    if (best_trial_dir / "metrics.json").exists():
        import shutil
        shutil.copy(best_trial_dir / "metrics.json", out / "metrics.json")
        shutil.copy(best_trial_dir / "config.yaml",  out / "config.yaml")

    log.info("Optuna study complete. Best trial: #%d", study.best_trial.number)
    log.info("Results written to: %s", out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 experiment.py <spec_json_path> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
