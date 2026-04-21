"""
Campaign builder — converts input YAML files to lists of ExperimentSpec.

Two entry points:
  from_benchmarking(yaml_path) — Benchmarking mode: cartesian product of
      trainers × models × datasets, filtered by compatibility.
  from_custom(yaml_path) — Custom mode: per-experiment configs with overrides.

Both return list[ExperimentSpec] which is the common sink for experiment.py.
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path

import yaml

from campaign.compatibility import filter_combinations, skipped_combinations
from campaign.config_loader import (
    list_defaults,
    load_default,
    load_yaml,
    merge,
    normalize_optuna_attrs,
    resolve_model_for_dataset,
    resolve_model_for_trainer_and_dataset,
)
from campaign.experiment_spec import ExperimentSpec

log = logging.getLogger(__name__)

# Default runtime settings used when not specified in input YAML
_DEFAULT_RUNTIME = {
    "epochs": 10,
    "device": "cuda",
    "seed": 42,
    "log_level": "INFO",
    "neurobench": True,
}


def from_benchmarking(yaml_path: str | Path) -> list[ExperimentSpec]:
    """
    Build experiment list from a benchmarking.yaml file.

    The input file lists trainers, models and datasets to compare.
    Empty lists mean "use all available defaults".
    The cartesian product is filtered for compatibility.

    Example input:
        trainers: [bptt, stsf]
        models: [fc_snn]
        datasets: [MNIST, FashionMNIST]
        runtime:
            epochs: 20

    Returns:
        List of fully resolved ExperimentSpec, one per valid triple.
    """
    cfg = load_yaml(Path(yaml_path))

    trainers = cfg.get("trainers") or list_defaults("trainers")
    models   = cfg.get("models")   or list_defaults("models")
    datasets = cfg.get("datasets") or list_defaults("datasets")
    runtime  = merge(_DEFAULT_RUNTIME, cfg.get("runtime") or {})
    opt      = bool(cfg.get("opt", False))
    optuna   = cfg.get("optuna") or {}

    # Report skipped combinations for transparency
    skipped = skipped_combinations(trainers, models, datasets)
    if skipped:
        log.info(
            "Skipping %d incompatible combinations: %s",
            len(skipped),
            skipped,
        )

    valid = filter_combinations(trainers, models, datasets)
    if not valid:
        raise ValueError(
            "No valid (trainer, model, dataset) combinations found. "
            "Check compatibility.py or your input file."
        )

    specs = []
    for trainer_name, model_name, dataset_name in valid:
        spec = _build_spec(
            trainer_name=trainer_name,
            model_name=model_name,
            dataset_name=dataset_name,
            trainer_override={},
            model_override={},
            dataset_override={},
            runtime=runtime,
            exp_name=None,
            opt=opt,
            optuna=optuna,
        )
        specs.append(spec)

    log.info("Built %d experiment specs from benchmarking config.", len(specs))
    return specs


def from_custom(yaml_path: str | Path) -> list[ExperimentSpec]:
    """
    Build experiment list from a custom experiments.yaml file.

    Each top-level key is one experiment. The sub-keys are:
        name:    human-readable label (optional, defaults to key)
        opt:     enable Optuna (optional, default False)
        trainer: {name: ..., <overrides>}
        model:   {name: ..., <overrides>}
        dataset: {name: ..., <overrides>}
        runtime: {epochs: ..., device: ..., seed: ...}

    Example:
        my_exp:
            name: bptt_small_fc_mnist
            trainer:
                name: bptt
                lr: 5e-4
            model:
                name: fc_snn
                hidden_sizes: [128]
            dataset:
                name: MNIST
                T: 25
            runtime:
                epochs: 5

    Returns:
        List of ExperimentSpec, one per experiment entry.
    """
    raw = load_yaml(Path(yaml_path))
    specs = []

    for key, exp_cfg in raw.items():
        if not isinstance(exp_cfg, dict):
            log.warning("Skipping non-dict entry '%s' in custom config.", key)
            continue

        trainer_cfg = exp_cfg.get("trainer") or {}
        model_cfg   = exp_cfg.get("model")   or {}
        dataset_cfg = exp_cfg.get("dataset") or {}
        runtime_override = exp_cfg.get("runtime") or {}

        trainer_name = trainer_cfg.get("name") or trainer_cfg.get("trainer")
        model_name   = model_cfg.get("name")   or model_cfg.get("model")
        dataset_name = dataset_cfg.get("name") or dataset_cfg.get("dataset")

        if not all([trainer_name, model_name, dataset_name]):
            raise ValueError(
                f"Experiment '{key}' is missing trainer/model/dataset name. "
                "Each component config must include a 'name' field."
            )

        spec = _build_spec(
            trainer_name=trainer_name,
            model_name=model_name,
            dataset_name=dataset_name,
            trainer_override={k: v for k, v in trainer_cfg.items() if k != "name"},
            model_override={k: v for k, v in model_cfg.items() if k != "name"},
            dataset_override={k: v for k, v in dataset_cfg.items() if k != "name"},
            runtime=merge(_DEFAULT_RUNTIME, runtime_override),
            exp_name=exp_cfg.get("name") or key,
            opt=bool(exp_cfg.get("opt", False)),
            optuna=exp_cfg.get("optuna") or {},
        )
        specs.append(spec)

    log.info("Built %d experiment specs from custom config.", len(specs))
    return specs


# ------------------------------------------------------------------ #
# Internal helpers                                                      #
# ------------------------------------------------------------------ #

def _build_spec(
    trainer_name: str,
    model_name: str,
    dataset_name: str,
    trainer_override: dict,
    model_override: dict,
    dataset_override: dict,
    runtime: dict,
    exp_name: str | None,
    opt: bool,
    optuna: dict | None = None,
) -> ExperimentSpec:
    """
    Load defaults for each component, apply overrides, resolve model-for-dataset,
    and return a fully resolved ExperimentSpec.
    """
    # Load defaults
    trainer_default = load_default("trainers", trainer_name)
    model_default   = load_default("models", model_name)
    dataset_default = load_default("datasets", dataset_name)

    # Apply user overrides on top of defaults
    trainer_cfg = merge(trainer_default, trainer_override)
    model_raw   = merge(model_default, model_override)
    dataset_cfg = merge(dataset_default, dataset_override)

    # Apply per-trainer then per-dataset model section overrides.
    # resolve_model_for_trainer_and_dataset merges: default → trainer → dataset.
    # Falls back to the old per-dataset-only resolver for models that lack trainer sections.
    model_cfg = resolve_model_for_trainer_and_dataset(model_raw, trainer_name, dataset_name)

    # Flatten Optuna attribute dicts for normal runs
    if not opt:
        trainer_cfg = normalize_optuna_attrs(trainer_cfg)
        model_cfg   = normalize_optuna_attrs(model_cfg)
        dataset_cfg = normalize_optuna_attrs(dataset_cfg)

    # Inject names so experiment.py can dispatch
    trainer_cfg["name"] = trainer_name.lower()
    model_cfg["name"]   = model_name.lower()
    dataset_cfg["name"] = dataset_name.lower()

    # Add algorithm name to model cfg so get_network can pick the right arch
    model_cfg["algorithm_name"] = trainer_name.lower()

    name = exp_name or f"{trainer_name.lower()}_{model_name.lower()}_{dataset_name.lower()}"

    return ExperimentSpec(
        name=name,
        opt=opt,
        trainer=trainer_cfg,
        model=model_cfg,
        dataset=dataset_cfg,
        runtime=runtime,
        optuna=optuna or {},
    )
