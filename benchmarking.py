#!/usr/bin/env python3
"""Run full trainer/network benchmarking from YAML configs.

Flow overview:
1. Load global benchmark configuration (`configs/benchmarking.yaml`).
2. Load all network blueprints (`configs/networks/*.yaml`).
3. Build the full experiment matrix (trainer x network x dataset).
4. Reject invalid combinations using explicit compatibility rules:
   - required/excluded BaseSNN tags,
   - required network attributes,
   - architecture allow-lists.
5. Execute each valid experiment:
   - instantiate network and trainer,
   - train for configured epochs,
   - evaluate accuracy/loss,
   - optionally run NeuroBench metrics.
6. Save all artifacts:
   - manifest of scheduled experiments,
   - JSON/CSV/Markdown summaries,
   - plots (if matplotlib is available).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ============================================================================
# Optional plotting backend
# ============================================================================
try:  # Optional dependency in some environments.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - depends on local env
    plt = None
    _HAS_MATPLOTLIB = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datasets.get_loader import get_loader
from networks.base_snn import BaseSNN
from networks.benchmarking.conv_snn import ConvSNN
from networks.benchmarking.fc_snn import FCSNN
from networks.benchmarking.r_snn import RSNN
from networks.benchmarking.vg11_snn import VG11SNN
from utils.experiment_logger import set_all_seeds
from utils.neurobench_eval import run_neurobench

# ============================================================================
# Core registry/constants
# ============================================================================
_BASE_TAGS = ("fully_connected", "convolutional", "recurrent", "single_layer", "vgg")
_NETWORK_FACTORY = {
    "fc_snn": FCSNN,
    "r_snn": RSNN,
    "conv_snn": ConvSNN,
    "vg11_snn": VG11SNN,
}


@dataclass
class ExperimentSpec:
    """One fully materialized experiment to execute."""

    experiment_id: str
    trainer_name: str
    trainer_module: str
    trainer_class_name: str
    network_name: str
    network_architecture: str
    dataset: str
    tags: list[str]
    trainer_config: dict[str, Any]
    network_config: dict[str, Any]
    dataset_config: dict[str, Any]


@dataclass
class ExperimentResult:
    """Normalized result payload for one experiment run."""

    experiment_id: str
    trainer_name: str
    network_name: str
    network_architecture: str
    dataset: str
    status: str
    final_accuracy: float | None
    final_loss: float | None
    epochs: int
    total_wall_time_s: float | None
    avg_epoch_time_ms: float | None
    error: str | None
    neurobench: dict[str, Any]


def _parse_csv_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI option into a clean list."""

    if value is None:
        return None
    parts = [chunk.strip() for chunk in value.split(",")]
    parts = [chunk for chunk in parts if chunk]
    return parts or None


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries, preserving nested keys."""

    out = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _to_plain_number(value: Any) -> float | None:
    """Convert tensors/scalars into plain Python float when possible."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_prediction(pred: Any) -> torch.Tensor:
    """Normalize trainer predictions to a 1D class-index tensor."""

    if isinstance(pred, torch.Tensor):
        tensor = pred.detach()
    else:
        tensor = torch.as_tensor(pred)

    if tensor.dim() == 0:
        return tensor.unsqueeze(0).long()
    if tensor.dim() == 1:
        return tensor.long()
    if tensor.dim() == 2 and tensor.shape[1] == 1:
        return tensor.squeeze(1).long()
    if tensor.dim() >= 2:
        flat = tensor.reshape(tensor.shape[0], -1)
        return flat.argmax(dim=1).long()
    return tensor.long()


def _format_metric(value: float | None, decimals: int = 4) -> str:
    """Format numeric metric for tables, handling missing values."""

    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _format_seconds(value: float | None) -> str:
    """Format seconds for summary tables."""

    if value is None:
        return "N/A"
    return f"{value:.1f}s"


def _resolve_device(device_name: str) -> torch.device:
    """Resolve user device string to an available torch device."""

    requested = str(device_name).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _optimizer_from_name(name: str | None, params, lr: float):
    """Build optimizer by short name from configuration."""

    if name is None:
        return None
    key = str(name).lower()
    if key == "adam":
        return torch.optim.Adam(params, lr=lr)
    if key == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=False)
    if key == "nag":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if key == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"Unsupported optimizer '{name}'")


def _network_tags(network: BaseSNN) -> set[str]:
    """Extract active semantic tags from BaseSNN boolean properties."""

    tags: set[str] = set()
    for tag in _BASE_TAGS:
        if bool(getattr(network, tag, False)):
            tags.add(tag)
    return tags


class BenchmarkingSuite:
    """Orchestrates discovery, scheduling, execution, and reporting."""

    def __init__(
        self,
        benchmark_config_path: Path,
        networks_dir: Path,
        datasets: list[str] | None = None,
        trainers: list[str] | None = None,
        networks: list[str] | None = None,
        epochs_override: int | None = None,
        batch_size_override: int | None = None,
        lr_override: float | None = None,
        timesteps_override: int | None = None,
        device_override: str | None = None,
        seed_override: int | None = None,
        max_train_batches_override: int | None = None,
        max_test_batches_override: int | None = None,
        run_neurobench: bool = False,
    ) -> None:
        self.benchmark_config_path = benchmark_config_path
        self.networks_dir = networks_dir
        self.datasets_filter = {d.lower() for d in datasets} if datasets else None
        self.trainers_filter = {t.lower() for t in trainers} if trainers else None
        self.networks_filter = {n.lower() for n in networks} if networks else None
        self.epochs_override = epochs_override
        self.batch_size_override = batch_size_override
        self.lr_override = lr_override
        self.timesteps_override = timesteps_override
        self.device_override = device_override
        self.seed_override = seed_override
        self.max_train_batches_override = max_train_batches_override
        self.max_test_batches_override = max_test_batches_override
        self.run_neurobench_override = run_neurobench

        self.config: dict[str, Any] = {}
        self.network_files: list[Path] = []
        self.dataset_defaults: dict[str, Any] = {}
        self.available_trainers: dict[str, tuple[type | None, str | None]] = {}
        self.valid_experiments: list[ExperimentSpec] = []
        self.skipped_experiments: list[dict[str, Any]] = []

    def initialize(self) -> None:
        """Load all inputs and precompute valid/invalid experiment sets."""

        self._load_config()
        self._load_network_files()
        self._resolve_trainers()
        self._build_experiment_dictionary()

    def _load_config(self) -> None:
        """Load benchmark YAML and apply CLI overrides in one place."""

        with open(self.benchmark_config_path, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle) or {}

        execution_cfg = self.config.setdefault("execution", {})
        experiment_cfg = self.config.setdefault("experiment", {})
        data_cfg = self.config.setdefault("data", {})

        if self.epochs_override is not None:
            execution_cfg["epochs"] = int(self.epochs_override)
        if self.batch_size_override is not None:
            execution_cfg["batch_size"] = int(self.batch_size_override)
        if self.lr_override is not None:
            execution_cfg["learning_rate"] = float(self.lr_override)
        if self.timesteps_override is not None:
            execution_cfg["timesteps"] = int(self.timesteps_override)
        if self.device_override is not None:
            execution_cfg["device"] = self.device_override
        if self.seed_override is not None:
            experiment_cfg["seed"] = int(self.seed_override)
        if self.max_train_batches_override is not None:
            execution_cfg["max_train_batches"] = int(self.max_train_batches_override)
        if self.max_test_batches_override is not None:
            execution_cfg["max_test_batches"] = int(self.max_test_batches_override)
        if self.run_neurobench_override:
            execution_cfg["run_neurobench"] = True

        self.dataset_defaults = data_cfg.get("dataset_defaults", {})
        if not data_cfg.get("datasets"):
            if self.dataset_defaults:
                data_cfg["datasets"] = list(self.dataset_defaults.keys())
            else:
                data_cfg["datasets"] = ["MNIST"]

    def _load_network_files(self) -> None:
        """Collect all network config files under `configs/networks`."""

        if not self.networks_dir.exists():
            raise FileNotFoundError(f"Network config directory not found: {self.networks_dir}")
        self.network_files = sorted(
            [
                path
                for path in self.networks_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
            ]
        )

    def _resolve_trainers(self) -> None:
        """Import trainer classes declared in YAML and cache failures."""

        trainer_cfg = self.config.get("trainers", {})
        for trainer_name, spec in trainer_cfg.items():
            if not bool(spec.get("enabled", True)):
                self.available_trainers[trainer_name] = (None, "disabled")
                continue
            if self.trainers_filter and trainer_name.lower() not in self.trainers_filter:
                self.available_trainers[trainer_name] = (None, "filtered_out")
                continue

            module_name = spec.get("module")
            class_name = spec.get("class_name")
            if not module_name or not class_name:
                self.available_trainers[trainer_name] = (
                    None,
                    "missing module/class_name in config",
                )
                continue

            try:
                module = import_module(module_name)
                trainer_class = getattr(module, class_name)
                self.available_trainers[trainer_name] = (trainer_class, None)
            except Exception as exc:  # pragma: no cover - depends on local env
                self.available_trainers[trainer_name] = (
                    None,
                    f"import error: {type(exc).__name__}: {exc}",
                )

    def _selected_datasets(self) -> list[str]:
        """Return dataset list after optional CLI filtering."""

        datasets = self.config.get("data", {}).get("datasets", ["MNIST"])
        if self.datasets_filter:
            datasets = [d for d in datasets if d.lower() in self.datasets_filter]
        return datasets

    def _network_payload_for_dataset(
        self, network_cfg: dict[str, Any], dataset_name: str
    ) -> dict[str, Any]:
        """Apply dataset-specific overrides to a base network config."""

        payload = deepcopy(network_cfg)
        overrides = payload.get("dataset_overrides", {}).get(dataset_name, {})

        payload["model"] = _deep_merge(payload.get("model", {}), overrides.get("model", {}))
        payload["network_kwargs"] = _deep_merge(
            payload.get("network_kwargs", {}), overrides.get("network_kwargs", {})
        )

        ds_defaults = self.dataset_defaults.get(dataset_name, {})
        architecture = payload.get("model", {}).get("architecture")

        model_cfg = payload["model"]
        layer_sizes = list(model_cfg.get("layer_sizes", []))
        if layer_sizes and architecture in ("fc_snn", "r_snn") and "input_size" in ds_defaults:
            layer_sizes[0] = int(ds_defaults["input_size"])
        if layer_sizes and "num_classes" in ds_defaults:
            layer_sizes[-1] = int(ds_defaults["num_classes"])
        model_cfg["layer_sizes"] = layer_sizes

        if "num_classes" in ds_defaults and "num_classes" not in payload["network_kwargs"]:
            payload["network_kwargs"]["num_classes"] = int(ds_defaults["num_classes"])
        if "input_shape" in ds_defaults and "in_shape" not in payload["network_kwargs"]:
            payload["network_kwargs"]["in_shape"] = list(ds_defaults["input_shape"])

        return payload

    def _instantiate_network(self, network_payload: dict[str, Any]) -> BaseSNN:
        """Instantiate one benchmarking network from normalized payload."""

        architecture = network_payload.get("model", {}).get("architecture")
        if architecture not in _NETWORK_FACTORY:
            raise ValueError(
                f"Unsupported benchmarking architecture '{architecture}'. "
                f"Expected one of {sorted(_NETWORK_FACTORY.keys())}"
            )
        cls = _NETWORK_FACTORY[architecture]
        kwargs = deepcopy(network_payload.get("network_kwargs", {}))
        return cls(**kwargs)

    def _compatibility_reasons(
        self,
        trainer_name: str,
        trainer_spec: dict[str, Any],
        architecture: str,
        tags: set[str],
        network: BaseSNN,
    ) -> list[str]:
        """Return all reasons why a trainer/network pair should be skipped."""

        reasons: list[str] = []

        required_tags = set(trainer_spec.get("requires_all_tags", []))
        missing_tags = sorted(required_tags - tags)
        if missing_tags:
            reasons.append(f"missing required tags: {missing_tags}")

        excluded_tags = set(trainer_spec.get("excludes_any_tags", []))
        violating = sorted(excluded_tags & tags)
        if violating:
            reasons.append(f"contains excluded tags: {violating}")

        allowed_architectures = trainer_spec.get("allowed_architectures", [])
        if allowed_architectures and architecture not in allowed_architectures:
            reasons.append(
                f"architecture '{architecture}' not allowed ({allowed_architectures})"
            )

        missing_attrs = [
            name
            for name in trainer_spec.get("requires_network_attrs", [])
            if not hasattr(network, name)
        ]
        if missing_attrs:
            reasons.append(f"missing required attrs: {missing_attrs}")

        if trainer_name not in self.available_trainers:
            reasons.append("trainer is not listed in configuration")

        return reasons

    def _build_experiment_dictionary(self) -> None:
        """Build full valid experiment list and explicit skip list.

        This is the main scheduling step:
        - iterate each network YAML,
        - apply per-dataset overrides,
        - probe network tags/attributes,
        - test every trainer against compatibility rules.
        """

        datasets = self._selected_datasets()
        trainer_cfgs: dict[str, Any] = self.config.get("trainers", {})

        # Outer loop: network configs define model blueprints.
        for network_file in self.network_files:
            with open(network_file, "r", encoding="utf-8") as handle:
                raw_network_cfg = yaml.safe_load(handle) or {}

            network_name = raw_network_cfg.get("name", network_file.stem)
            if not bool(raw_network_cfg.get("enabled", True)):
                continue
            if self.networks_filter and network_name.lower() not in self.networks_filter:
                continue

            # Middle loop: materialize each blueprint on each target dataset.
            for dataset_name in datasets:
                payload = self._network_payload_for_dataset(raw_network_cfg, dataset_name)
                try:
                    # Probe network once so we can inspect tags/attrs before scheduling.
                    probe_network = self._instantiate_network(payload)
                except Exception as exc:
                    self.skipped_experiments.append(
                        {
                            "trainer": "*",
                            "network": network_name,
                            "dataset": dataset_name,
                            "reason": f"network init failed: {type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                tags = _network_tags(probe_network)
                architecture = payload.get("model", {}).get("architecture", "unknown")

                # Inner loop: test every trainer against this probed network.
                for trainer_name, trainer_spec in trainer_cfgs.items():
                    trainer_class, trainer_status = self.available_trainers.get(
                        trainer_name, (None, "missing")
                    )
                    if trainer_class is None:
                        if trainer_status not in ("filtered_out", "disabled"):
                            self.skipped_experiments.append(
                                {
                                    "trainer": trainer_name,
                                    "network": network_name,
                                    "dataset": dataset_name,
                                    "reason": trainer_status,
                                }
                            )
                        continue

                    reasons = self._compatibility_reasons(
                        trainer_name=trainer_name,
                        trainer_spec=trainer_spec,
                        architecture=architecture,
                        tags=tags,
                        network=probe_network,
                    )

                    if reasons:
                        self.skipped_experiments.append(
                            {
                                "trainer": trainer_name,
                                "network": network_name,
                                "dataset": dataset_name,
                                "reason": "; ".join(reasons),
                            }
                        )
                        continue

                    # All checks passed: register concrete runnable experiment.
                    experiment_id = f"{dataset_name}__{trainer_name}__{network_name}"
                    self.valid_experiments.append(
                        ExperimentSpec(
                            experiment_id=experiment_id,
                            trainer_name=trainer_name,
                            trainer_module=trainer_spec["module"],
                            trainer_class_name=trainer_spec["class_name"],
                            network_name=network_name,
                            network_architecture=architecture,
                            dataset=dataset_name,
                            tags=sorted(tags),
                            trainer_config=deepcopy(trainer_spec),
                            network_config=deepcopy(payload),
                            dataset_config=deepcopy(self.dataset_defaults.get(dataset_name, {})),
                        )
                    )

    def _build_data_loaders(
        self,
        dataset_name: str,
        batch_size: int,
        timesteps: int,
        device: torch.device,
        flatten_inputs: bool,
        use_raw_loader: bool,
        seed: int,
    ):
        """Build dataset loaders with optional single-process fallback.

        Some CI/HPC sandbox environments block multiprocessing queues, so this
        method can force `num_workers=0` via `single_process_data_loading`.
        """

        train_loader, test_loader = get_loader(
            dataset_name,
            batch_size,
            timesteps,
            flatten=flatten_inputs,
            device=device,
            seed=seed,
            raw_for_local_classifier=use_raw_loader,
        )

        if bool(self.config.get("execution", {}).get("single_process_data_loading", True)):
            train_loader = self._clone_loader_single_worker(train_loader, shuffle=True)
            test_loader = self._clone_loader_single_worker(test_loader, shuffle=False)

        return train_loader, test_loader

    @staticmethod
    def _clone_loader_single_worker(loader: DataLoader, shuffle: bool) -> DataLoader:
        """Recreate a DataLoader with `num_workers=0` preserving key settings."""

        kwargs: dict[str, Any] = {
            "dataset": loader.dataset,
            "batch_size": loader.batch_size,
            "shuffle": shuffle,
            "num_workers": 0,
            "pin_memory": bool(getattr(loader, "pin_memory", False)),
            "drop_last": bool(getattr(loader, "drop_last", False)),
            "collate_fn": loader.collate_fn,
        }
        generator = getattr(loader, "generator", None)
        if generator is not None and shuffle:
            kwargs["generator"] = generator
        return DataLoader(**kwargs)

    @staticmethod
    def _train_one_epoch(
        trainer,
        train_loader,
        device: torch.device,
        max_batches: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, float]:
        """Train exactly one epoch and return aggregate loss/accuracy."""

        trainer.network.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        non_blocking = device.type == "cuda"
        expected_batches: int | None = None
        try:
            expected_batches = int(len(train_loader))
        except TypeError:
            expected_batches = None
        if expected_batches is not None and max_batches is not None:
            expected_batches = min(expected_batches, int(max_batches))
        if progress_callback is not None and expected_batches is not None:
            progress_callback(0, expected_batches)

        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            # Source loaders emit [B, T, ...]; trainers expect [T, B, ...].
            data = data.transpose(0, 1).to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            batch_size = int(data.shape[1])

            trainer.reset()
            loss, pred = trainer.train_sample(data, target)

            # Different trainers return different prediction shapes; normalize first.
            pred_labels = _extract_prediction(pred).to(target.device)
            if pred_labels.shape[0] != target.shape[0]:
                pred_labels = pred_labels[: target.shape[0]]

            loss_value = _to_plain_number(loss)
            if loss_value is not None:
                total_loss += loss_value * batch_size

            total_correct += int(pred_labels.eq(target).sum().item())
            total_samples += batch_size

            if progress_callback is not None and expected_batches is not None:
                progress_callback(batch_idx, expected_batches)

            if max_batches is not None and batch_idx >= max_batches:
                break

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    @staticmethod
    @torch.no_grad()
    def _evaluate(
        network: BaseSNN,
        test_loader,
        device: torch.device,
        max_batches: int | None = None,
    ) -> float:
        """Evaluate accuracy over test loader with temporal readout accumulation."""

        network.eval()
        correct = 0
        total = 0
        non_blocking = device.type == "cuda"
        use_constant_input = getattr(network, "constant_input_per_timestep", False)

        for batch_idx, (data, target) in enumerate(test_loader, start=1):
            data = data.transpose(0, 1).to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)

            if use_constant_input:
                # Local-classifier family expects static repeated current per timestep.
                x_const = data.mean(dim=0)
                if (
                    not getattr(network, "uses_raw_input", False)
                    and x_const.dim() == 2
                    and x_const.shape[1] == 784
                ):
                    x_const = (x_const * 0.3081 + 0.1307).clamp(0.0, 1.0)

            network.reset()
            readout_sum = None
            for t in range(data.shape[0]):
                # Standard temporal readout: sum final-layer readout over timesteps.
                x_t = x_const if use_constant_input else data[t]
                out = network(x_t)
                if isinstance(out, (tuple, list)):
                    if (
                        len(out) >= 2
                        and isinstance(out[1], (list, tuple))
                        and len(out[1]) > 0
                    ):
                        readout = out[1][-1]
                    else:
                        readout = out[0][-1]
                else:
                    readout = out
                readout_sum = readout if readout_sum is None else (readout_sum + readout)

            preds = readout_sum.argmax(dim=1)
            correct += int(preds.eq(target).sum().item())
            total += int(target.shape[0])

            if max_batches is not None and batch_idx >= max_batches:
                break

        return correct / total if total > 0 else 0.0

    @staticmethod
    def _trainer_kwargs(
        trainer_class,
        trainer_spec: dict[str, Any],
        network: BaseSNN,
        lr: float,
        batch_size: int,
        quantization: bool,
    ) -> dict[str, Any]:
        """Assemble trainer kwargs and drop keys unsupported by constructor."""

        training_cfg = trainer_spec.get("training", {})
        optimizer_name = training_cfg.get("optimizer")
        use_optimizer = bool(training_cfg.get("use_optimizer", False))

        optimizer = None
        if use_optimizer and optimizer_name is not None:
            optimizer = _optimizer_from_name(optimizer_name, network.parameters(), lr)

        candidate_kwargs: dict[str, Any] = {
            "network": network,
            "lr": lr,
            "batch_size": batch_size,
            "quant": quantization,
            "use_optimizer": use_optimizer,
            "optimizer": optimizer,
        }
        candidate_kwargs.update(trainer_spec.get("params", {}))

        init_sig = signature(trainer_class.__init__)
        accepts_var_kwargs = any(
            p.kind.name == "VAR_KEYWORD" for p in init_sig.parameters.values()
        )

        filtered: dict[str, Any] = {}
        for key, value in candidate_kwargs.items():
            if key in init_sig.parameters and key != "self":
                filtered[key] = value
            elif accepts_var_kwargs:
                filtered[key] = value

        return filtered

    def _run_single_experiment(self, spec: ExperimentSpec, device: torch.device) -> ExperimentResult:
        """Execute one scheduled experiment end-to-end."""

        trainer_class, trainer_status = self.available_trainers.get(spec.trainer_name, (None, None))
        if trainer_class is None:
            return ExperimentResult(
                experiment_id=spec.experiment_id,
                trainer_name=spec.trainer_name,
                network_name=spec.network_name,
                network_architecture=spec.network_architecture,
                dataset=spec.dataset,
                status="skipped",
                final_accuracy=None,
                final_loss=None,
                epochs=0,
                total_wall_time_s=None,
                avg_epoch_time_ms=None,
                error=trainer_status,
                neurobench={},
            )

        execution_cfg = self.config.get("execution", {})
        experiment_cfg = self.config.get("experiment", {})

        epochs = int(execution_cfg.get("epochs", 1))
        batch_size = int(execution_cfg.get("batch_size", 128))
        default_timesteps = int(execution_cfg.get("timesteps", 25))
        lr = float(execution_cfg.get("learning_rate", 1e-3))
        timesteps = int(spec.dataset_config.get("timesteps", default_timesteps))
        seed = int(experiment_cfg.get("seed", 42))
        max_train_batches = execution_cfg.get("max_train_batches")
        max_test_batches = execution_cfg.get("max_test_batches")
        show_epoch_progress = bool(execution_cfg.get("show_epoch_progress", True))
        max_train_batches = (
            int(max_train_batches) if max_train_batches is not None else None
        )
        max_test_batches = int(max_test_batches) if max_test_batches is not None else None

        # Keep comparability across trainers by resetting RNG before each run.
        set_all_seeds(seed, deterministic=bool(experiment_cfg.get("deterministic", True)))

        network = self._instantiate_network(spec.network_config)
        network = network.to(device)

        flatten_inputs = not bool(getattr(network, "convolutional", False))
        use_raw_loader = spec.trainer_name in ("ell", "fell", "bell") and spec.dataset == "MNIST"

        train_loader, test_loader = self._build_data_loaders(
            dataset_name=spec.dataset,
            batch_size=batch_size,
            timesteps=timesteps,
            device=device,
            flatten_inputs=flatten_inputs,
            use_raw_loader=use_raw_loader,
            seed=seed,
        )

        trainer_kwargs = self._trainer_kwargs(
            trainer_class=trainer_class,
            trainer_spec=spec.trainer_config,
            network=network,
            lr=lr,
            batch_size=batch_size,
            quantization=bool(spec.network_config.get("model", {}).get("quantization", False)),
        )

        # Trainer is built only after filtered kwargs are known.
        trainer = trainer_class(**trainer_kwargs).to(device)

        requires_grad = bool(
            spec.trainer_config.get("training", {}).get("requires_grad", False)
        )

        epoch_times_ms: list[float] = []
        final_loss: float | None = None
        final_acc: float | None = None
        run_neurobench = bool(execution_cfg.get("run_neurobench", False))
        neurobench_payload: dict[str, Any] = {}

        start_time = time.perf_counter()
        try:
            with torch.set_grad_enabled(requires_grad):
                for epoch_idx in range(epochs):
                    use_live_progress = show_epoch_progress and sys.stdout.isatty()
                    progress_line_width = 0
                    last_rendered_percent = -1

                    def _render_epoch_progress(batch_idx: int, total_batches: int) -> None:
                        nonlocal progress_line_width, last_rendered_percent
                        if not use_live_progress or total_batches <= 0:
                            return
                        bar_width = 28
                        filled = min(
                            bar_width, int(round((batch_idx / total_batches) * bar_width))
                        )
                        percent = int(round((batch_idx / total_batches) * 100.0))
                        percent = max(0, min(100, percent))
                        # Avoid costly terminal refreshes when text would be identical.
                        if percent == last_rendered_percent:
                            return
                        last_rendered_percent = percent
                        bar = "#" * filled + "-" * (bar_width - filled)
                        line = (
                            f"  Epoch {epoch_idx + 1}/{epochs} "
                            f"[{bar}] {percent:3d}%"
                        )
                        progress_line_width = max(progress_line_width, len(line))
                        print("\r" + line, end="", flush=True)

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    epoch_start = time.perf_counter()

                    try:
                        metrics = self._train_one_epoch(
                            trainer,
                            train_loader,
                            device,
                            max_batches=max_train_batches,
                            progress_callback=_render_epoch_progress,
                        )
                    finally:
                        if use_live_progress and progress_line_width > 0:
                            # Remove the transient progress line so final output
                            # only keeps the concise epoch metric line.
                            print(
                                "\r" + (" " * progress_line_width) + "\r",
                                end="",
                                flush=True,
                            )

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    epoch_end = time.perf_counter()
                    epoch_times_ms.append((epoch_end - epoch_start) * 1000.0)

                    final_loss = float(metrics["loss"])
                    final_acc = float(
                        self._evaluate(
                            network,
                            test_loader,
                            device,
                            max_batches=max_test_batches,
                        )
                    )
                    print(
                        f"  Epoch {epoch_idx + 1}/{epochs} | "
                        f"train_acc={metrics['accuracy']:.4f} | "
                        f"test_acc={final_acc:.4f} | "
                        f"loss={final_loss:.4f}"
                    )

            if run_neurobench:
                try:
                    # NeuroBench is optional and does not block core benchmark output.
                    nb = run_neurobench(
                        network=network,
                        test_loader=test_loader,
                        device=str(device),
                        num_timesteps=timesteps,
                    )
                    neurobench_payload = self._jsonify(nb)
                except Exception as nb_exc:  # pragma: no cover - optional path
                    neurobench_payload = {"error": f"{type(nb_exc).__name__}: {nb_exc}"}

        except Exception as exc:
            # Return structured failure payload instead of crashing the full suite.
            return ExperimentResult(
                experiment_id=spec.experiment_id,
                trainer_name=spec.trainer_name,
                network_name=spec.network_name,
                network_architecture=spec.network_architecture,
                dataset=spec.dataset,
                status="failed",
                final_accuracy=final_acc,
                final_loss=final_loss,
                epochs=epochs,
                total_wall_time_s=time.perf_counter() - start_time,
                avg_epoch_time_ms=(sum(epoch_times_ms) / len(epoch_times_ms)) if epoch_times_ms else None,
                error=f"{type(exc).__name__}: {exc}",
                neurobench={
                    "traceback": traceback.format_exc(limit=8),
                },
            )

        total_wall = time.perf_counter() - start_time
        avg_ms = sum(epoch_times_ms) / len(epoch_times_ms) if epoch_times_ms else None

        return ExperimentResult(
            experiment_id=spec.experiment_id,
            trainer_name=spec.trainer_name,
            network_name=spec.network_name,
            network_architecture=spec.network_architecture,
            dataset=spec.dataset,
            status="ok",
            final_accuracy=final_acc,
            final_loss=final_loss,
            epochs=epochs,
            total_wall_time_s=total_wall,
            avg_epoch_time_ms=avg_ms,
            error=None,
            neurobench=neurobench_payload,
        )

    @staticmethod
    def _jsonify(value: Any) -> Any:
        """Recursively convert tensors/custom objects into JSON-safe values."""

        if isinstance(value, dict):
            return {k: BenchmarkingSuite._jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [BenchmarkingSuite._jsonify(v) for v in value]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _print_summary_table(results: list[ExperimentResult]) -> None:
        """Render compact console table for quick run inspection."""

        rows = []
        for result in results:
            rows.append(
                {
                    "dataset": result.dataset,
                    "trainer": result.trainer_name,
                    "network": result.network_name,
                    "status": result.status,
                    "acc": _format_metric(result.final_accuracy, 4),
                    "loss": _format_metric(result.final_loss, 4),
                    "wall": _format_seconds(result.total_wall_time_s),
                    "epoch": _format_metric(result.avg_epoch_time_ms, 1),
                }
            )

        headers = {
            "dataset": "Dataset",
            "trainer": "Trainer",
            "network": "Network",
            "status": "Status",
            "acc": "Acc",
            "loss": "Loss",
            "wall": "Wall",
            "epoch": "Epoch ms",
        }
        cols = ["dataset", "trainer", "network", "status", "acc", "loss", "wall", "epoch"]
        widths = {
            col: max(len(headers[col]), *(len(str(row[col])) for row in rows)) if rows else len(headers[col])
            for col in cols
        }

        print("\n" + "=" * 120)
        print("BENCHMARK SUMMARY")
        print("=" * 120)
        header = " | ".join(headers[col].ljust(widths[col]) for col in cols)
        print(header)
        print("-" * len(header))
        for row in rows:
            print(" | ".join(str(row[col]).ljust(widths[col]) for col in cols))
        print("=" * 120)

    @staticmethod
    def _save_csv(results: list[ExperimentResult], path: Path) -> None:
        """Write flat CSV export for downstream analysis."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "experiment_id",
                    "dataset",
                    "trainer",
                    "network",
                    "architecture",
                    "status",
                    "final_accuracy",
                    "final_loss",
                    "epochs",
                    "total_wall_time_s",
                    "avg_epoch_time_ms",
                    "error",
                ]
            )
            for result in results:
                writer.writerow(
                    [
                        result.experiment_id,
                        result.dataset,
                        result.trainer_name,
                        result.network_name,
                        result.network_architecture,
                        result.status,
                        result.final_accuracy,
                        result.final_loss,
                        result.epochs,
                        result.total_wall_time_s,
                        result.avg_epoch_time_ms,
                        result.error,
                    ]
                )

    @staticmethod
    def _save_markdown(
        results: list[ExperimentResult],
        skipped: list[dict[str, Any]],
        path: Path,
    ) -> None:
        """Write human-readable Markdown summary with skipped reasons."""

        lines = []
        lines.append("# Benchmark Summary")
        lines.append("")
        lines.append("## Results")
        lines.append("")
        lines.append("| Dataset | Trainer | Network | Status | Accuracy | Loss | Wall Time |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for result in results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        result.dataset,
                        result.trainer_name,
                        result.network_name,
                        result.status,
                        _format_metric(result.final_accuracy, 4),
                        _format_metric(result.final_loss, 4),
                        _format_seconds(result.total_wall_time_s),
                    ]
                )
                + " |"
            )

        lines.append("")
        lines.append("## Skipped Combinations")
        lines.append("")
        if skipped:
            lines.append("| Trainer | Network | Dataset | Reason |")
            lines.append("|---|---|---|---|")
            for item in skipped:
                lines.append(
                    f"| {item.get('trainer', '')} | {item.get('network', '')} | "
                    f"{item.get('dataset', '')} | {item.get('reason', '')} |"
                )
        else:
            lines.append("No combinations were skipped.")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _save_plots(results: list[ExperimentResult], output_dir: Path) -> None:
        """Save summary visualizations (bar + per-dataset heatmaps)."""

        if not _HAS_MATPLOTLIB:
            return
        ok_results = [result for result in results if result.status == "ok" and result.final_accuracy is not None]
        if not ok_results:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        labels = [
            f"{result.dataset}\n{result.trainer_name}\n{result.network_name}"
            for result in ok_results
        ]
        accuracies = [float(result.final_accuracy) for result in ok_results]

        fig_width = max(10, len(labels) * 0.7)
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        bars = ax.bar(range(len(labels)), accuracies, color="#2b8cbe")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Final Accuracy")
        ax.set_title("Final Accuracy by Experiment")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right")
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(idx, min(0.98, height + 0.01), f"{height:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "accuracy_bar.png", dpi=180)
        plt.close(fig)

        datasets = sorted({result.dataset for result in ok_results})
        for dataset in datasets:
            subset = [result for result in ok_results if result.dataset == dataset]
            trainers = sorted({result.trainer_name for result in subset})
            networks = sorted({result.network_name for result in subset})
            matrix = np.full((len(trainers), len(networks)), np.nan, dtype=np.float32)

            for result in subset:
                i = trainers.index(result.trainer_name)
                j = networks.index(result.network_name)
                matrix[i, j] = float(result.final_accuracy)

            fig, ax = plt.subplots(figsize=(max(6, len(networks) * 1.6), max(4, len(trainers) * 0.8)))
            cmap = matplotlib.colormaps.get_cmap("viridis").copy()
            cmap.set_bad(color="#f0f0f0")
            image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap)
            ax.set_xticks(range(len(networks)))
            ax.set_xticklabels(networks, rotation=45, ha="right")
            ax.set_yticks(range(len(trainers)))
            ax.set_yticklabels(trainers)
            ax.set_title(f"Accuracy Heatmap - {dataset}")
            for i in range(len(trainers)):
                for j in range(len(networks)):
                    value = matrix[i, j]
                    if not np.isnan(value):
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(output_dir / f"accuracy_heatmap_{dataset}.png", dpi=180)
            plt.close(fig)

    def run(self, dry_run: bool = False) -> int:
        """Run the complete benchmark suite and persist all artifacts."""

        execution_cfg = self.config.get("execution", {})
        experiment_cfg = self.config.get("experiment", {})

        # Each invocation gets a timestamped output folder.
        out_root = Path(experiment_cfg.get("output_dir", "./benchmark_results"))
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_root / f"full_benchmark_{run_stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        device = _resolve_device(execution_cfg.get("device", "cuda"))
        print(f"Device: {device}")
        print(f"Valid experiments: {len(self.valid_experiments)}")
        print(f"Skipped combinations: {len(self.skipped_experiments)}")

        # Save planned schedule before execution for full reproducibility.
        manifest = {
            "config": self.config,
            "experiments": [asdict(spec) for spec in self.valid_experiments],
            "skipped": self.skipped_experiments,
            "device": str(device),
        }
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        if dry_run:
            print(f"Dry run completed. Manifest saved to {run_dir / 'experiment_manifest.json'}")
            return 0

        results: list[ExperimentResult] = []
        continue_on_error = bool(execution_cfg.get("continue_on_error", True))

        # Execute experiments sequentially in deterministic order.
        for idx, spec in enumerate(self.valid_experiments, start=1):
            print(
                f"[{idx}/{len(self.valid_experiments)}] "
                f"{spec.dataset} | {spec.trainer_name} | {spec.network_name}"
            )

            result = self._run_single_experiment(spec, device=device)
            results.append(result)
            if result.status == "failed":
                print(f"  failed: {result.error}")

            if result.status == "failed" and not continue_on_error:
                print(f"Stopping early due to failure in {result.experiment_id}")
                break

        # Console summary first, then persistent artifacts.
        self._print_summary_table(results)

        results_payload = {
            "metadata": {
                "timestamp": run_stamp,
                "device": str(device),
                "pytorch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "seed": experiment_cfg.get("seed", 42),
            },
            "results": [asdict(result) for result in results],
            "skipped": self.skipped_experiments,
        }
        (run_dir / "results.json").write_text(
            json.dumps(results_payload, indent=2), encoding="utf-8"
        )

        self._save_csv(results, run_dir / "results.csv")
        self._save_markdown(results, self.skipped_experiments, run_dir / "summary.md")
        self._save_plots(results, run_dir)

        print(f"Results saved to: {run_dir}")
        return 0


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Run full SNN trainer/network benchmark suite")
    # Files / config roots.
    parser.add_argument("--config", type=str, default="configs/benchmarking.yaml")
    parser.add_argument("--networks-dir", type=str, default="configs/networks")
    # Global execution overrides.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # Optional filters for matrix dimensions.
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--algorithms", type=str, default=None)
    parser.add_argument("--networks", type=str, default=None)
    # Optional extras.
    parser.add_argument("--run-neurobench", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suite = BenchmarkingSuite(
        benchmark_config_path=Path(args.config),
        networks_dir=Path(args.networks_dir),
        datasets=_parse_csv_list(args.datasets),
        trainers=_parse_csv_list(args.algorithms),
        networks=_parse_csv_list(args.networks),
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        timesteps_override=args.timesteps,
        device_override=args.device,
        seed_override=args.seed,
        max_train_batches_override=args.max_train_batches,
        max_test_batches_override=args.max_test_batches,
        run_neurobench=args.run_neurobench,
    )

    suite.initialize()
    return suite.run(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
