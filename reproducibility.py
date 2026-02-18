#!/usr/bin/env python3
"""Run reproducibility experiments from per-paper config files.

Flow:
1. Discover all config files under `configs/reproducibility`.
2. Optionally filter to a subset of trainers (`--algorithms`).
3. Run each selected config with the standard training pipeline.
4. Report only the key paper-comparison metrics: accuracy and loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Allow running this file as a script without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from main import get_device, get_trainer, trainable
from utils.checkpoint import CheckpointManager
from utils.config import load_config, validate_config
from utils.experiment_logger import ExperimentLogger


@dataclass
class ReproducibilitySpec:
    """One planned reproducibility run from one config file."""

    experiment_id: str
    config_path: str
    config_name: str
    experiment_name: str
    trainer_name: str
    dataset: str


@dataclass
class ReproducibilityResult:
    """Final metrics for one reproducibility run."""

    experiment_id: str
    config_name: str
    config_path: str
    trainer_name: str
    dataset: str
    status: str
    final_accuracy: float | None
    final_loss: float | None
    total_wall_time_s: float | None
    error: str | None


def _parse_csv_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI value into a non-empty list."""

    if value is None:
        return None
    parts = [chunk.strip() for chunk in value.split(",")]
    parts = [chunk for chunk in parts if chunk]
    return parts or None


def _to_plain_number(value: Any) -> float | None:
    """Convert scalar-like objects to a plain Python float."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_raw_config(path: Path) -> dict[str, Any]:
    """Load a YAML/JSON file as a raw dictionary."""

    with open(path, "r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle) or {}
        elif path.suffix.lower() == ".json":
            payload = json.load(handle) or {}
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


class ReproducibilitySuite:
    """Discover, run, and report reproducibility config experiments."""

    def __init__(
        self,
        configs_dir: Path,
        trainers: list[str] | None = None,
        output_dir: Path = Path("./benchmark_results/reproducibility"),
        epochs_override: int | None = None,
        batch_size_override: int | None = None,
        lr_override: float | None = None,
        timesteps_override: int | None = None,
        device_override: str | None = None,
        seed_override: int | None = None,
        continue_on_error: bool = True,
    ) -> None:
        self.configs_dir = configs_dir
        self.output_dir = output_dir
        self.trainers_filter = {name.lower() for name in trainers} if trainers else None
        self.epochs_override = epochs_override
        self.batch_size_override = batch_size_override
        self.lr_override = lr_override
        self.timesteps_override = timesteps_override
        self.device_override = device_override
        self.seed_override = seed_override
        self.continue_on_error = continue_on_error

        self.config_files: list[Path] = []
        self.valid_experiments: list[ReproducibilitySpec] = []
        self.skipped_experiments: list[dict[str, Any]] = []

    def initialize(self) -> None:
        """Discover config files and build the runnable experiment list."""

        if not self.configs_dir.exists():
            raise FileNotFoundError(f"Reproducibility config directory not found: {self.configs_dir}")
        self.config_files = sorted(
            [
                path
                for path in self.configs_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".json"}
            ]
        )
        self._build_experiment_dictionary()

    def _build_experiment_dictionary(self) -> None:
        """Read config metadata and materialize runnable experiments."""

        for config_path in self.config_files:
            try:
                raw_cfg = _load_raw_config(config_path)
            except Exception as exc:
                self.skipped_experiments.append(
                    {
                        "config": str(config_path),
                        "reason": f"config parse error: {type(exc).__name__}: {exc}",
                    }
                )
                continue

            trainer_name = str(raw_cfg.get("trainer", {}).get("name", "")).strip()
            if not trainer_name:
                self.skipped_experiments.append(
                    {
                        "config": str(config_path),
                        "reason": "missing trainer.name",
                    }
                )
                continue

            if self.trainers_filter and trainer_name.lower() not in self.trainers_filter:
                continue

            dataset = str(raw_cfg.get("data", {}).get("dataset", "unknown"))
            config_name = config_path.stem
            experiment_name = str(raw_cfg.get("experiment", {}).get("name", config_name))
            experiment_id = f"{trainer_name}__{dataset}__{config_name}"

            self.valid_experiments.append(
                ReproducibilitySpec(
                    experiment_id=experiment_id,
                    config_path=str(config_path),
                    config_name=config_name,
                    experiment_name=experiment_name,
                    trainer_name=trainer_name,
                    dataset=dataset,
                )
            )

    def _apply_overrides(self, config) -> None:
        """Apply optional CLI overrides to one loaded config."""

        if self.epochs_override is not None:
            config.training.epochs = int(self.epochs_override)
        if self.batch_size_override is not None:
            config.training.batch_size = int(self.batch_size_override)
        if self.lr_override is not None:
            config.training.learning_rate = float(self.lr_override)
        if self.timesteps_override is not None:
            config.data.timesteps = int(self.timesteps_override)
        if self.device_override is not None:
            config.hardware.device = str(self.device_override)
        if self.seed_override is not None:
            config.experiment.seed = int(self.seed_override)

    def _run_single_experiment(
        self, spec: ReproducibilitySpec, run_dir: Path
    ) -> ReproducibilityResult:
        """Run one reproducibility config and capture final metrics."""

        start_time = time.perf_counter()
        logger: ExperimentLogger | None = None

        try:
            config = load_config(spec.config_path)
            self._apply_overrides(config)

            issues = validate_config(config)
            fatal_issues = [issue for issue in issues if "must" in issue]
            if fatal_issues:
                return ReproducibilityResult(
                    experiment_id=spec.experiment_id,
                    config_name=spec.config_name,
                    config_path=spec.config_path,
                    trainer_name=spec.trainer_name,
                    dataset=spec.dataset,
                    status="failed",
                    final_accuracy=None,
                    final_loss=None,
                    total_wall_time_s=time.perf_counter() - start_time,
                    error="; ".join(fatal_issues),
                )
            if issues:
                print(f"  config warnings for {spec.config_name}:")
                for issue in issues:
                    print(f"    - {issue}")

            # Keep artifacts grouped under this reproducibility suite run.
            config.experiment.log_dir = str(run_dir / spec.config_name)

            logger = ExperimentLogger(
                experiment_name=config.experiment.name,
                config=config.to_dict(),
                seed=config.experiment.seed,
                log_dir=f"{config.experiment.log_dir}/{config.experiment.name}",
                deterministic=config.experiment.deterministic,
            )

            device = get_device(config)
            logger.setup(device)
            logger.save_context()

            checkpoint_manager = CheckpointManager(
                checkpoint_dir=Path(logger.log_dir) / "checkpoints",
                save_best=config.checkpoint.save_best,
                save_latest=config.checkpoint.save_latest,
                save_every=config.checkpoint.save_every,
                max_keep=config.checkpoint.max_keep,
                metric_name="accuracy",
                metric_mode="max",
            )

            trainer_class = get_trainer(config.trainer.name)
            metrics = trainable(
                config=config,
                trainer_class=trainer_class,
                logger=logger,
                checkpoint_manager=checkpoint_manager,
                start_epoch=0,
                resume_checkpoint=None,
            )

            final_accuracy = None
            final_loss = None
            if isinstance(metrics, dict):
                final_accuracy = _to_plain_number(metrics.get("final_accuracy"))
                final_loss = _to_plain_number(metrics.get("final_loss"))
            else:
                final_accuracy = _to_plain_number(metrics)

            return ReproducibilityResult(
                experiment_id=spec.experiment_id,
                config_name=spec.config_name,
                config_path=spec.config_path,
                trainer_name=spec.trainer_name,
                dataset=spec.dataset,
                status="ok",
                final_accuracy=final_accuracy,
                final_loss=final_loss,
                total_wall_time_s=time.perf_counter() - start_time,
                error=None,
            )
        except Exception as exc:
            return ReproducibilityResult(
                experiment_id=spec.experiment_id,
                config_name=spec.config_name,
                config_path=spec.config_path,
                trainer_name=spec.trainer_name,
                dataset=spec.dataset,
                status="failed",
                final_accuracy=None,
                final_loss=None,
                total_wall_time_s=time.perf_counter() - start_time,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if logger is not None:
                try:
                    logger.close()
                except Exception:
                    pass

    def _report_results(self, results: list[ReproducibilityResult], run_dir: Path) -> None:
        """Print and persist reproducibility summary focused on accuracy/loss."""

        def format_metric(value: float | None) -> str:
            if value is None:
                return "N/A"
            return f"{value:.4f}"

        rows = [
            {
                "config": result.config_name,
                "trainer": result.trainer_name,
                "dataset": result.dataset,
                "status": result.status,
                "accuracy": format_metric(result.final_accuracy),
                "loss": format_metric(result.final_loss),
            }
            for result in results
        ]
        headers = {
            "config": "Config",
            "trainer": "Trainer",
            "dataset": "Dataset",
            "status": "Status",
            "accuracy": "Accuracy",
            "loss": "Loss",
        }
        columns = ["config", "trainer", "dataset", "status", "accuracy", "loss"]
        widths = {
            column: max(len(headers[column]), *(len(str(row[column])) for row in rows))
            if rows
            else len(headers[column])
            for column in columns
        }

        print("\n" + "=" * 96)
        print("REPRODUCIBILITY SUMMARY")
        print("=" * 96)
        header_line = " | ".join(headers[column].ljust(widths[column]) for column in columns)
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))
        print("=" * 96)

        csv_path = run_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "experiment_id",
                    "config",
                    "config_path",
                    "trainer",
                    "dataset",
                    "status",
                    "final_accuracy",
                    "final_loss",
                    "total_wall_time_s",
                    "error",
                ]
            )
            for result in results:
                writer.writerow(
                    [
                        result.experiment_id,
                        result.config_name,
                        result.config_path,
                        result.trainer_name,
                        result.dataset,
                        result.status,
                        result.final_accuracy,
                        result.final_loss,
                        result.total_wall_time_s,
                        result.error,
                    ]
                )

        lines = [
            "# Reproducibility Summary",
            "",
            "| Config | Trainer | Dataset | Status | Accuracy | Loss |",
            "|---|---|---|---|---:|---:|",
        ]
        for result in results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        result.config_name,
                        result.trainer_name,
                        result.dataset,
                        result.status,
                        format_metric(result.final_accuracy),
                        format_metric(result.final_loss),
                    ]
                )
                + " |"
            )
        lines.append("")
        (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    def run(self, dry_run: bool = False) -> int:
        """Execute all selected reproducibility configs and write artifacts."""

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"reproducibility_{run_stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "configs_dir": str(self.configs_dir),
            "trainer_filter": sorted(self.trainers_filter) if self.trainers_filter else None,
            "overrides": {
                "epochs": self.epochs_override,
                "batch_size": self.batch_size_override,
                "learning_rate": self.lr_override,
                "timesteps": self.timesteps_override,
                "device": self.device_override,
                "seed": self.seed_override,
            },
            "experiments": [asdict(spec) for spec in self.valid_experiments],
            "skipped": self.skipped_experiments,
        }
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        print(f"Discovered config files: {len(self.config_files)}")
        print(f"Runnable experiments: {len(self.valid_experiments)}")
        print(f"Skipped files: {len(self.skipped_experiments)}")

        if dry_run:
            print(f"Dry run completed. Manifest saved to {run_dir / 'experiment_manifest.json'}")
            return 0

        results: list[ReproducibilityResult] = []
        for idx, spec in enumerate(self.valid_experiments, start=1):
            print(
                f"[{idx}/{len(self.valid_experiments)}] "
                f"{spec.trainer_name} | {spec.dataset} | {Path(spec.config_path).name}"
            )
            result = self._run_single_experiment(spec, run_dir=run_dir)
            results.append(result)

            if result.status == "failed":
                print(f"  failed: {result.error}")
                if not self.continue_on_error:
                    print(f"Stopping early due to failure in {spec.experiment_id}")
                    break

        self._report_results(results, run_dir)

        results_payload = {
            "metadata": {
                "timestamp": run_stamp,
                "run_dir": str(run_dir),
            },
            "results": [asdict(result) for result in results],
            "skipped": self.skipped_experiments,
        }
        (run_dir / "results.json").write_text(
            json.dumps(results_payload, indent=2), encoding="utf-8"
        )

        print(f"Results saved to: {run_dir}")
        return 1 if any(result.status == "failed" for result in results) else 0


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Run reproducibility experiments from configs/reproducibility"
    )
    parser.add_argument("--configs-dir", type=str, default="configs/reproducibility")
    parser.add_argument("--algorithms", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results/reproducibility",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suite = ReproducibilitySuite(
        configs_dir=Path(args.configs_dir),
        trainers=_parse_csv_list(args.algorithms),
        output_dir=Path(args.output_dir),
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        timesteps_override=args.timesteps,
        device_override=args.device,
        seed_override=args.seed,
        continue_on_error=not args.fail_fast,
    )
    suite.initialize()
    return suite.run(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
