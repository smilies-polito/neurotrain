#!/usr/bin/env python3
"""
Campaign runner — entry point for all benchmarking and custom experiments.

Usage:
    # Benchmarking mode (compare trainers × models × datasets)
    python3 run_exp_campaign.py --benchmarking config/benchmarking.yaml --name my_bench

    # Custom mode (run user-defined experiments with overrides)
    python3 run_exp_campaign.py --custom config/experiments.yaml --name my_custom

Options:
    --benchmarking PATH   Path to benchmarking YAML config
    --custom       PATH   Path to custom experiments YAML config
    --name         NAME   Campaign name (default: timestamp)
    --output       DIR    Root output directory (default: experiments/)
    --dry-run             Print the experiment list without running anything
"""

import argparse
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent
_SRC  = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from campaign.campaign_builder import from_benchmarking, from_custom
from campaign.experiment_spec import ExperimentSpec
from campaign.results import (
    copy_input_config,
    load_experiment_metrics,
    save_campaign_summary,
)

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SNN benchmarking campaign runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmarking", metavar="PATH", help="Benchmarking config YAML")
    mode.add_argument("--custom",       metavar="PATH", help="Custom experiments YAML")

    parser.add_argument(
        "--name", default=None,
        help="Campaign name. Default: timestamp (YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--output", default="experiments",
        help="Root output directory. Default: experiments/",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the experiment list and exit without running",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable tqdm progress bars during training and evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.root.addFilter(
        type("_TonicCacheFilter", (logging.Filter,), {
            "filter": staticmethod(lambda r: "not in cache" not in r.getMessage())
        })()
    )

    # ── Build experiment list ───────────────────────────────────────────────
    input_path = Path(args.benchmarking or args.custom)
    if not input_path.exists():
        log.error("Config file not found: %s", input_path)
        sys.exit(1)

    log.info("Loading experiments from: %s", input_path)
    if args.benchmarking:
        specs = from_benchmarking(input_path)
    else:
        specs = from_custom(input_path)

    if not specs:
        log.error("No experiments to run. Check your config file.")
        sys.exit(1)

    log.info("Found %d experiment(s) to run.", len(specs))

    # ── Dry run ─────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — {len(specs)} experiment(s):")
        for i, spec in enumerate(specs, 1):
            print(f"  {i:3d}. {spec.name}")
            print(f"       trainer={spec.trainer['name']}  "
                  f"model={spec.model['name']}  "
                  f"dataset={spec.dataset['name']}")
        print(f"{'='*60}\n")
        return

    # ── Create campaign output directory ───────────────────────────────────
    name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_dir = Path(args.output) / name
    campaign_dir.mkdir(parents=True, exist_ok=True)
    copy_input_config(campaign_dir, input_path)
    log.info("Campaign output: %s", campaign_dir)

    # ── Run experiments ─────────────────────────────────────────────────────
    all_metrics: list[dict] = []
    failed: list[str] = []

    for i, spec in enumerate(specs, 1):
        exp_out = campaign_dir / "experiments" / spec.name
        exp_out.mkdir(parents=True, exist_ok=True)

        log.info(
            "[%d/%d] Running: %s  (trainer=%s, model=%s, dataset=%s)",
            i, len(specs), spec.name,
            spec.trainer["name"], spec.model["name"], spec.dataset["name"],
        )

        try:
            spec.runtime["progress"] = not args.no_progress
            _run_inline(spec, exp_out)

            metrics = load_experiment_metrics(exp_out)
            if metrics:
                all_metrics.append(metrics)
            else:
                log.warning("No metrics returned by experiment '%s'.", spec.name)
                failed.append(spec.name)

        except Exception as e:
            log.error("Experiment '%s' failed: %s", spec.name, e)
            failed.append(spec.name)

    # ── Write campaign summary ──────────────────────────────────────────────
    if all_metrics:
        save_campaign_summary(campaign_dir, all_metrics)
        log.info("Summary written to: %s", campaign_dir / "summary.csv")

    # ── Print final report ──────────────────────────────────────────────────
    _print_summary(all_metrics, failed, campaign_dir)


def _run_inline(spec: ExperimentSpec, out_dir: Path) -> None:
    """Run one experiment in the current process."""
    import importlib.util

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        tmp.write(spec.to_json())
        tmp_path = tmp.name

    try:
        spec_mod = importlib.util.spec_from_file_location(
            "experiment", _REPO / "experiment.py"
        )
        mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(mod)
        mod.main(tmp_path, str(out_dir))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _print_summary(
    all_metrics: list[dict],
    failed: list[str],
    campaign_dir: Path,
) -> None:
    """Print a human-readable results table."""
    print(f"\n{'='*70}")
    print(f"Campaign results — {len(all_metrics)} succeeded, {len(failed)} failed")
    print(f"{'='*70}")
    if all_metrics:
        header = f"{'Name':<40} {'Test Acc':>9} {'Train Loss':>11}"
        print(header)
        print("-" * 65)
        for m in all_metrics:
            print(
                f"{m.get('name', '?'):<40} "
                f"{m.get('test_accuracy', 0.0):>9.4f} "
                f"{m.get('train_loss', 0.0):>11.4f}"
            )
    if failed:
        print("\nFailed experiments:")
        for name in failed:
            print(f"  - {name}")
    print(f"\nOutput directory: {campaign_dir.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
