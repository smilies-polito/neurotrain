"""
Result serialization for individual experiments and campaign summaries.

Per-experiment output (written by experiment.py):
    experiments/<campaign>/<exp_name>/
        config.yaml   — resolved ExperimentSpec
        metrics.json  — accuracy, loss, neurobench results
        log.txt       — stdout captured during the run

Campaign-level output (written by run_exp_campaign.py):
    experiments/<campaign>/
        campaign.yaml — copy of the input config file
        summary.json  — all experiments, one dict per experiment
        summary.csv   — flat table suitable for spreadsheet analysis
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from campaign.experiment_spec import ExperimentSpec


def experiment_dir(campaign_dir: Path, exp_name: str) -> Path:
    """Return (and create) the output directory for one experiment."""
    d = campaign_dir / "experiments" / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_experiment_config(exp_dir: Path, spec: ExperimentSpec) -> None:
    """Write the resolved ExperimentSpec as config.yaml."""
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(asdict(spec), f, default_flow_style=False, sort_keys=False)


def save_experiment_metrics(exp_dir: Path, metrics: dict) -> None:
    """Write per-experiment metrics as metrics.json."""
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def load_experiment_metrics(exp_dir: Path) -> dict:
    """Load metrics.json from an experiment directory. Returns {} on failure."""
    path = exp_dir / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_campaign_summary(
    campaign_dir: Path,
    all_metrics: list[dict],
) -> None:
    """
    Write summary.json and summary.csv from a list of per-experiment metric dicts.

    Each dict is expected to have at least:
        name, trainer, model, dataset, test_accuracy, train_loss

    Any neurobench sub-dict is flattened with a "nb_" prefix.
    """
    # Flatten neurobench dict entries for CSV
    flat_rows = [_flatten(m) for m in all_metrics]

    # JSON summary
    with open(campaign_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # CSV summary — collect all possible columns first
    if not flat_rows:
        return
    all_keys = list(dict.fromkeys(k for row in flat_rows for k in row))
    with open(campaign_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def copy_input_config(campaign_dir: Path, input_path: Path) -> None:
    """Copy the input YAML file into the campaign directory as campaign.yaml."""
    dest = campaign_dir / "campaign.yaml"
    dest.write_text(input_path.read_text())


# ------------------------------------------------------------------ #
# Internal helpers                                                      #
# ------------------------------------------------------------------ #

def _flatten(d: dict, prefix: str = "") -> dict:
    """Recursively flatten nested dicts with a dot/underscore prefix."""
    result = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten(v, prefix=f"{key}_"))
        else:
            result[key] = v
    return result
