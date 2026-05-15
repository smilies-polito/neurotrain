#!/usr/bin/env python3
"""
make_res_heatmap.py — Build the paper results heatmap by parsing the values
directly from paper_experiments/README.md.

Whenever you update a number in the README tables, just re-run this script
and the heatmap will reflect the new values automatically.

Usage:
    python3 paper_experiments/make_res_heatmap.py [--output DIR] [--format png|svg|pdf]
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker

_REPO = Path(__file__).resolve().parent.parent
_SRC  = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from generate_results import (
    display_trainer, display_model, display_dataset,
)


# ── Name maps (README display -> internal key) ─────────────────────────────

TRAINER_KEY = {
    "BPTT":     "bptt",
    "DECOLLE":  "decolle",
    "EPROP":    "eprop",
    "ESD_RTRL": "es_d_rtrl",
    "ETLP":     "etlp",
    "OSTL":     "ostl",
    "OSTTP":    "osttp",
    "OTTT":     "ottt",
    "STSF":     "stsf",
    "TP":       "tp",
}

DATASET_KEY = {
    "MNIST":       "mnist",
    "F-MNIST":     "fashionmnist",
    "CIFAR10":     "cifar10",
    "SVHN":        "svhn",
    "NMNIST":      "nmnist",
    "DVSGest.":    "dvsgesture",
    "DVSCifar10":  "dvscifar10",
    "SHD":         "shd",
}

MODEL_KEY = {
    "FC":   "fc_snn",
    "RC":   "r_snn",
    "Conv": "conv_snn",
}

# Matches the first decimal number in a table cell, e.g. "0.978 🟢", "0.315* 🟢 [1]"
_NUMBER_RE = re.compile(r"\d+\.\d+")


def _parse_cell(cell: str) -> float | None:
    """Return the accuracy float from a cell, or None if not available (⚫/🔴)."""
    m = _NUMBER_RE.search(cell)
    return float(m.group()) if m else None


def _strip(cell: str) -> str:
    return cell.strip()


# ── README parser ──────────────────────────────────────────────────────────

def parse_readme(readme_path: Path) -> pd.DataFrame:
    """
    Walk through the README and extract per-trainer result tables.

    Expected structure:
        ## TRAINER_NAME
        ...optional prose...
        | Network | MNIST | F-MNIST | ... |
        | ------- | ...                   |   <- separator, skipped
        | FC      | 0.978 🟢 | ...        |
        ...
    """
    text = readme_path.read_text()
    records: list[tuple] = []

    current_trainer: str | None = None
    dataset_cols: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # ── Trainer section header (## NAME) ──────────────────────────────
        h2 = re.match(r"^##\s+(\S+)", line)
        if h2:
            name = h2.group(1).upper()
            current_trainer = TRAINER_KEY.get(name)
            dataset_cols = []
            continue

        # Skip lines outside recognised trainer sections
        if current_trainer is None:
            continue

        # ── Markdown table rows ───────────────────────────────────────────
        if not line.startswith("|"):
            continue

        cols = [_strip(c) for c in line.split("|")[1:-1]]

        # Header row: "| Network | MNIST | F-MNIST | ..."
        if cols and cols[0].upper() == "NETWORK":
            dataset_cols = [DATASET_KEY.get(c) for c in cols[1:]]
            continue

        # Separator row: "| ------- | :---: | ..."
        if cols and re.match(r"^[-:]+$", cols[0]):
            continue

        # Data row: "| FC | 0.978 🟢 | ..."
        model_key = MODEL_KEY.get(cols[0]) if cols else None
        if model_key is None or not dataset_cols:
            continue

        for dataset_key, cell in zip(dataset_cols, cols[1:]):
            if dataset_key is None:
                continue
            acc = _parse_cell(cell)
            if acc is not None:
                records.append((current_trainer, model_key, dataset_key, acc))

    if not records:
        sys.exit("ERROR: No results parsed from README. Check the file structure.")

    return pd.DataFrame(records, columns=["trainer", "model", "dataset", "test_accuracy"])


# ── Paper heatmap ──────────────────────────────────────────────────────────

_FONT_DATASET_TITLE = 19
_FONT_AXIS_LABEL    = 15
_FONT_CELL_VALUE    = 14
_FONT_COLORBAR      = 17

_NAN_COLOR = "#d0d0d0"   # neutral gray for unsupported / failed cells


def make_paper_heatmap(df: pd.DataFrame, output_path: Path, min_acc: float = 0.0) -> None:
    """
    Paper-optimised heatmap: no figure title, horizontal model labels,
    tight subplot spacing, and larger fonts throughout.
    NaN cells are filled with a neutral gray so they are clearly distinct
    from low-accuracy cells. The color scale floor is raised to the actual
    minimum present in the data so the full gradient is used.
    """
    datasets = sorted(df["dataset"].unique())
    trainers = sorted(df["trainer"].unique())
    models   = sorted(df["model"].unique())

    trainer_labels = [display_trainer(t) for t in trainers]
    model_labels   = [display_model(m)   for m in models]

    n_datasets = len(datasets)
    ncols = min(n_datasets, 4)
    nrows = math.ceil(n_datasets / ncols)

    cell_w = 2.1
    cell_h = 1.15

    fig_w = max(cell_w * len(models) * ncols + 2.5, 16)
    fig_h = max(cell_h * len(trainers) * nrows + 2.2, 10)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.ravel()

    # Raise the color floor to the lowest value actually in the data so the
    # full green gradient maps to the range that actually appears.
    vmin = max(min_acc, float(df["test_accuracy"].min()) - 0.02)
    vmin = max(0.0, round(vmin * 10) / 10)   # snap to nearest 0.1

    cmap = plt.cm.YlGn
    norm = mcolors.Normalize(vmin=vmin, vmax=1.0)

    # Build a version of the colormap that renders NaN as gray
    cmap_with_nan = plt.cm.YlGn.copy()
    cmap_with_nan.set_bad(color=_NAN_COLOR)

    im = None

    for i, dataset in enumerate(datasets):
        ax  = axes_flat[i]
        sub = df[df["dataset"] == dataset]

        mat = np.full((len(trainers), len(models)), np.nan)
        for ri, trainer in enumerate(trainers):
            for ci, model in enumerate(models):
                cell = sub[(sub["trainer"] == trainer) & (sub["model"] == model)]
                if not cell.empty:
                    vals = cell["test_accuracy"].dropna()
                    if len(vals) > 0:
                        mat[ri, ci] = vals.mean()

        im = ax.imshow(mat, cmap=cmap_with_nan, norm=norm, aspect="auto")

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(
            model_labels,
            rotation=0,
            ha="center",
            fontsize=_FONT_AXIS_LABEL,
            fontweight="semibold",
        )

        # Trainer labels only on the leftmost subplot of each row
        if i % ncols == 0:
            ax.set_yticks(range(len(trainers)))
            ax.set_yticklabels(
                trainer_labels,
                fontsize=_FONT_AXIS_LABEL,
                fontweight="semibold",
            )
        else:
            ax.set_yticks(range(len(trainers)))
            ax.set_yticklabels([])

        ax.set_title(
            display_dataset(dataset),
            fontsize=_FONT_DATASET_TITLE,
            fontweight="bold",
            pad=9,
        )

        # Subtle white grid between cells
        ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(trainers), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Cell value annotations
        for ri in range(len(trainers)):
            for ci in range(len(models)):
                val = mat[ri, ci]
                if not np.isnan(val):
                    # Switch to white text once the background is dark enough
                    text_color = "white" if val > (vmin + (1.0 - vmin) * 0.6) else "#1a1a1a"
                    ax.text(
                        ci, ri, f"{val * 100:.1f}%",
                        ha="center", va="center",
                        fontsize=_FONT_CELL_VALUE,
                        fontweight="bold",
                        color=text_color,
                    )
                else:
                    ax.text(
                        ci, ri, "N/S",
                        ha="center", va="center",
                        fontsize=_FONT_CELL_VALUE - 1,
                        color="#888888",
                    )

    for ax in axes_flat[n_datasets:]:
        ax.set_visible(False)

    # Reserve space at the top for the colorbar
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.91], h_pad=1.8, w_pad=0.6)

    # Shared horizontal colorbar above the subplots
    if im is not None:
        # Re-create a mappable with the plain cmap (no gray NaN) for the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.20, 0.935, 0.60, 0.030])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cb.set_label("Test accuracy", fontsize=_FONT_COLORBAR, labelpad=7)
        cb.ax.tick_params(labelsize=_FONT_COLORBAR - 1)
        cb.ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-results heatmap by reading paper_experiments/README.md.",
    )
    parser.add_argument(
        "--readme", type=Path,
        default=Path(__file__).resolve().parent / "README.md",
        help="Path to the README with result tables (default: paper_experiments/README.md)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory (default: paper_experiments/)",
    )
    parser.add_argument(
        "--format", choices=["png", "svg", "pdf"], default="png",
        help="Image format (default: png); use pdf for LaTeX/Overleaf",
    )
    parser.add_argument(
        "--min-acc", type=float, default=0.0,
        help="Minimum accuracy for colour scale (default: 0.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.readme.exists():
        sys.exit(f"ERROR: README not found at {args.readme}")

    df = parse_readme(args.readme)
    print(f"Parsed {len(df)} results from {args.readme.name}: "
          f"{df['dataset'].nunique()} datasets, "
          f"{df['trainer'].nunique()} trainers, "
          f"{df['model'].nunique()} models.")

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"results_heatmap.{args.format}"
    make_paper_heatmap(df, out_path, min_acc=args.min_acc)
    print("Done.")


if __name__ == "__main__":
    main()
