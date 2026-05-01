#!/usr/bin/env python3
"""
generate_results.py — Post-campaign results visualisation for NeuroTrain.

Reads a campaign summary.csv (or summary.json) and produces:
  1. Per-dataset Markdown tables (trainer × model, test accuracy)
  2. A heatmap PNG  (trainer × model per dataset, colour-coded by accuracy)
  3. Optionally, a NeuroBench metrics table in Markdown

Usage:
    python generate_results.py <campaign_dir> [options]

    python generate_results.py experiments/paper/
    python generate_results.py experiments/paper/ --neurobench
    python generate_results.py experiments/paper/ --output docs/results/
    python generate_results.py experiments/paper/ --readme README.md

Arguments:
    campaign_dir        Path to the campaign directory containing summary.csv

Options:
    --output DIR        Output directory for generated files (default: campaign_dir)
    --readme PATH       If given, injects the Markdown tables and heatmap into this
                        README between <!-- RESULTS_START --> and <!-- RESULTS_END -->
    --neurobench        Also generate a NeuroBench metrics table
    --format FMT        Image format: png (default) or svg
    --min-acc FLOAT     Minimum accuracy to colour (default: 0.0)
    --no-heatmap        Skip heatmap generation (text tables only)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker


# ── Display name maps ──────────────────────────────────────────────────────
# Maps internal lowercase keys to correctly capitalised display names.

TRAINER_DISPLAY = {
    "bptt":       "BPTT",
    "ostl":       "OSTL",
    "osttp":      "OSTTP",
    "eprop":      "E-prop",
    "es_d_rtrl":  "ESD-RTRL",
    "etlp":       "ETLP",
    "stsf":       "STSF",
    "drtp":       "DRTP",
    "ottt":       "OTTT",
    "otpe":       "OTPE",
    "decolle":    "DECOLLE",
    "stop":       "STOP",
    "tp":         "TP",
    "ell":        "ELL",
    "fell":       "FELL",
    "bell":       "BELL",
    "stllr":      "STLLR",
}

MODEL_DISPLAY = {
    "fc_snn":        "FC-SNN",
    "conv_snn":      "Conv-SNN",
    "r_snn":         "R-SNN",
    "vgg9_cifar10":  "VGG9",
    "vgg9_svhn":     "VGG9",
    "vgg9_dvsgest":  "VGG9",
}

DATASET_DISPLAY = {
    "mnist":         "MNIST",
    "fashionmnist":  "Fashion-MNIST",
    "cifar10":       "CIFAR-10",
    "svhn":          "SVHN",
    "nmnist":        "N-MNIST",
    "dvsgesture":    "DVSGesture",
    "dvscifar10":    "DVS-CIFAR10",
    "shd":           "SHD",
}

DATASET_SUBTITLE = {
    "mnist":         "Test accuracy, rate-coded input, 10 classes.",
    "fashionmnist":  "Test accuracy, rate-coded input, 10 classes.",
    "cifar10":       "Test accuracy, rate-coded input, 10 classes.",
    "svhn":          "Test accuracy, rate-coded input, 10 classes.",
    "nmnist":        "Test accuracy, event-based neuromorphic input, 10 classes.",
    "dvsgesture":    "Test accuracy, event-based neuromorphic input, 11 gesture classes.",
    "dvscifar10":    "Test accuracy, event-based neuromorphic input, 10 classes.",
    "shd":           "Test accuracy, event-based neuromorphic input, 20 classes.",
}

def display_trainer(name: str) -> str:
    return TRAINER_DISPLAY.get(name.lower(), name.upper())

def display_model(name: str) -> str:
    return MODEL_DISPLAY.get(name.lower(), name)

def display_dataset(name: str) -> str:
    return DATASET_DISPLAY.get(name.lower(), name)


# ── Typography constants ───────────────────────────────────────────────────

FONT_SUPERTITLE = 20   # figure suptitle
FONT_TITLE      = 17   # dataset subplot title
FONT_AXIS_LABEL = 14   # x/y tick labels
FONT_CELL_VALUE = 15   # accuracy value inside each cell
FONT_COLORBAR   = 13   # colourbar label and ticks

# ── Helpers ────────────────────────────────────────────────────────────────

def load_summary(campaign_dir: Path) -> pd.DataFrame:
    """Load summary.csv or summary.json from a campaign directory."""
    csv_path  = campaign_dir / "summary.csv"
    json_path = campaign_dir / "summary.json"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif json_path.exists():
        with open(json_path) as f:
            records = json.load(f)
        # Flatten neurobench sub-dict with nb_ prefix
        flat = []
        for r in records:
            row = {k: v for k, v in r.items() if k != "neurobench" and k != "epoch_metrics"}
            nb = r.get("neurobench") or {}
            if isinstance(nb, dict):
                for k, v in nb.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            row[f"nb_{k}_{kk}"] = vv
                    else:
                        row[f"nb_{k}"] = v
            flat.append(row)
        df = pd.DataFrame(flat)
    else:
        sys.exit(f"No summary.csv or summary.json found in {campaign_dir}")

    required = {"trainer", "model", "dataset", "test_accuracy"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"summary is missing columns: {missing}")

    return df


def format_acc(val) -> str:
    """Format accuracy as percentage string, or — for missing values."""
    try:
        f = float(val)
        if np.isnan(f):
            return "—"
        return f"{f * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


# ── Markdown tables ────────────────────────────────────────────────────────

def make_markdown_tables(df: pd.DataFrame) -> str:
    """Generate one Markdown accuracy table per dataset."""
    lines = []
    datasets = sorted(df["dataset"].unique())

    for dataset in datasets:
        sub      = df[df["dataset"] == dataset]
        trainers = sorted(sub["trainer"].unique())
        models   = sorted(sub["model"].unique())

        lines.append(f"### {display_dataset(dataset)}\n")
        subtitle = DATASET_SUBTITLE.get(dataset.lower(), "Test accuracy (mean ± std where multiple seeds available).")
        lines.append(f"*{subtitle}*\n")

        model_labels = [display_model(m) for m in models]
        header = "| Algorithm | " + " | ".join(model_labels) + " |"
        sep    = "|---" + "|---" * len(models) + "|"
        lines.append(header)
        lines.append(sep)

        for trainer in trainers:
            row = f"| {display_trainer(trainer)} |"
            for model in models:
                cell = sub[(sub["trainer"] == trainer) & (sub["model"] == model)]
                if cell.empty:
                    row += " — |"
                else:
                    acc_vals = cell["test_accuracy"].dropna()
                    if len(acc_vals) == 0:
                        row += " — |"
                    elif len(acc_vals) == 1:
                        row += f" {format_acc(acc_vals.iloc[0])} |"
                    else:
                        mean = acc_vals.mean()
                        std  = acc_vals.std()
                        row += f" {format_acc(mean)} ±{std*100:.1f}% |"
            lines.append(row)

        lines.append("")  # blank line between tables

    return "\n".join(lines)


# ── NeuroBench table ───────────────────────────────────────────────────────

NEUROBENCH_COLS = {
    "nb_ClassificationAccuracy": "Accuracy (NB)",
    "nb_ActivationSparsity":     "Act. Sparsity",
    "nb_MembraneUpdates":        "Membrane Updates",
    "nb_Footprint":              "Footprint (B)",
    "nb_ConnectionSparsity":     "Conn. Sparsity",
    "nb_ParameterCount":         "Parameters",
}

def make_neurobench_table(df: pd.DataFrame) -> str:
    """Generate a NeuroBench metrics table (all experiments, one row each)."""
    nb_cols = [c for c in NEUROBENCH_COLS if c in df.columns]
    if not nb_cols:
        return "*No NeuroBench metrics found in summary — run with `neurobench: true`.*\n"

    lines = ["### NeuroBench Metrics\n"]
    display_cols = ["trainer", "model", "dataset"] + nb_cols
    headers = ["Algorithm", "Model", "Dataset"] + [NEUROBENCH_COLS[c] for c in nb_cols]

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|---" * len(headers) + "|")

    for _, row in df[display_cols].sort_values(["dataset", "trainer", "model"]).iterrows():
        cells = [
            display_trainer(str(row["trainer"])),
            display_model(str(row["model"])),
            display_dataset(str(row["dataset"])),
        ]
        for c in nb_cols:
            val = row[c]
            try:
                cells.append(f"{float(val):.4f}")
            except (TypeError, ValueError):
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


# ── Heatmap ────────────────────────────────────────────────────────────────

def make_heatmap(df: pd.DataFrame, output_path: Path, min_acc: float = 0.0,
                 campaign_name: str = "") -> None:
    """
    Generate a heatmap PNG: one subplot per dataset,
    arranged in a 2x3 grid (3 datasets per row, 2 rows).
    Rows = trainers, columns = models, colour = test accuracy.
    """
    datasets = sorted(df["dataset"].unique())
    trainers = sorted(df["trainer"].unique())
    models   = sorted(df["model"].unique())

    trainer_labels = [display_trainer(t) for t in trainers]
    model_labels   = [display_model(m)   for m in models]

    n_datasets = len(datasets)

    # Fixed layout: 2 rows, 3 columns
    nrows, ncols = 2, 3
    max_panels = nrows * ncols
    if n_datasets > max_panels:
        raise ValueError(
            f"2x3 heatmap grid supports at most {max_panels} datasets, "
            f"but found {n_datasets}: {datasets}"
        )

    # Slightly smaller subplot sizing so 3 panels fit per row
    cell_w = 1.9
    cell_h = 1.15

    fig_w = max(cell_w * len(models) * ncols + 3.0, 16)
    fig_h = max(cell_h * len(trainers) * nrows + 3.2, 11)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    cmap = plt.cm.YlGn
    norm = mcolors.Normalize(vmin=min_acc, vmax=1.0)

    im = None

    for i, dataset in enumerate(datasets):
        ax  = axes_flat[i]
        sub = df[df["dataset"] == dataset]

        # Build accuracy matrix: rows = trainers, cols = models
        mat = np.full((len(trainers), len(models)), np.nan)
        for ri, trainer in enumerate(trainers):
            for ci, model in enumerate(models):
                cell = sub[(sub["trainer"] == trainer) & (sub["model"] == model)]
                if not cell.empty:
                    vals = cell["test_accuracy"].dropna()
                    if len(vals) > 0:
                        mat[ri, ci] = vals.mean()

        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        # Axis tick labels
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(
            model_labels,
            rotation=28,
            ha="right",
            fontsize=FONT_AXIS_LABEL,
            fontweight="semibold",
        )

        ax.set_yticks(range(len(trainers)))
        ax.set_yticklabels(
            trainer_labels,
            fontsize=FONT_AXIS_LABEL,
            fontweight="semibold",
        )

        # Dataset title
        ax.set_title(
            display_dataset(dataset),
            fontsize=FONT_TITLE,
            fontweight="bold",
            pad=12,
        )

        # Subtle grid lines between cells
        ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(trainers), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Cell annotations
        for ri in range(len(trainers)):
            for ci in range(len(models)):
                val = mat[ri, ci]
                if not np.isnan(val):
                    text_color = "white" if val > 0.65 else "#1a1a1a"
                    ax.text(
                        ci, ri, f"{val * 100:.1f}%",
                        ha="center", va="center",
                        fontsize=FONT_CELL_VALUE,
                        fontweight="bold",
                        color=text_color,
                    )
                else:
                    ax.text(
                        ci, ri, "—",
                        ha="center", va="center",
                        fontsize=FONT_CELL_VALUE,
                        color="#bbbbbb",
                    )

    # Hide unused panels if fewer than 6 datasets
    for ax in axes_flat[n_datasets:]:
        ax.set_visible(False)

    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M")
    title_parts = ["NeuroTrain — Benchmarking Results"]
    if campaign_name:
        title_parts.append(f"Campaign: {campaign_name}")
    title_parts.append(timestamp)

    fig.suptitle(
        "\n".join(title_parts),
        fontsize=FONT_SUPERTITLE,
        fontweight="bold",
        y=0.97,
    )

    # Leave room at the top for title + horizontal colorbar
    plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.82], h_pad=2.2, w_pad=2.0)

    # One shared horizontal colourbar below the title
    if im is not None:
        cbar_ax = fig.add_axes([0.30, 0.845, 0.40, 0.022])  # [left, bottom, width, height]
        cb = fig.colorbar(
            im,
            cax=cbar_ax,
            orientation="horizontal",
        )
        cb.set_label("Test accuracy", fontsize=FONT_COLORBAR, labelpad=6)
        cb.ax.tick_params(labelsize=FONT_COLORBAR - 1)
        cb.ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved → {output_path}")

# ── README injection ───────────────────────────────────────────────────────

START_MARKER = "<!-- RESULTS_START -->"
END_MARKER   = "<!-- RESULTS_END -->"

def inject_into_readme(
    readme_path: Path,
    md_tables: str,
    campaign_dir: Path,
    heatmap_path: Path | None = None,
    ) -> None:
    """
    Replace content between RESULTS_START and RESULTS_END markers in README.

    Injects:
      - A provenance line identifying the source campaign
      - The heatmap image, copied into docs/figures with the campaign name
      - The per-dataset Markdown accuracy tables
    """
    text = readme_path.read_text()
    if START_MARKER not in text or END_MARKER not in text:
        print(f"  WARNING: markers {START_MARKER} / {END_MARKER} not found in {readme_path}")
        print("  Add them to README.md around the results section to enable auto-injection.")
        return

    block_lines = [
        f"*Results from campaign `{campaign_dir.name}`. "
        f"Generated by `scripts/generate_results.py`.*",
        "",
    ]

    if heatmap_path is not None and heatmap_path.exists():
        # Save/copy heatmap into docs/figures using the campaign name
        figures_dir = readme_path.resolve().parent / "docs" / "figures" / "campaign_results"
        figures_dir.mkdir(parents=True, exist_ok=True)

        safe_campaign_name = "".join(
            c if c.isalnum() or c in "-_." else "_"
            for c in campaign_dir.name
        )

        docs_heatmap_path = figures_dir / f"{safe_campaign_name}_heatmap{heatmap_path.suffix}"
        shutil.copy2(heatmap_path, docs_heatmap_path)

        print(f"  Heatmap copied → {docs_heatmap_path}")

        # Express docs/figures path relative to README directory
        rel = docs_heatmap_path.resolve().relative_to(readme_path.resolve().parent)
        rel = rel.as_posix()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        block_lines += [
            f"![NeuroTrain benchmarking results — test accuracy heatmap, "
            f"algorithms (rows) × architecture–dataset combinations (cols). "
            f"Campaign: {campaign_dir.name} — {timestamp}]({rel})",
            f"*NeuroTrain — Benchmarking Results · Campaign: `{campaign_dir.name}` · {timestamp}*",
            "",
        ]

    block_lines.append(md_tables)
    injected = "\n".join(block_lines)

    before = text.split(START_MARKER)[0]
    after = text.split(END_MARKER)[1]
    new_text = before + START_MARKER + "\n\n" + injected + "\n" + END_MARKER + after

    readme_path.write_text(new_text)
    print(f"  README updated → {readme_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate NeuroTrain results tables and heatmap from a campaign summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("campaign_dir", type=Path,
                        help="Campaign directory containing summary.csv or summary.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: same as campaign_dir)")
    parser.add_argument("--readme", type=Path, default=None,
                        help="README.md to inject tables into (requires markers)")
    parser.add_argument("--neurobench", action="store_true",
                        help="Also generate a NeuroBench metrics table")
    parser.add_argument("--format", choices=["png", "svg"], default="png",
                        help="Heatmap image format (default: png)")
    parser.add_argument("--min-acc", type=float, default=0.0,
                        help="Minimum accuracy for heatmap colour scale (default: 0.0)")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip heatmap generation")
    args = parser.parse_args()

    campaign_dir = args.campaign_dir.resolve()
    output_dir   = (args.output or campaign_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading summary from {campaign_dir} ...")
    df = load_summary(campaign_dir)
    print(f"  {len(df)} experiments loaded across "
          f"{df['dataset'].nunique()} datasets, "
          f"{df['trainer'].nunique()} trainers, "
          f"{df['model'].nunique()} models.")

    # Markdown tables
    print("Generating Markdown tables ...")
    md_tables = make_markdown_tables(df)
    md_path   = output_dir / "results_tables.md"
    md_path.write_text(md_tables)
    print(f"  Tables saved → {md_path}")

    if args.neurobench:
        print("Generating NeuroBench table ...")
        nb_md   = make_neurobench_table(df)
        nb_path = output_dir / "neurobench_table.md"
        nb_path.write_text(nb_md)
        print(f"  NeuroBench table saved → {nb_path}")
        md_tables += "\n" + nb_md

    # Heatmap
    heatmap_path = None
    if not args.no_heatmap:
        print("Generating heatmap ...")
        heatmap_path = output_dir / f"results_heatmap.{args.format}"
        make_heatmap(df, heatmap_path, min_acc=args.min_acc,
                     campaign_name=campaign_dir.name)

    # README injection
    if args.readme:
        print(f"Injecting into {args.readme} ...")
        inject_into_readme(
            readme_path=args.readme.resolve(),
            md_tables=md_tables,
            campaign_dir=campaign_dir,
            heatmap_path=heatmap_path,
        )

    print("Done.")


if __name__ == "__main__":
    main()