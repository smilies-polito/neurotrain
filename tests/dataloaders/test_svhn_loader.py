#!/usr/bin/env python3
"""Smoke test for SVHNLoader — verifies label normalisation (0-9) and batch shapes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from datasets.svhn_loader import SVHNLoader

T = 10
B = 32


def check_split(loader, split: str) -> None:
    print(f"[{split}]")

    counts = torch.zeros(10, dtype=torch.long)
    total_samples = 0

    for batch_idx, (data, target) in enumerate(loader):
        assert data.shape[1:] == (B, 3, 32, 32) or data.shape[0] == T, (
            f"unexpected data shape: {tuple(data.shape)}"
        )
        assert target.dtype == torch.long, (
            f"target dtype should be long, got {target.dtype}"
        )
        assert target.min().item() >= 0, (
            f"batch {batch_idx}: labels must be >= 0"
        )
        assert target.max().item() <= 9, (
            f"batch {batch_idx}: labels must be <= 9 — raw SVHN label 10 not normalised"
        )

        for label in range(10):
            counts[label] += (target == label).sum()
        total_samples += target.numel()

        if batch_idx == 0:
            print(f"\t first batch — data.shape={tuple(data.shape)}  "
                  f"target.shape={tuple(target.shape)}  dtype={target.dtype}")
            print(f"\t label range: min={target.min().item()}  max={target.max().item()}")
            print(f"\t unique labels in first batch: {sorted(target.unique().tolist())}")

    print(f"\t total samples across ALL batches: {total_samples}")
    print(f"\t label counts across entire dataset:")
    for label in range(10):
        print(f"\t\t label {label}: {counts[label].item()} samples")
    print("\t OK (all labels in [0, 9])")


def main() -> None:
    torch.set_printoptions(precision=3, sci_mode=False)

    train_loader, test_loader = SVHNLoader(
        batch_size=B,
        T=T,
        pin_memory=False,
        seed=0,
        num_workers=0,
    )

    check_split(train_loader, "TRAIN")

    # Sample pixel stats from first batch
    first_data, _ = next(iter(train_loader))
    x = first_data[0, 0, 0]
    print(f"\t sample pixel: shape={tuple(x.shape)} min={x.min().item():.3f} "
          f"max={x.max().item():.3f} mean={x.mean().item():.3f}")

    check_split(test_loader, "TEST")


if __name__ == "__main__":
    main()
