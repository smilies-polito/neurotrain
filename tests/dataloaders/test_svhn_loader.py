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


def check_split(data: torch.Tensor, target: torch.Tensor, split: str) -> None:
    print(f"[{split}]")
    print(f"\t data.shape={tuple(data.shape)}  expected [{T}, {B}, 3, 32, 32]")
    print(f"\t target.shape={tuple(target.shape)}  dtype={target.dtype}")
    print(f"\t label range: min={target.min().item()}  max={target.max().item()}")
    print(f"\t unique labels: {sorted(target.unique().tolist())}")

    assert data.shape == (T, B, 3, 32, 32), f"unexpected data shape: {tuple(data.shape)}"
    assert target.dtype == torch.long, f"target dtype should be long, got {target.dtype}"
    assert target.min().item() >= 0, "labels must be >= 0"
    assert target.max().item() <= 9, "labels must be <= 9 — raw SVHN label 10 not normalised"
    print("\t OK (labels in [0, 9])")


def main() -> None:
    torch.set_printoptions(precision=3, sci_mode=False)

    train_loader, test_loader = SVHNLoader(
        batch_size=B,
        T=T,
        pin_memory=False,
        seed=0,
        num_workers=0,
    )

    train_data, train_target = next(iter(train_loader))
    check_split(train_data, train_target, "TRAIN")

    x = train_data[0, 0, 0]
    print(f"\t sample pixel: shape={tuple(x.shape)} min={x.min().item():.3f} max={x.max().item():.3f} mean={x.mean().item():.3f}")

    test_data, test_target = next(iter(test_loader))
    check_split(test_data, test_target, "TEST")


if __name__ == "__main__":
    main()
