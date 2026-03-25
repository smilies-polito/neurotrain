#!/usr/bin/env python3
"""Minimal smoke test for SVHNLoader."""

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


def main() -> None:
    torch.set_printoptions(precision=3, sci_mode=False)

    train_loader, test_loader = SVHNLoader(
        batch_size=5,
        T=10,
        pin_memory=False,
        seed=0,
        num_workers=0,
    )

    train_data, train_target = next(iter(train_loader))
    print(f"[TRAIN] \n\t data.shape={tuple(train_data.shape)} exprected [batch, timesteps, channels, height, width] \n\t data.ndim={train_data.ndim} \n\t dtype={train_data.dtype}")
    print(f"\t target.shape={tuple(train_target.shape)} \n\t target.ndim={train_target.ndim} \n\t dtype={train_target.dtype}")
    print(f"\t sample_target={train_target[0]}")
    x = train_data[0, 0, 0]
    print(f"\t sample_data: shape={tuple(x.shape)} min={x.min().item():.3f} max={x.max().item():.3f} mean={x.float().mean().item():.3f} nnz={(x!=0).sum().item()}")
    x = x.squeeze()
    s = "\n".join("".join("1" if v else "." for v in row) for row in (x > 0).tolist())
    print("\t sample_data:\n" + s)
    print("BAD VALUES in sample_data!" if not torch.all((x == 0) | (x == 1)) else "OK (all 0/1)")



    test_data, test_target = next(iter(test_loader))
    assert test_data.size(0) == test_target.size(0), "test batch size mismatch"
    print(f"[TEST] \n\t data.shape={tuple(test_data.shape)} exprected [batch, timesteps, channels, height, width] \n\t data.ndim={test_data.ndim} \n\t dtype={test_data.dtype}")
    print(f"\t target.shape={tuple(test_target.shape)} \n\t target.ndim={test_target.ndim} \n\t dtype={test_target.dtype}")
    print(f"\t sample_target={test_target[0]}")
    x = test_data[0, 0, 0]  
    print(f"\t sample_data: shape={tuple(x.shape)} min={x.min().item():.3f} max={x.max().item():.3f} mean={x.float().mean().item():.3f} nnz={(x!=0).sum().item()}")
    x = x.squeeze()
    s = "\n".join("".join("1" if v else "." for v in row) for row in (x > 0).tolist())
    print("\t sample_data:\n" + s)
    print("BAD VALUES in sample_data!" if not torch.all((x == 0) | (x == 1)) else "OK (all 0/1)")



if __name__ == "__main__":
    main()