# Test programs

This directory contains small, self-contained scripts used as smoke/integration tests for the repo.

## Tests for trainer validation

Each test is a single Python script that launches **one trainer on one dataset with one network**.

Naming convention:
`[trainer]_[dataset]_[network].py`.

Template (trainer-validation scripts are the same shape):
- bootstrap imports so it can run from `tests/` (adds `src/` to `sys.path`; may include small import workarounds)
- parse a minimal CLI (`--epochs`, `--batch-size`, `--timesteps`, `--lr`, `--beta`, `--threshold`, `--seed`, `--device`)
- build: dataset loader -> network -> trainer
- run a short train/eval loop and print basic metrics
- optional Optuna mode when `--optuna-trials > 0` (runs the same loop per trial)

## Dataloader smoke tests

`tests/dataloaders/test_*_loader.py` instantiates a loader, pulls one train/test batch, and prints/asserts basic shapes and values.
