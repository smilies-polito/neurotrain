# Reproducibility and baseline checks (ELL/FELL/BELL)

Use this when comparing runs (e.g. before vs after merge, or across branches) to ensure you're comparing the same setup.

---

## 2. Environment – how to check

### 2.1 Random seed and data order

- **Problem:** `run_all_benchmarks.py` did not set any random seed. So:
  - Model init (weights) differs between runs.
  - DataLoader `shuffle=True` uses PyTorch’s default RNG → different batch order each run.
- **Check:** Run the same command twice and see if accuracy changes; if it does, the run is not seed-locked.

- **Fix (reproducible runs):** Use a fixed seed at start. From the repo root:
  ```bash
  python run_all_benchmarks.py --datasets MNIST --algorithms ell --epochs 5
  ```
  The script sets `seed=42` by default and prints it; use `--seed 42` explicitly if you want. Run the same command twice; you should get the same final accuracy (and similar loss curve). With `num_workers > 0`, DataLoader worker RNG can still cause tiny variation; for strict reproducibility use `--batch-size` and same seed.

- **Code locations:**
  - Seed setting: `src/utils/experiment_logger.py` → `set_all_seeds(seed, deterministic)`.
  - `main.py` uses it (config-driven runs); `run_all_benchmarks.py` now sets seed at start when you use the default (see “Reproducible run” below).

### 2.2 PyTorch and CUDA versions

- **Problem:** Different PyTorch/CUDA versions can change numerics (e.g. reduction order, cuDNN algorithms) and thus accuracy.
- **Check:** Print versions in the same run that produces your metrics:

  ```bash
  python -c "
  import torch
  print('PyTorch:', torch.__version__)
  print('CUDA available:', torch.cuda.is_available())
  if torch.cuda.is_available():
      print('CUDA version:', torch.version.cuda)
      print('cuDNN:', torch.backends.cudnn.version())
  "
  ```

  Save this output (e.g. in a log or `benchmark_results/` next to the JSON) for both “good” and “bad” runs so you can compare.

- **In-script:** `run_all_benchmarks.py` now prints a short “Environment” block at start (PyTorch version, CUDA availability, and the seed used). Use that to document “what I ran with”.

---

## 3. Baseline memory – how to check (hyperparameters)

- **Problem:** “Better accuracies on 11-ell-fell-bell” might be from more epochs, different LR, or a different entrypoint (e.g. config-based run with different defaults).
- **Check:**

  1. **What you run now (run_all_benchmarks):**
     - Defaults: `epochs=50`, `batch_size=128`, `lr=0.001`, MNIST `timesteps=25`, `layer_sizes=[784, 256, 10]`.
     - Override with CLI: `--epochs 50 --lr 0.001 --batch-size 128`.

  2. **What you ran on 11-ell-fell-bell:**
     - If you used **config-based runner**:  
       `python src/benchmark_runner.py --config configs/benchmark_comparison.yaml`  
       → Hyperparameters come from that YAML: `configs/benchmark_comparison.yaml` has `epochs: 50`, `batch_size: 128`, `lr: 0.001`. So they match run_all_benchmarks defaults.
     - If you used **main.py** with a custom config:  
       Check that config’s `training.epochs`, `training.batch_size`, `training.learning_rate` and `data.timesteps` (and model `layer_sizes`) and align them to run_all_benchmarks for a fair comparison.

  3. **Side-by-side table:** Fill this for “then” vs “now”:

  | Setting      | 11-ell-fell-bell (then) | Current run (now) |
  |-------------|--------------------------|-------------------|
  | Entrypoint  | e.g. benchmark_runner + YAML | run_all_benchmarks.py |
  | epochs      | ?                        | 50 (default)      |
  | batch_size  | ?                        | 128               |
  | lr          | ?                        | 0.001             |
  | timesteps   | ?                        | 25 (MNIST)        |
  | layer_sizes | ?                        | [784, 256, 10]    |
  | seed set?   | ?                        | yes (if using new script) |
  | PyTorch/CUDA| ?                        | (from env print)  |

- **Recommendation:** Run the same setup on both branches:  
  e.g. `python run_all_benchmarks.py --datasets MNIST --algorithms ell,fell,bell --epochs 50 --lr 0.001` with seed and env printed, and compare the printed seed + env + hyperparameters to what you had on 11-ell-fell-bell.

---

## Reproducible run (seed + env printed)

`run_all_benchmarks.py` sets a default seed (42) at start and prints an “Environment” line (PyTorch version, CUDA, seed). So you can:

1. Run once and note the printed seed and versions.
2. Re-run with the same command: you should get the same (or very close) accuracy.
3. When comparing to an old run, note whether the old run used a seed and which PyTorch/CUDA version it used.

Example output at start of run:

```
Environment: PyTorch 2.x.x, CUDA available: True, seed=42
```

Use that in your logs or in the results JSON so “baseline memory” (point 3) is explicit.
