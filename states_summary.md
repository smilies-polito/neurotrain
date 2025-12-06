# Project States Summary

> Tracking the evolution of the SNN Training Benchmarking project through development phases.

---

## State 0: Initial State (Pre-Refactoring)

**Date:** Before 2025-12-01  
**Tag:** `initial`

### Description

Basic spiking neural network training script with STSF (Spiking Time Sparse Feedback) learning algorithm. Monolithic codebase without modular architecture or reproducibility features.

### Characteristics

| Aspect | Status |
|--------|--------|
| Learning Algorithms | STSF only (embedded in training loop) |
| Configuration | CLI arguments only, no config files |
| Reproducibility | Manual seed setting, no context logging |
| Checkpointing | None |
| Testing | Minimal or none |
| Code Quality | No formatters/linters enforced |

### Capabilities

- Train STSF on MNIST
- Basic accuracy tracking
- Manual hyperparameter tuning

### Limitations

- No experiment reproducibility
- No checkpoint/resume support
- Hard to add new learning algorithms
- No standardized benchmarking

---

## State 1: Modular + Reproducibility

**Date:** 2025-12-03  
**Tag:** `v0.1.0-modular`

### Description

Refactored into modular architecture with abstract base classes, YAML configuration system, comprehensive experiment logging, checkpointing, and test suite. Ready for benchmarking infrastructure.

### Characteristics

| Aspect | Status |
|--------|--------|
| Learning Algorithms | STSF implemented; BaseTrainer interface defined |
| Configuration | Dataclass-based config + YAML files + CLI overrides |
| Reproducibility | Full context logging (seeds, git, environment, RNG state) |
| Checkpointing | Best/latest/periodic saving with resume support |
| Testing | Pytest suite (~90% coverage) |
| Code Quality | Black, flake8, isort, pre-commit hooks |

### Architecture

```
src/
├── trainers/
│   ├── base_trainer.py      # Abstract interface
│   └── stsf_trainer.py      # STSF implementation
├── networks/
│   └── fc_network.py        # FCNetwork with snnTorch
├── datasets/                # 5 dataset loaders
├── utils/
│   ├── config.py            # Typed config system
│   ├── checkpoint.py        # Checkpointing
│   └── experiment_logger.py # Reproducibility
└── LearningAlgorithms.py    # Train/eval loops
```

### Capabilities

- Reproducible experiments with full context logging
- Resume training from checkpoints
- YAML-based configuration with CLI overrides
- TensorBoard integration
- Graceful exit (Ctrl+C saves checkpoint)
- Multiple datasets: MNIST, FashionMNIST, CIFAR10, SVHN, DVSGesture
- Quantization support for hardware deployment

---

## State 2: Current State (BPTT Baseline + NeuroBench Integration)

**Date:** 2025-12-06  
**Tag:** `v0.2.0-benchmarking`

### Description

Complete benchmarking infrastructure with BPTT baseline trainer, NeuroBench v2.1.0 integration, multi-dataset benchmarking runner, and comprehensive training/efficiency metrics. Supports automated comparison of SNN learning algorithms across multiple classification datasets.

### Characteristics

| Aspect | Status |
|--------|--------|
| Learning Algorithms | STSF + **BPTT** (using snnTorch functional API) |
| Benchmarking | Full NeuroBench v2.1.0 integration with custom wrapper |
| Datasets | 5 standard + 4 NeuroBench official (loaders ready) |
| Timing | `time.perf_counter()` with `torch.cuda.synchronize()` |
| NeuroBench Metrics | ParameterCount, Footprint, ActivationSparsity, SynapticOperations (Effective + Dense MACs), MembraneUpdates |
| Output | JSON results + formatted summary tables |

### New Components

```
src/
├── trainers/
│   └── bptt_trainer.py          # NEW: BPTT with snnTorch functional API
├── datasets/
│   └── neurobench_loaders.py    # NEW: SpeechCommands, WISDM, PrimateReaching, MackeyGlass
├── utils/
│   └── neurobench_eval.py       # NEW: NeuroBench v2.1.0 wrapper with custom postprocessor
├── benchmark_runner.py          # NEW: Single-dataset benchmark runner
run_all_benchmarks.py            # NEW: Multi-dataset benchmark orchestrator
configs/
└── benchmark_comparison.yaml    # NEW: Benchmark configuration
```

### Key Implementation Details

#### BPTT Trainer (`bptt_trainer.py`)
- Uses `snntorch.functional` loss functions (ce_rate, ce_count, mse_count)
- Properly handles optimizer recreation on `.to(device)` for CUDA compatibility
- Matches `BaseTrainer` interface for fair algorithm comparison
- ~156 lines (thin wrapper, not reimplementation)

#### NeuroBench Integration (`neurobench_eval.py`)
- Custom `NeuroBenchWrapper` class handles temporal loop for FCNetwork
- Uses `SNNTorchModel` with `custom_forward=True` for proper spike format
- `spike_to_prediction` postprocessor converts `[batch, T, classes]` → `[batch]`
- Properly moves tensors to correct device (fixes CPU/CUDA mismatch)
- Returns spikes as `[timesteps, batch, classes]` (NeuroBench expected format)

#### Timing Approach
- **Replaced PyTorch profiler** (too slow, 10x+ overhead) with `time.perf_counter()`
- Uses `torch.cuda.synchronize()` before/after for accurate GPU wall-clock time
- Reports per-epoch timing for meaningful algorithm comparison

#### Summary Tables
Two summary tables produced:

1. **Training Summary** (per dataset, per algorithm):
   - Final test accuracy
   - Total wall-clock time
   - Average time per epoch

2. **NeuroBench Metrics Summary**:
   - ParameterCount, Footprint
   - ActivationSparsity (spike sparsity)
   - Effective MACs (actual ops with sparsity)
   - Dense MACs (baseline without sparsity)
   - **Savings %** (compute reduction from spike sparsity)
   - MembraneUpdates

### Capabilities

- **Automated benchmarking**: `python run_all_benchmarks.py --epochs 10 --device cuda`
- **Fair algorithm comparison**: Same network, same data, different trainers
- **NeuroBench metrics**: Industry-standard SNN efficiency metrics
- **Multi-dataset support**: Run across MNIST, CIFAR10, FashionMNIST, SVHN simultaneously
- **JSON output**: `benchmark_results/full_benchmark_<timestamp>.json`
- **Compute savings visualization**: See efficiency gain from spike sparsity

### Bug Fixes Applied

| Issue | Fix |
|-------|-----|
| NeuroBench API (`static_metrics` kwarg) | Changed to `metric_list=[static, workload]` |
| `'NoneType' not iterable` | Pass `[]` not `None` for preprocessors/postprocessors |
| Device mismatch (CPU/CUDA) | Pass `device` to wrapper, call `x.to(self.device)` |
| `too many values to unpack` | Use `custom_forward=True`, return `[T, B, C]` tensor |
| SVHN wrong import | Changed `from torchvision.datasets import MNIST` → `SVHN` |
| CIFAR10 missing torch | Added `import torch` to `cifar10_loader.py` |
| Profiler overhead | Replaced with `time.perf_counter()` + `cuda.synchronize()` |
| SynapticOperations dict output | Extract `Effective_MACs` and `Dense` separately |
| ConnectionSparsity always 0 | Replaced with Savings % (more meaningful for dense networks) |

### Dependencies (Container)

```
torch, torchvision, torchaudio, torchcodec
snntorch==0.9.4
neurobench==2.1.0
tensorboard, pytorch_lightning, pyyaml, tqdm, matplotlib, ray[all]
```

### Current Limitations

1. **NeuroBench datasets**: SpeechCommands requires `torchcodec`, WISDM requires `pytorch_lightning` (added to .def, need container rebuild)
2. **Regression tasks**: PrimateReaching, MackeyGlass loaders ready but trainers only support classification
3. **DVSGesture**: May need updates for NeuroBench compatibility

### Usage

```bash
# Run full benchmark suite
python run_all_benchmarks.py --epochs 50 --device cuda

# With custom settings
python run_all_benchmarks.py \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.001 \
    --device cuda \
    --output-dir ./benchmark_results

# Using Singularity container
singularity exec src/snn-training-benchmarking.sif \
    python3 run_all_benchmarks.py --epochs 50 --device cuda
```

### Sample Output

```
============================================================
TRAINING SUMMARY (10 epochs)
============================================================
Dataset        | BPTT Acc  | STSF Acc  | BPTT Wall Time  | STSF Wall Time  | BPTT Time/epoch  | STSF Time/epoch
------------------------------------------------------------
MNIST          | 97.23%    | 91.72%    | 45.3s           | 32.1s           | 4521.2ms         | 3210.5ms
CIFAR10        | 48.12%    | 42.35%    | 89.2s           | 67.4s           | 8920.1ms         | 6740.3ms

============================================================
NEUROBENCH METRICS SUMMARY
============================================================
Dataset        | Algo   | Params       | Footprint    | ActSpars   | Eff. MACs      | Dense MACs     | Savings  | MemUpdates
------------------------------------------------------------
MNIST          | BPTT   | 203,264      | 810.7 KB     | 0.7275     | 1,234,567      | 5,082,500      | 75.7%    | 6,649
MNIST          | STSF   | 203,264      | 810.7 KB     | 0.3836     | 3,128,450      | 5,082,500      | 38.4%    | 6,649
```

---

## State 3: Target State (Full Benchmark Suite)

**Date:** TBD  
**Tag:** `v0.3.0-complete`

### Description

Full benchmark suite with regression support, all NeuroBench datasets working, and additional learning algorithms.

### Planned Features

- [ ] Regression task support in trainers (MSE loss, continuous targets)
- [ ] Rebuild container with `torchcodec`, `pytorch_lightning`
- [ ] Enable SpeechCommands, WISDM, PrimateReaching, MackeyGlass
- [ ] Add e-prop trainer
- [ ] Power consumption metrics (pynvml)
- [ ] Interactive visualization dashboard

---

## State Transition Summary

| From | To | Key Changes |
|------|----|-------------|
| State 0 → State 1 | Modular architecture, reproducibility, checkpointing, testing |
| State 1 → State 2 | BPTT trainer, NeuroBench v2.1.0 integration, multi-dataset benchmarking, efficiency metrics |
| State 2 → State 3 | Regression support, full NeuroBench dataset suite, additional algorithms |

---

## Version Control Tags

```bash
# Tag current state
git tag -a v0.2.0-benchmarking -m "BPTT baseline + NeuroBench integration"

# After full benchmark suite
git tag -a v0.3.0-complete -m "Full benchmark suite with regression support"
```
