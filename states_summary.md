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

## State 1: Current State (Modular + Reproducibility)

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

### Ready For

- [ ] BPTT baseline implementation (via snnTorch)
- [ ] NeuroBench integration
- [ ] Algorithm comparison benchmarking
- [ ] Additional learning algorithms (e-prop, STDP)

---

## State 2: Target State (Benchmarking Infrastructure)

**Date:** TBD  
**Tag:** `v0.2.0-benchmarking`

### Description

Complete benchmarking infrastructure with BPTT baseline, NeuroBench integration, and standardized algorithm comparison metrics.

### Planned Characteristics

| Aspect | Status |
|--------|--------|
| Learning Algorithms | STSF + BPTT (via `snntorch.backprop.BPTT`) |
| Benchmarking | NeuroBench harness integration |
| Metrics | Accuracy, timing (`torch.profiler`), NeuroBench SNN metrics |
| Comparison | Fair algorithm comparison on same architectures |

### New Components

```
src/
├── trainers/
│   └── bptt_trainer.py      # NEW: Wraps snntorch.backprop.BPTT
├── utils/
│   └── neurobench_eval.py   # NEW: NeuroBench harness wrapper
└── benchmark_runner.py      # NEW: Algorithm comparison runner
```

### Target Capabilities

- Compare STSF vs BPTT on identical networks
- NeuroBench metrics: activation sparsity, synaptic operations
- PyTorch profiler timing: CPU time, CUDA time
- JSON output for analysis
- Reproducible benchmark runs

---

## State Transition Summary

| From | To | Key Changes |
|------|----|-------------|
| State 0 → State 1 | Modular architecture, reproducibility, checkpointing, testing |
| State 1 → State 2 | BPTT baseline, NeuroBench integration, benchmarking |

---

## Version Control Tags

```bash
# Tag current state
git tag -a v0.1.0-modular -m "Modular architecture with reproducibility"

# After benchmarking implementation
git tag -a v0.2.0-benchmarking -m "NeuroBench + BPTT benchmarking"
```

