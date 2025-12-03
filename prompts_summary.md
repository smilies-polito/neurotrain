# Prompts Summary

> Reproducible prompts for each development phase. Each prompt is designed to be self-contained and generate the required changes.

---

## Prompt History

| # | Phase | State Transition | Status |
|---|-------|------------------|--------|
| 1 | Reproducibility & Logging | State 0 → State 1 | ✅ Complete |
| 2 | BPTT Benchmarking Baseline | State 1 → State 2 | 🔲 Next |

---

## Prompt 1: Reproducibility & Logging Infrastructure

**Purpose:** Transform monolithic training script into modular architecture with full reproducibility.

**State Transition:** State 0 (Initial) → State 1 (Modular + Reproducibility)

### Exact Prompt

```
Refactor the SNN training codebase to add:

1. MODULAR ARCHITECTURE
   - Create abstract BaseTrainer class with train_sample() and reset() methods
   - Move STSF implementation to STSFTrainer inheriting from BaseTrainer
   - Separate network definitions into src/networks/
   - Create factory functions: get_trainer(), get_loader()

2. CONFIGURATION SYSTEM
   - Implement dataclass-based configuration in src/utils/config.py
   - Support YAML/JSON config files in configs/
   - CLI arguments override config file values
   - Validate configuration with helpful error messages

3. REPRODUCIBILITY
   - Create ExperimentLogger class in src/utils/experiment_logger.py
   - Log: Python/NumPy/PyTorch/CUDA seeds
   - Log: Environment (Python version, PyTorch version, CUDA, device)
   - Log: Git commit hash, branch, dirty state
   - Log: Full RNG state for exact resume
   - Save to experiment_context.json

4. CHECKPOINTING
   - Create CheckpointManager in src/utils/checkpoint.py
   - Support: save_best, save_latest, save_every N epochs
   - Include: model state, optimizer state, epoch, metrics, RNG state
   - Implement resume from checkpoint
   - Add graceful exit (Ctrl+C saves checkpoint)

5. TESTING
   - Create pytest test suite in tests/
   - Test config loading/validation
   - Test checkpoint save/load
   - Test trainer interface
   - Aim for >80% coverage

6. CODE QUALITY
   - Add pyproject.toml with Black, isort, flake8 config
   - Add pre-commit hooks
   - Format all code with Black (88 char line length)

Follow these conventions:
- Use type hints for all function signatures
- Write docstrings (Google style)
- Use snake_case for functions, PascalCase for classes
- Keep learning algorithm logic in trainers, not networks
```

### Generated Files

- `src/utils/config.py` — Dataclass configuration system
- `src/utils/checkpoint.py` — Checkpointing with resume
- `src/utils/experiment_logger.py` — Reproducibility logging
- `src/trainers/base_trainer.py` — Abstract trainer interface
- `src/trainers/stsf_trainer.py` — STSF implementation
- `configs/*.yaml` — Configuration files
- `tests/` — Test suite
- `pyproject.toml` — Project configuration

---

## Prompt 2: BPTT Benchmarking Baseline (NEXT)

**Purpose:** Add BPTT baseline and NeuroBench integration for algorithm benchmarking.

**State Transition:** State 1 (Modular + Reproducibility) → State 2 (Benchmarking Infrastructure)

### Exact Prompt

```
Add BPTT baseline and NeuroBench benchmarking infrastructure.

CONTEXT:
- Reference PROJECT_STATE.md for current codebase structure
- Reference TASK_NEUROBENCH_BPTT_BASELINE.md for detailed requirements

CORE PRINCIPLE: REUSE EXISTING LIBRARIES
- Use snntorch.backprop.BPTT (do NOT reimplement BPTT)
- Use snntorch.functional.* for loss functions
- Use neurobench.models.SNNTorchModel for model wrapping
- Use neurobench.benchmarks.Benchmark for evaluation harness
- Use neurobench.metrics.* for all SNN metrics
- Use torch.profiler for timing (do NOT write custom timers)

DELIVERABLES:

1. src/trainers/bptt_trainer.py (~40 lines)
   - Thin wrapper around snntorch.backprop.BPTT
   - Match BaseTrainer interface (train_sample, reset)
   - Use snntorch.functional loss functions
   
2. src/utils/neurobench_eval.py (~30 lines)
   - Wrap network with neurobench.models.SNNTorchModel
   - Run neurobench.benchmarks.Benchmark
   - Return all NeuroBench metrics directly

3. src/benchmark_runner.py (~150 lines)
   - Orchestrate algorithm comparison
   - Use torch.profiler for CPU/CUDA timing
   - Run NeuroBench evaluation after training
   - Output JSON results with:
     - Accuracy at checkpoint epochs
     - Timing (wall, CPU, CUDA)
     - NeuroBench metrics (activation sparsity, synaptic ops)
     - Algorithm metadata (is_local, requires_backprop)

4. configs/benchmark_comparison.yaml
   - Configure algorithms to compare: [bptt, stsf]
   - Dataset, architecture, epochs, checkpoint_epochs
   - Output directory

5. Modify main.py
   - Uncomment line 205 to register BPTTTrainer
   - Import BPTTTrainer

VALIDATION:
- BPTT trainer achieves >95% accuracy on MNIST
- NeuroBench metrics compute without errors
- Comparison runs: BPTT vs STSF on same architecture
- JSON output contains all specified metrics

DO NOT IMPLEMENT:
- Custom loss functions (use snntorch.functional)
- Custom timing code (use torch.profiler)  
- Custom SNN metrics (use neurobench.metrics)
- Custom benchmark harness (use neurobench.benchmarks.Benchmark)
```

### Expected Generated Files

- `src/trainers/bptt_trainer.py` — BPTT wrapper
- `src/utils/neurobench_eval.py` — NeuroBench integration
- `src/benchmark_runner.py` — Comparison runner
- `configs/benchmark_comparison.yaml` — Benchmark config
- Modified `main.py` — Register BPTT trainer

---

## Prompt Template for Future Phases

### Prompt 3: Additional Learning Algorithms (Future)

```
Add [ALGORITHM_NAME] learning algorithm trainer.

CONTEXT:
- Reference PROJECT_STATE.md for current structure
- Reference existing trainers: stsf_trainer.py, bptt_trainer.py

REQUIREMENTS:
- Inherit from BaseTrainer
- Implement train_sample(data, target) -> (loss, pred)
- Implement reset()
- Keep learning logic in trainer, not network
- Add to ALGORITHM_INFO in benchmark_runner.py
- Add tests in tests/test_trainers.py
- Document algorithm characteristics:
  - is_local: [True/False]
  - requires_backprop: [True/False]
  - time_complexity: O(...)
  - space_complexity: O(...)

ALGORITHM DESCRIPTION:
[Insert algorithm description and reference paper]
```

### Prompt 4: Convolutional Networks (Future)

```
Add convolutional SNN architecture for image benchmarks.

CONTEXT:
- Reference src/networks/fc_network.py for pattern
- Must work with existing trainers (STSF, BPTT)

REQUIREMENTS:
- Create src/networks/conv_network.py
- Use snnTorch neurons (snn.Leaky) directly
- Return (spk_rec, mem_rec) from forward()
- Implement reset() for membrane states
- Support CIFAR10, DVSGesture input shapes
- Add to network factory in main.py
- Add tests in tests/test_networks.py
```

---

## Usage

To reproduce any development phase:

1. Checkout the previous state tag
2. Copy the exact prompt for the target phase
3. Run the prompt with full context (PROJECT_STATE.md, relevant files)
4. Verify deliverables match expected files
5. Tag the new state

```bash
# Example: Reproduce State 1 → State 2
git checkout v0.1.0-modular
# Apply Prompt 2 with AI assistant
# Verify deliverables
git tag -a v0.2.0-benchmarking -m "NeuroBench + BPTT benchmarking"
```

