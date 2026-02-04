# Prompts Summary

> Reproducible prompts for each development phase. Each prompt is designed to be self-contained and generate the required changes.

---

## Prompt History

| # | Phase | State Transition | Status |
|---|-------|------------------|--------|
| 1 | Reproducibility & Logging | State 0 → State 1 | ✅ Complete |
| 2 | BPTT Benchmarking Baseline | State 1 → State 2 | ✅ Complete |
| 3 | Full Benchmark Suite | State 2 → State 3 | 🔲 Pending |
| 4 | ELL / FELL / BELL Integration | State 2/3 → State 4 | 🔲 Next |

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

## Prompt 2: BPTT Benchmarking Baseline ✅ COMPLETE

**Purpose:** Add BPTT baseline and NeuroBench integration for algorithm benchmarking.

**State Transition:** State 1 (Modular + Reproducibility) → State 2 (BPTT + NeuroBench)

**Completion Date:** 2025-12-06

### Exact Prompt (Refined)

```
Add BPTT baseline and NeuroBench v2.1.0 benchmarking infrastructure.

CONTEXT:
- Reference docs/states_summary.md for current codebase structure
- See "Key Code Patterns" section below for implementation details

CRITICAL IMPLEMENTATION NOTES (lessons learned):

1. NEUROBENCH v2.1.0 API CHANGES
   - Use metric_list=[static_metrics, workload_metrics] (two sublists)
   - Pass metric CLASSES not instances: [Footprint, ParameterCount] not [Footprint()]
   - preprocessors/postprocessors must be [] not None
   - SynapticOperations returns dict: {"Effective_MACs": ..., "Dense": ...}

2. SNNTORCHMODEL WRAPPER
   - FCNetwork doesn't work directly with SNNTorchModel
   - Create NeuroBenchWrapper class that:
     - Stores device and moves inputs with x.to(self.device)
     - Handles temporal loop internally
     - Returns spikes as [timesteps, batch, classes]
   - Use SNNTorchModel(wrapper, custom_forward=True)

3. POSTPROCESSOR FOR CLASSIFICATION
   - Create spike_to_prediction function:
     - Input: [batch, timesteps, classes]
     - Sum over time, argmax, return [batch] on CPU
     - MUST return on CPU to match labels from DataLoader

4. TIMING APPROACH
   - DO NOT use torch.profiler (10x+ overhead)
   - Use time.perf_counter() with torch.cuda.synchronize() before/after
   - Reports accurate wall-clock time for GPU operations

5. BPTT TRAINER
   - Use snntorch.functional loss functions (ce_rate_loss, etc.)
   - Recreate optimizer in .to(device) method (optimizer holds parameter refs)
   - Use torch.enable_grad() context in train_sample

DELIVERABLES:

1. src/trainers/bptt_trainer.py (~150 lines)
   - Match BaseTrainer interface (train_sample, reset, to)
   - Use snntorch.functional loss functions
   - Handle optimizer recreation on device change
   
2. src/utils/neurobench_eval.py (~230 lines)
   - NeuroBenchWrapper class with device handling
   - spike_to_prediction postprocessor
   - run_neurobench() function with proper metric_list format

3. src/benchmark_runner.py (~300 lines)
   - Single-dataset benchmark runner
   - time.perf_counter() timing with cuda.synchronize()

4. run_all_benchmarks.py (~340 lines)
   - Multi-dataset orchestrator
   - Two summary tables: Training + NeuroBench metrics
   - Show Effective MACs, Dense MACs, and Savings %
   - JSON output to benchmark_results/

5. src/datasets/neurobench_loaders.py (~200 lines)
   - Loaders for SpeechCommands, WISDM, PrimateReaching, MackeyGlass
   - Note: Some require pytorch_lightning/torchcodec (add to .def)

6. configs/benchmark_comparison.yaml
   - Configure algorithms to compare: [bptt, stsf]
   - Dataset, architecture, epochs, checkpoint_epochs

7. Modify existing files:
   - main.py: Import and register BPTTTrainer
   - src/datasets/get_loader.py: Add NeuroBench loaders
   - src/datasets/cifar10_loader.py: Add import torch
   - src/datasets/svhn_loader.py: Fix SVHN import
   - src/utils/config.py: Add NeuroBench datasets to valid list
   - src/snn-training-benchmarking.def: Add torchaudio, torchcodec, pytorch_lightning
   - .gitignore: Add benchmark_results/, src/Data/

VALIDATION:
- BPTT trainer achieves >95% accuracy on MNIST
- NeuroBench metrics compute without device errors
- Summary tables show Params, Footprint, ActSpars, Eff MACs, Dense MACs, Savings
- JSON output contains all specified metrics

DO NOT:
- Use torch.profiler (too slow)
- Pass None to preprocessors/postprocessors (use [])
- Use old NeuroBench API (static_metrics= kwarg)
- Forget cuda.synchronize() for GPU timing
```

### Generated Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/trainers/bptt_trainer.py` | 156 | BPTT with snnTorch functional API |
| `src/utils/neurobench_eval.py` | 231 | NeuroBench v2.1.0 wrapper |
| `src/benchmark_runner.py` | ~300 | Single-dataset benchmark |
| `run_all_benchmarks.py` | ~340 | Multi-dataset orchestrator |
| `src/datasets/neurobench_loaders.py` | ~200 | NeuroBench dataset loaders |
| `configs/benchmark_comparison.yaml` | 33 | Benchmark configuration |

### Modified Files

| File | Change |
|------|--------|
| `main.py` | Added BPTTTrainer import and registration |
| `src/datasets/get_loader.py` | Added NeuroBench dataset loaders |
| `src/datasets/cifar10_loader.py` | Added `import torch` |
| `src/datasets/svhn_loader.py` | Fixed import: MNIST → SVHN |
| `src/utils/config.py` | Added NeuroBench datasets to valid list |
| `src/utils/experiment_logger.py` | Fixed TensorBoard hparams sanitization |
| `src/snn-training-benchmarking.def` | Added torchaudio, torchcodec, pytorch_lightning |
| `.gitignore` | Added benchmark_results/, src/Data/ |

### Key Bugs Fixed

| Issue | Cause | Fix |
|-------|-------|-----|
| `unexpected keyword argument 'static_metrics'` | NeuroBench v2.x API change | Use `metric_list=[static, workload]` |
| `'NoneType' object is not iterable` | None passed to processors | Use `[]` not `None` |
| `mat2 is on cuda:0, different from cpu` | Input tensor on wrong device | Add `x.to(self.device)` in wrapper |
| `too many values to unpack` | Wrong return format | Use `custom_forward=True`, return `[T,B,C]` |
| Profiler 10x overhead | torch.profiler too heavy | Use `time.perf_counter()` + `cuda.synchronize()` |
| SynapticOperations shows dict | API returns dict not scalar | Extract `Effective_MACs` and `Dense` keys |

### Key Code Patterns

#### NeuroBench v2.1.0 Benchmark Setup
```python
from neurobench.metrics.static import Footprint, ConnectionSparsity, ParameterCount
from neurobench.metrics.workload import ActivationSparsity, SynapticOperations, MembraneUpdates

# CORRECT: metric_list with two sublists, pass CLASSES not instances
benchmark = Benchmark(
    model=nb_model,
    dataloader=test_loader,
    preprocessors=[],              # Must be [], not None
    postprocessors=[spike_to_prediction],
    metric_list=[[Footprint, ParameterCount], [ActivationSparsity, SynapticOperations]],
)
```

#### NeuroBench Wrapper Pattern
```python
class NeuroBenchWrapper(torch.nn.Module):
    def __init__(self, network, num_timesteps, device):
        super().__init__()
        self.network = network
        self.num_timesteps = num_timesteps
        self.device = device  # CRITICAL: Store device

    def forward(self, x):
        x = x.to(self.device)  # CRITICAL: Move input to correct device
        self.network.reset()
        all_spikes = []
        for t in range(self.num_timesteps):
            spks, _ = self.network(x[t])
            all_spikes.append(spks[-1])
        return torch.stack(all_spikes, dim=0)  # [T, B, classes]

# CRITICAL: Use custom_forward=True
nb_model = SNNTorchModel(wrapped_model, custom_forward=True)
```

#### Postprocessor for Classification
```python
def spike_to_prediction(preds: torch.Tensor) -> torch.Tensor:
    """[batch, T, classes] -> [batch] on CPU"""
    spike_sum = preds.sum(dim=1)
    return spike_sum.argmax(dim=1).cpu()  # CRITICAL: Must be on CPU
```

#### GPU Timing Pattern
```python
if use_cuda:
    torch.cuda.synchronize()  # Wait for GPU before timing
epoch_start = time.perf_counter()

train_one_epoch(trainer, train_loader, device)

if use_cuda:
    torch.cuda.synchronize()  # Wait for GPU after work
epoch_ms = (time.perf_counter() - epoch_start) * 1000.0
```

---

## Prompt 3: Full Benchmark Suite (NEXT)

**Purpose:** Add regression support, enable all NeuroBench datasets, rebuild container.

**State Transition:** State 2 (BPTT + NeuroBench) → State 3 (Full Benchmark Suite)

### Exact Prompt

```
Complete the benchmark suite with regression support and all NeuroBench datasets.

CONTEXT:
- Reference docs/states_summary.md for current structure
- See docs/prompts_summary.md Prompt 2 for NeuroBench integration details
- Current state: BPTT + NeuroBench working for classification datasets

DELIVERABLES:

1. REGRESSION TASK SUPPORT
   - Modify BPTTTrainer and STSFTrainer to support regression:
     - Add task_type parameter: "classification" | "regression"
     - For regression: use MSE loss, continuous targets
     - For classification: existing behavior
   - Modify train_sample() return: for regression, pred is continuous values
   - Add regression accuracy metric (e.g., R² score, MSE)

2. REBUILD CONTAINER
   - Verify snn-training-benchmarking.def includes:
     - torchaudio (for SpeechCommands)
     - torchcodec (for torchaudio backend)
     - pytorch_lightning (for WISDM)
   - Build and test container:
     singularity build --fakeroot snn-training-benchmarking.sif snn-training-benchmarking.def

3. ENABLE ALL NEUROBENCH DATASETS
   - Verify SpeechCommandsLoader works in new container
   - Verify WISDMLoader works in new container
   - Enable PrimateReaching (regression)
   - Enable MackeyGlass (regression)
   - Update run_all_benchmarks.py to include all datasets

4. NEUROBENCH REGRESSION METRICS
   - Update neurobench_eval.py for regression tasks:
     - Use MSE metric instead of ClassificationAccuracy
     - Adjust postprocessor for continuous outputs
   - Add regression results to summary tables

5. SUMMARY TABLE UPDATES
   - Add task type column (Classification/Regression)
   - For regression: show MSE instead of Accuracy
   - Keep NeuroBench efficiency metrics for all tasks

VALIDATION:
- SpeechCommands loads and trains without errors
- WISDM loads and trains without errors
- PrimateReaching trains with MSE loss, reports R² or MSE
- MackeyGlass trains with MSE loss, reports R² or MSE
- All 9 datasets run in single benchmark invocation

DO NOT:
- Break existing classification functionality
- Remove any working features
- Change NeuroBench wrapper unless necessary for regression
```

### Expected Generated Files

- Modified `src/trainers/bptt_trainer.py` — Add regression support
- Modified `src/trainers/stsf_trainer.py` — Add regression support
- Modified `src/utils/neurobench_eval.py` — Add regression metrics
- Modified `run_all_benchmarks.py` — Enable all datasets, add task type column
- Rebuilt `src/snn-training-benchmarking.sif` — Container with all dependencies

---

## Prompt 4: Integrate ELL, FELL, and BELL (Local Classifier Algorithms)

**Purpose:** Integrate the three local-learning algorithms from "Deep Spike Learning with Local Classifiers" (Ma et al., IEEE T-Cyb 2022) into the benchmarking framework, enabling fair comparison with BPTT, STSF, DECOLLE, and OTTT.

**State Transition:** State 2/3 → State 4 (ELL/FELL/BELL Integration)

**Reference Implementation:** `existing_implementations/deep_spike_learning_with_local_classifiers-main/`

### Exact Prompt

```
Integrate ELL, FELL, and BELL learning algorithms into the SNN training benchmarking framework.

## Context

- **Current project state:** docs/states_summary.md (State 2: BPTT + NeuroBench working)
- **Reference implementation:** existing_implementations/deep_spike_learning_with_local_classifiers-main/
- **Pattern to follow:** src/trainers/decolle_trainer.py (local learning, BaseTrainer interface)
- **Key reference files:**
  - local_linear_ELL.py, local_linear_FELL.py, local_linear_BELL.py — core learning rules
  - linearFA.py — Feedback Alignment layer (optional, for --fa mode)
  - surrogate_gradient.py — ExponentialSurroGrad (Heaviside with surrogate)
  - main_train.py — training loop, model interface
  - models/MNISTDNN.py — single-layer DNN with local classifier

**Algorithm summary:**
- **ELL (Event-based Local Learning):** Per-layer local classifiers, MSE to one-hot; membrane/spike detached between timesteps (no temporal gradients). Most bio-plausible, fully local.
- **FELL (Full Event-based Local Learning):** Same structure; retain_graph=True, gradients flow through time; per-step backward.
- **BELL (Backprop Event-based Local Learning):** Same structure; no detach; single backward at end of time window (full BPTT through local losses).

**Data format:** Framework provides data as [T, B, F] (rate-coded spikes). Reference uses [B, F] with same input each timestep for "real" encoding. Adapt first layer to accept data[t] at each step (spike input) to match framework conventions.

## Objective

Add ELL, FELL, and BELL as plug-and-play trainers that conform to BaseTrainer, work with the existing data pipeline, and participate in run_all_benchmarks.py and NeuroBench evaluation.

## Requirements

1. **Shared building blocks**
   - Port ExponentialSurroGrad from surrogate_gradient.py to src/utils/surrogate_gradient.py (or equivalent).
   - Port LinearFA from linearFA.py to src/utils/linear_fa.py (device-agnostic, no .cuda() calls).
   - Create LocalLossBlockLinear (or equivalent) that supports ELL/FELL/BELL variants via a mode flag or subclass.

2. **Network architecture**
   - Create src/networks/local_classifier_network.py: feedforward SNN with per-layer encoder + decoder_y (local classifier).
   - Support configurable layer_sizes matching existing FC architecture (e.g. [784, 256, 10]).
   - First layer receives data[t] at each timestep; hidden layers receive previous layer spikes. Output: spike_sum over time, argmax for prediction.
   - Implement reset() for membrane/state; implement forward(data, target, target_onehot, time_window) returning (error_percent, loss) or equivalent for training.
   - For evaluation: forward in eval mode returns spike_sum (no target needed); predict via argmax.

3. **Trainers**
   - Create src/trainers/ell_trainer.py, fell_trainer.py, bell_trainer.py (or one ell_fell_bell_trainer.py with algorithm="ell"|"fell"|"bell").
   - Each trainer:
     - Wraps LocalClassifierNetwork (or builds it from config).
     - Implements BaseTrainer: train_sample(data, target) -> (loss, pred), reset(), and parameters needed for .to(device).
   - Input: data [T, B, F], target [B]. Build target_onehot internally.
   - Call network forward in training mode; extract loss and pred (argmax of spike_sum).
   - Handle learning rate scheduling if needed (reference uses set_learning_rate on model).

4. **Registration and config**
   - Register ell, fell, bell in main.py get_trainer().
   - Add to benchmark_runner.py ALGORITHM_INFO and get_trainer logic.
   - Add configs: configs/mnist_ell.yaml, configs/mnist_fell.yaml, configs/mnist_bell.yaml (or extend benchmark_comparison.yaml).
   - Add ell, fell, bell to configs/benchmark_comparison.yaml algorithms list (optional, for full comparison).

5. **Compatibility**
   - Use framework’s Config for layer_sizes, timesteps, lr, beta/threshold, etc.
   - Support existing datasets (MNIST, FashionMNIST, CIFAR10) via get_loader; data remains [T, B, F].
   - Ensure NeuroBench evaluation works: LocalClassifierNetwork must support the temporal forward pass expected by NeuroBenchWrapper (reset, then for t in T: out = network(x[t])).

## Deliverables

- [ ] src/utils/surrogate_gradient.py (~20 lines) — ExponentialSurroGrad
- [ ] src/utils/linear_fa.py (~70 lines) — LinearFA, device-agnostic
- [ ] src/networks/local_classifier_network.py (~200–250 lines) — LocalClassifierNetwork with ELL/FELL/BELL modes
- [ ] src/trainers/ell_trainer.py (~80 lines) — ELLTrainer (or unified trainer with algorithm param)
- [ ] src/trainers/fell_trainer.py (~80 lines) — FELLTrainer
- [ ] src/trainers/bell_trainer.py (~80 lines) — BELLTrainer
- [ ] configs/mnist_ell.yaml, configs/mnist_fell.yaml, configs/mnist_bell.yaml
- [ ] Modify: main.py — register trainers
- [ ] Modify: src/benchmark_runner.py — ALGORITHM_INFO, get_trainer
- [ ] Modify: src/utils/config.py — add ell, fell, bell to valid trainer names if needed
- [ ] tests/test_trainers.py — add tests for ELL/FELL/BELL (smoke: train a few batches, check loss/pred shape)

## Constraints

- Reuse framework patterns: BaseTrainer, get_loader, Config. Do NOT replicate main_train.py’s custom loop.
- Use type hints and docstrings (Google style). Follow Black (88), flake8.
- Do NOT add new heavy dependencies; use PyTorch only for LinearFA and surrogate gradient.
- Keep learning logic in trainers/networks; do not pollute LearningAlgorithms.py.
- For NeuroBench: ensure network has reset() and forward(x) returning spikes (or adapt NeuroBenchWrapper if needed).

## Validation

- ELL trainer achieves >90% on MNIST after 50 epochs (same config as other algorithms).
- FELL and BELL achieve comparable or better accuracy.
- `python run_all_benchmarks.py --epochs 10` runs without errors when ell/fell/bell are in algorithms.
- NeuroBench metrics compute for ell/fell/bell (no device/shape errors).
- All new tests pass: pytest tests/test_trainers.py -k ell.

## Out of Scope

- Convolutional local classifier networks (local_conv_*); FC/DNN only for this prompt.
- Feedback alignment (--fa) can be optional; prioritize default (no FA) first.
- Changes to DECOLLE, STSF, BPTT, OTTT, or e-prop implementations.
```

### Expected Generated Files

| File | Purpose |
|------|---------|
| `src/utils/surrogate_gradient.py` | Exponential surrogate gradient for Heaviside |
| `src/utils/linear_fa.py` | Feedback Alignment linear layer |
| `src/networks/local_classifier_network.py` | SNN with per-layer local classifiers |
| `src/trainers/ell_trainer.py` | ELL trainer |
| `src/trainers/fell_trainer.py` | FELL trainer |
| `src/trainers/bell_trainer.py` | BELL trainer |
| `configs/mnist_ell.yaml` | ELL config |
| `configs/mnist_fell.yaml` | FELL config |
| `configs/mnist_bell.yaml` | BELL config |

### Key Integration Notes

1. **Data format:** Reference first layer uses same x each timestep; framework uses `data[t]`. Use `data[t]` for first layer input at step t to stay consistent with rate coding.
2. **Loss:** Local MSE to one-hot targets; sum over timesteps. Return scalar loss from `train_sample`.
3. **Prediction:** Spike sum over time, argmax. Same as BPTT/STSF for fairness.
4. **NeuroBench:** LocalClassifierNetwork must expose `reset()` and a forward that can be called per-timestep for wrapper compatibility.

---

## Prompt Template for Future Phases

### Prompt 5: Additional Learning Algorithms (Future)

```
Add [ALGORITHM_NAME] learning algorithm trainer.

CONTEXT:
- Reference docs/states_summary.md for current structure
- Reference existing trainers: src/trainers/stsf_trainer.py, src/trainers/bptt_trainer.py

REQUIREMENTS:
- Inherit from BaseTrainer
- Implement train_sample(data, target) -> (loss, pred)
- Implement reset() and to(device)
- Support both classification and regression (task_type parameter)
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

### Prompt 6: Convolutional Networks (Future)

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
3. Run the prompt with full context (docs/states_summary.md, docs/prompts_summary.md)
4. Verify deliverables match expected files
5. Tag the new state

```bash
# Example: Reproduce State 1 → State 2
git checkout v0.1.0-modular
# Apply Prompt 2 with AI assistant
# Verify deliverables
git tag -a v0.2.0-benchmarking -m "BPTT baseline + NeuroBench integration"

# Example: Reproduce State 2 → State 3
git checkout v0.2.0-benchmarking
# Apply Prompt 3 with AI assistant
# Verify deliverables
git tag -a v0.3.0-complete -m "Full benchmark suite with regression support"

# Example: Reproduce ELL/FELL/BELL Integration (Prompt 4)
git checkout v0.2.0-benchmarking  # or v0.3.0-complete
# Apply Prompt 4 with AI assistant (full context: docs/, existing_implementations/)
# Verify: pytest tests/test_trainers.py -k ell
# Verify: python run_all_benchmarks.py --epochs 10
git tag -a v0.4.0-ell-fell-bell -m "ELL, FELL, BELL local classifier algorithms"
```
