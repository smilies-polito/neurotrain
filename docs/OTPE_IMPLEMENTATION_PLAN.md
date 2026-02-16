# OTPE Integration Implementation Plan

## Purpose
Integrate OTPE (Online Training with Predictive Error) into the SNN training benchmarking framework, enabling uniform comparison with BPTT, STSF, DECOLLE, OTTT, and other algorithms.

## Current State

### Already Completed
- ✅ OTPE trainer registered in `main.py` (`get_trainer()` function)
- ✅ OTPE trainer registered in `src/benchmark_runner.py` (ALGORITHM_INFO and instantiation)
- ✅ OTPE trainer registered in `run_all_benchmarks.py` (ALGORITHMS dict)
- ✅ OTPE tests written in `tests/test_trainers.py`

### Missing/Incomplete
- ❌ `src/trainers/otpe_trainer.py` - Implementation file missing
- ❌ `configs/mnist_otpe.yaml` - Configuration file missing
- ❌ Previous implementation attempts had dimension mismatch errors

## Implementation Steps

### Step 1: Create OTPE Trainer (`src/trainers/otpe_trainer.py`)

**Key Requirements:**
- Inherit from `BaseTrainer`
- Follow structure of `ottt_trainer.py` (similar algorithm)
- Preserve original OTPE algorithm logic from JAX/Flax implementation
- Fix dimension handling issues

**Critical Dimension Fixes:**

1. **E and R_hat initialization:**
   - ❌ Wrong: `torch.zeros_like(layer.weight.data)` → `[out_features, in_features]`
   - ✅ Correct: `torch.zeros(layer.in_features, layer.out_features, ...)` → `[in_features, out_features]`
   - Reason: Gradients computed as `[in_features, out_features]`, must match

2. **Error propagation matmul:**
   - ❌ Wrong: `torch.matmul(g_u[l+1] * g_bar[l+1], weight.transpose(0, 1))`
   - ✅ Correct: `torch.matmul(g_u[l+1] * g_bar[l+1], weight)`
   - Reason: PyTorch `Linear.weight` is already `[out_features, in_features]`, which is correct for matmul

3. **Weight update transpose:**
   - Compute `grad_w` as `[in_features, out_features]`
   - Transpose before applying: `grad_w_T = grad_w.transpose(0, 1)` → `[out_features, in_features]`
   - Apply via `_apply_update(layer, grad_w_T, grad_b)`

**Core OTPE Algorithm (preserve from original):**

For each timestep `t`:
1. Forward pass: `spks, mems = self.network(data[t])`
2. Compute eligibility traces:
   - `ds_du_prev = surrogate_derivative(post_mem)` - [batch, out_features]
   - `ds_dtheta_cur = pre_act.transpose(0, 1) @ ds_du_prev / batch_size` - [in_features, out_features]
   - `du_cur_dtheta_cur = pre_act_mean.unsqueeze(1).expand(-1, out_features)` - [in_features, out_features]
   - Update E: `E[l] = du_cur_du_prev * E[l] + du_cur_dtheta_cur` (map_u)
   - Compute ds_dtheta: `ds_dtheta = E[l] * ds_du_prev_mean.unsqueeze(0) + ds_dtheta_cur` (map_s)
   - Update R_hat: `R_hat[l] = sig_tau * R_hat[l] + ds_dtheta` (map_r)
   - Update g_bar: `g_bar[l] = ratio * g_bar[l] + (1-ratio) * (ds_du_prev / sig_tau)`

3. After all timesteps, error propagation:
   - Initialize: `g_u[-1] = error_signal` - [batch, out_features]
   - Backward: `g_u[l] = (g_u[l+1] * g_bar[l+1]) @ weight[l+1]` - [batch, out_features_l]

4. Weight updates:
   - `error_sum = g_u[l].sum(dim=0)` - [out_features]
   - `grad_w = R_hat[l] * error_sum.unsqueeze(0)` - [in_features, out_features]
   - Transpose: `grad_w_T = grad_w.transpose(0, 1)` - [out_features, in_features]
   - Apply: `_apply_update(layer, grad_w_T, grad_b)`

**Methods to implement:**
- `__init__()` - Initialize trainer with parameters
- `surrogate_derivative()` - Sigmoid surrogate (same as OTTT)
- `_infer_trace_decay()` - Infer from network beta
- `_apply_update()` - Handle weight updates with transpose
- `train_sample()` - Core OTPE algorithm
- `reset()` - Reset network and optimizer
- `to(device)` - Move trainer and network to device

### Step 2: Create Configuration File (`configs/mnist_otpe.yaml`)

**Structure (follow `configs/mnist_default.yaml`):**
```yaml
experiment:
  name: "OTPE_MNIST"
  seed: 42
  deterministic: true
  log_dir: "./experiments"

model:
  architecture: "fc"
  layer_sizes: [784, 200, 10]
  beta: 0.9375
  threshold: 1.0
  quantization: false

training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.01
  optimizer: null

trainer:
  name: "otpe"
  trace_decay: null  # Will default to network beta
  surrogate_slope: 10.0
  online_updates: false  # Accumulate over sequence

data:
  dataset: "MNIST"
  timesteps: 10
  data_dir: "./src/Data"
  num_workers: 4

hardware:
  device: "auto"
  mixed_precision: false

checkpoint:
  save_every: 0
  save_best: true
  save_latest: true
  max_keep: 2
```

### Step 3: Verify Existing Registrations

**Files to check:**
1. `main.py` - Verify `OTPETrainer` import and `get_trainer()` registration
2. `src/benchmark_runner.py` - Verify ALGORITHM_INFO entry and instantiation logic
3. `run_all_benchmarks.py` - Verify ALGORITHMS dict entry
4. `tests/test_trainers.py` - Verify test class structure

**Expected registrations:**
- Import: `from trainers.otpe_trainer import OTPETrainer`
- Registration in factory functions
- Test fixtures and test methods

### Step 4: Testing and Validation

**Test checklist:**
- [ ] `pytest tests/test_trainers.py -k otpe` passes
- [ ] `python main.py --config configs/mnist_otpe.yaml` runs without errors
- [ ] `python run_all_benchmarks.py --epochs 10 --algorithms otpe --datasets MNIST` runs successfully
- [ ] No dimension mismatch errors
- [ ] Training logic entirely in trainer (no modifications to FCNetwork)
- [ ] Model remains in networks directory (FCNetwork unchanged)

## Algorithm Preservation Checklist

**Original OTPE logic to preserve:**
- [x] Eligibility trace E updates: `E = du_cur_du_prev * E + du_cur_dtheta_cur` (map_u)
- [x] Predictive error R_hat: `R_hat = sig_tau * R_hat + ds_dtheta` (map_r)
- [x] g_bar running average: `g_bar = ratio * g_bar + (1-ratio) * (ds_du_prev/sig_tau)`
- [x] Weight updates: `g_rec_params = error_signal * R_hat`
- [x] Error propagation: `g_to_send = (error * g_bar) @ kernel`
- [x] Ratio computation: `ratio = (sig_tau * ratio_old) / (sig_tau * ratio_old + 1)`

## Key Implementation Notes

1. **Follow OTTT structure:** Use `ottt_trainer.py` as primary structural reference
2. **Preserve mathematics:** Keep original OTPE equations; only adapt tensor shapes and framework interfaces
3. **Dimension handling:** Critical to get tensor shapes right:
   - Gradients computed as `[in_features, out_features]`
   - PyTorch weights stored as `[out_features, in_features]`
   - Transpose when applying updates
4. **Error propagation:** Use `weight` directly (not transpose) since PyTorch format matches what we need
5. **Device handling:** Implement `to(device)` method properly (same pattern as OTTT)

## Success Criteria

- ✅ OTPE trainer runs without dimension mismatch errors
- ✅ Achieves reasonable accuracy on MNIST (>80% after 50 epochs)
- ✅ Integrates seamlessly with `run_all_benchmarks.py`
- ✅ Works with NeuroBench evaluation pipeline
- ✅ All tests pass
- ✅ Training logic entirely in trainer (no network modifications)

## Files Summary

**Create:**
1. `src/trainers/otpe_trainer.py` (~350-400 lines)
2. `configs/mnist_otpe.yaml`

**Verify (already registered):**
1. `main.py` - Registration
2. `src/benchmark_runner.py` - Algorithm info
3. `run_all_benchmarks.py` - ALGORITHMS dict
4. `tests/test_trainers.py` - Test class
