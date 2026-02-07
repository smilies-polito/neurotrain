# ELL / FELL / BELL Accuracy Gap Analysis (30% vs ~90% on MNIST)

This document compares our implementation of ELL, FELL, and BELL with the reference **Deep Spike Learning with Local Classifiers** (Ma et al., IEEE Trans. Cybernetics 2022) in `existing_implementations/deep_spike_learning_with_local_classifiers-main/`, and identifies likely causes of the accuracy gap.

---

## 1. Input encoding (critical)

### Reference behavior

- **Data**: Raw images from DataLoader with `ToTensor()` only — shape `[B, 1, 28, 28]`, values in [0, 1] (no rate coding).
- **First layer**: The **same** flattened image `x` is used at **every** timestep inside the block:
  ```python
  for step in range(time_window):
      if self.first_layer:
          h = self.encoder(x)   # same x every step
  ```
- So the encoder receives a **constant current** (same 784-d vector for 10 steps). The LIF integrates this constant input over time with decay and reset.

### Our behavior

- **Data**: Rate-coded spikes from `datasets/rate.Rate(T)`: at each timestep we produce **binary** 0/1 spikes via `torch.rand_like(input).le(input).float()`.
- **First layer**: We pass `data[t]` at each step, so the encoder sees a **different** binary vector every timestep (stochastic rate coding).
- So we feed **time-varying binary spikes**; the reference feeds **constant continuous input**.

**Impact**: The reference effectively uses “same image repeated T times” (constant current into LIF). Our setup is true rate coding (different 0/1 pattern each step). The paper’s training dynamics and reported ~90% are for the constant-input regime. Using variable binary input changes the effective input distribution and temporal structure and can strongly hurt performance.

**Fix (conceptually)**:

- **Option A**: For ELL/FELL/BELL only, use a “constant current” input: compute the **mean** over time of the rate-coded batch, then feed that same vector at every step:
  - `x_const = data.mean(dim=0)`  → shape `[B, F]`
  - For `t in range(T)`: call `forward_step_all(x_const)` (same input every step).
- **Option B**: Add a separate data pipeline for local classifiers: no `Rate(T)`; use normalized flattened image `[B, 784]` and let the trainer loop over T steps passing the same `x` each time (matching the reference exactly).

---

## 2. Membrane decay (critical)

### Reference

- `decay = torch.exp(torch.tensor(-1 / args.tau))` with default `--tau 1`:
  - **decay = exp(-1) ≈ 0.368**
- So membrane potential decays **fast** (short memory).

### Ours

- `LocalClassifierNetwork` uses `beta` from config as the LIF decay:
  - `decay = float(beta)` with default `beta=0.9` in configs and benchmark.
  - **decay = 0.9**
- So membrane decays **slowly** (long memory).

**Impact**: With decay 0.9, the neuron integrates over many steps with little decay; with 0.368 it forgets quickly. This changes dynamics and gradient flow and can prevent matching the paper’s results.

**Fix**: For local classifiers when aiming to match the paper, set decay to the reference value, e.g.:

- Add a parameter (e.g. `tau`) and set `decay = exp(-1/tau)` with `tau=1` for the reference setup, **or**
- Set `beta=0.368` (or a config flag that maps to `decay=exp(-1)`) for ELL/FELL/BELL on MNIST.

---

## 3. Architecture

| Aspect        | Reference (MNISTDNN) | Ours (benchmark) |
|---------------|----------------------|-------------------|
| Hidden size   | 800                  | 256               |
| Layers        | 784 → 800 → 10       | 784 → 256 → 10    |

Smaller hidden size (256 vs 800) might reduce capacity but is unlikely alone to explain 30% vs 90%. It can be aligned for a fair comparison once input and decay are fixed.

---

## 4. Time window

- **Reference**: `--time-window 10` (10 steps per sample).
- **Ours**: e.g. `timesteps: 25` in config; loader uses `Rate(T)` so we run 25 steps.

Using more steps with **wrong** input (variable binary) and **wrong** decay can make the loss landscape and dynamics worse. Aligning to 10 steps when matching the reference is reasonable.

---

## 5. Learning rate and optimizer

- **Reference**: `lr=5e-4`, Adam, no weight decay; LR decay at milestones (e.g. 60, 120 for MNIST).
- **Ours**: e.g. `learning_rate: 0.0005` (5e-4), Adam, weight_decay=0.

LR and optimizer are already close; no major discrepancy here.

---

## 6. Surrogate gradient and loss

- **Reference**: `ExponentialSurroGrad` (Heaviside forward, exponential surrogate in backward).
- **Ours**: Same in `utils/surrogate_gradient.py`.

Surrogate and local MSE loss to one-hot are aligned; not a likely cause of the gap.

---

## 7. Per-step update and gradient flow

- **Reference**: Per timestep, per layer: `loss_sup = MSE(y_hat_spike, y_onehot)`, `zero_grad`, `backward`, `optimizer.step()`. ELL detaches membrane/spike between steps; FELL/BELL keep graph (retain_graph as needed).
- **Ours**: Same pattern (per-step, per-layer MSE, backward, step). ELL/FELL/BELL modes and detach behavior in `LocalClassifierBlock` match the reference idea.

So the main algorithmic differences are **input encoding** and **decay**, not the update structure.

---

## 8. Summary and recommended changes

| # | Difference            | Reference              | Ours                    | Severity | Suggested fix |
|---|------------------------|------------------------|-------------------------|----------|----------------|
| 1 | First-layer input      | Same image every step  | Different binary each t | **High** | Use constant input: mean over time or raw image repeated T times (see §1). |
| 2 | LIF decay             | exp(-1) ≈ 0.368 (τ=1)  | β = 0.9                 | **High** | Use decay = exp(-1) (or tau=1) for ELL/FELL/BELL when matching the paper. |
| 3 | Hidden size           | 800                    | 256                     | Medium   | Optionally use 784→800→10 for direct comparison. |
| 4 | Time window           | 10                     | 25                      | Medium   | Use T=10 for ELL/FELL/BELL when reproducing the paper. |

Implementing the two **high**-impact items (constant input for the first layer and decay ≈ 0.368) should be done first; then re-run ELL/FELL/BELL on MNIST and compare again to the reference (~90% in the paper).

---

## 9. Paper-identical mode (implemented)

The codebase now supports **paper-identical** logic for ELL, FELL, and BELL:

1. **Constant input per timestep**: The first layer receives the same vector every step (`x_const = data.mean(dim=0)`), matching the reference’s “same image repeated T times”.
2. **Decay from tau**: When `tau` is set (e.g. `tau: 1`), local classifier networks use `decay = exp(-1/tau)`; otherwise `decay = beta`.
3. **Paper defaults**: For ELL/FELL/BELL, `run_all_benchmarks.py` uses `tau=1`, `local_classifier_timesteps` (10 for MNIST), and for MNIST the optional architecture `[784, 800, 10]` (MNISTDNN).

**Config keys:**

- **tau** (optional): Passed to `benchmark_algorithm` and then to `get_network`; when set, `LocalClassifierNetwork` uses `decay = exp(-1/tau)`. Use `tau: 1` for paper-identical runs.
- **timesteps**: In configs (e.g. `mnist_ell.yaml`), set `data.timesteps: 10` for paper-identical time window.
- **model.tau**: In YAML configs used by `main.py`, set `model.tau: 1` if the loader passes it to the network factory.
- **local_classifier_timesteps**: In dataset configs in `run_all_benchmarks.py` (e.g. MNIST), set to `10` so ELL/FELL/BELL use 10 steps.
- **local_classifier_layer_sizes**: In dataset config (e.g. MNIST), set to `[784, 800, 10]` for paper-identical MNISTDNN when running ELL/FELL/BELL.

Evaluation uses constant input when the network has `constant_input_per_timestep = True` (set on `LocalClassifierNetwork`).

**Input scale (MNIST):** The reference uses raw pixels in [0, 1] (ToTensor only). Our loader uses Normalize then rate coding, so the mean-over-time constant input is on a different scale. For MNIST (784-d input), we denormalize before feeding: `x_const = (x_const * 0.3081 + 0.1307).clamp(0.0, 1.0)` so the constant input matches the reference’s [0, 1] range. This is applied in the ELL/FELL/BELL trainers and in `evaluate()`.

**Batch size:** Paper uses 100 for MNIST. In `run_all_benchmarks.py`, ELL/FELL/BELL use `local_classifier_batch_size` (default 100) when set.

---

## 10. Logic alignment with the reference

| Aspect | Reference | Ours | Match |
|--------|-----------|------|--------|
| First-layer input | Same image every step | Same vector every step; on MNIST via benchmark: raw [0,1] (MNISTLoaderRaw) | Yes for ELL/FELL/BELL on MNIST |
| LIF decay | exp(-1/tau), tau=1 → 0.368 | Same when `tau=1` | Yes |
| Time window | 10 | 10 for local classifiers (MNIST) | Yes |
| LIF dynamics | mem = mem*decay + h - spike*thresh*decay | Same in LocalClassifierBlock | Yes |
| Decoder LIF | Same recurrence | Same | Yes |
| Per-step update | zero_grad, loss_sup.backward(), step each t | Same (reverse-order backward for ELL) | Yes |
| Loss | MSE(y_hat_spike, y_onehot) | Same | Yes |
| Surrogate | ExponentialSurroGrad | Same | Yes |
| Architecture (MNIST) | 784→800→10 | 784→800→10 when ell/fell/bell + MNIST | Yes |
| Learning rate | 5e-4 | 5e-4 for local classifiers in run_all_benchmarks | Yes |
| Batch size | 100 | 100 for local classifiers (default in run_all_benchmarks) | Yes |

**Raw [0,1] data path (MNIST):** When running ELL/FELL/BELL on MNIST via `benchmark_algorithm` (e.g. `run_all_benchmarks.py`), the framework uses a **raw** loader: `MNISTLoaderRaw` in [datasets/mnist_loader.py](src/datasets/mnist_loader.py) with transform ToTensor() + flatten + repeat T (no Normalize, no rate coding). So the input is the same raw [0,1] image repeated T times, matching the reference exactly. `get_loader(..., raw_for_local_classifier=True)` is used for ELL/FELL/BELL on MNIST; the network gets `uses_raw_input = True` and trainers/evaluate skip denormalization.

---

## 11. When accuracy stays ~10% (flat accuracy, loss decreasing)

If after aligning input, decay, and architecture the accuracy remains near chance (~10% on MNIST) while loss decreases, possible causes are **single-class collapse** (model predicts one class) or **weak/zero gradients** in the decoder or encoder.

**Diagnostics (ELL):** Run with first-batch diagnostics enabled:

```bash
SNN_ELL_DEBUG=1 python run_all_benchmarks.py --epochs 2 --datasets MNIST --algorithms ell
```

This prints once on the first batch:

- **Prediction distribution**: class indices and counts (e.g. if all 100 predictions are class 3 → single-class collapse).
- **spk_sum stats**: min, max, mean, std and per-class mean (to see if one output dimension dominates).
- **Gradient norms**: encoder and decoder weight gradient norms at t=0 (zero or very small → gradient flow issue).

**Next steps:**

- If predictions are one class only: check init (e.g. decoder bias), threshold, or try predicting from decoder **membrane sum** instead of spike sum (readout change only).
- If decoder/encoder gradients are near zero: check surrogate, threshold, or scaling of the loss.
- Reproduce the reference script on the same machine (same epochs/seed) and compare curves to confirm the reference reaches ~90% in that environment.
- Confirm readout: we use `spk_sum.argmax(dim=1)` with `spk_sum` = sum over time of last layer’s `y_hat_spike`; reference uses `spike_sum.max(1)[1]` (equivalent).

**Alignment with the reference:** The reference uses **threshold = 1** (`--thresh 1` in `main_train.py`, default 1) and **seed = 1234**. We use the same threshold (1.0) and default to seed=42. If the encoder never fires (membrane &lt; 1) so decoder gradient is 0, try **`--seed 1234`** first to match the reference’s init. A workaround that is **not** in the reference is to lower the threshold (e.g. 0.2) so the encoder can spike and the decoder receives gradients; that fixes the zero-gradient issue but diverges from the paper’s hyperparameters.

---

## 12. Implementation discrepancies (FELL, BELL)

| Item | Reference | Ours | Status |
|------|-----------|------|--------|
| **FELL encoder/decoder** | No detach (full graph). | We **detach** recurrence for FELL: per-step backward + step would otherwise modify parameters in-place, then the next step’s backward (through the previous graph) hits “variable modified by inplace operation”. So we detach for FELL so the run does not crash. | **Divergence** (implementation constraint). |
| **BELL encoder/decoder** | No detach; one backward at end. | No detach: we keep recurrence in the graph; one backward and step at end of the batch. | **Aligned** |
| **FELL readout branch** | Builds `spike_sum` from `decoder_y(spike.detach())` for accuracy. | We use the same `y_hat_spike` as for the loss. | **Minor** (predictions identical; structural only). |

**Summary:** ELL: detach (unchanged). FELL: we detach recurrence so that per-step `optimizer.step()` does not conflict with backprop through time. BELL: we do not detach; we accumulate loss over T and run one backward and one step, so BPTT is preserved and there is no in-place conflict.

---

## 13. Version differences (reference vs our stack)

The reference repo specifies an environment that differs from our project’s:

| Component   | Reference (env.yaml, README) | Our project (requirements.txt, pyproject.toml) |
|------------|-----------------------------|----------------------------------------------|
| **Python** | 3.7.3                       | ≥3.9                                          |
| **PyTorch**| 1.0.1                       | ≥2.0.0                                        |
| **torchvision** | 0.2.1                 | ≥0.15.0                                       |
| **NumPy**  | 1.16.4                      | ≥1.24.0                                       |
| **Seed**   | 1234 (main_train.py default)| 42 (run_all_benchmarks default)               |

**Impact on spiking / init:**

- **nn.Linear init:** Both PyTorch 1.0 and 2.x use **Kaiming uniform** for `nn.Linear` weights, so the *type* of init is the same. Small implementation or default-gain differences between 1.0.1 and 2.x could still change the **scale** of the initial weights slightly, and thus the scale of `h = W @ x` and how often membrane crosses the threshold.
- **Random seed:** We use 42, the reference 1234. Different seeds → different draws of W → different initial membrane distribution → different fraction of neurons above threshold at step 0. This is often the dominant cause of “encoder fires in reference, doesn’t fire in ours.”
- **Python / NumPy:** Different versions can change RNG sequence (e.g. `numpy.random` vs `torch.manual_seed` and data-loading order), so even with the same seed number, the effective init and data order can differ across environments.

**Summary:** The reference was developed and run on **PyTorch 1.0.1** and **Python 3.7** with **seed 1234**. We run on **PyTorch ≥2.0** and **Python ≥3.9** with **seed 42**. Version and seed differences can change the initial encoder weights and thus whether the encoder spikes with threshold=1. To reduce that gap: use `--seed 1234` when benchmarking; for closer reproduction, consider a dedicated env with PyTorch 1.x and Python 3.7 (e.g. the reference’s env.yaml).

---

## 14. References

- **Paper**: Ma et al., “Deep Spike Learning With Local Classifiers,” IEEE Trans. Cybernetics, 2022.
- **Code**: `existing_implementations/deep_spike_learning_with_local_classifiers-main/`  
  - `main_train.py`, `local_linear_ELL.py`, `local_linear_FELL.py`, `load_dataset.py`, `models/MNISTDNN.py`.
