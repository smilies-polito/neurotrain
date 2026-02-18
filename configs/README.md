# Configs: Benchmark Suite + Network Blueprints

This repo’s new benchmark flow is configured by two things:

1. `configs/benchmarking.yaml`
2. `configs/networks/*.yaml` (one file per network backbone)

These are read by `benchmarking.py`.

---

## Big Picture (How They Combine)

The runner builds an experiment matrix:

```text
datasets (from configs/benchmarking.yaml)
  x trainers (from configs/benchmarking.yaml)
  x networks (from configs/networks/*.yaml)
  ------------------------------------------------
  => experiment list (invalid combos are skipped)
```

Each scheduled experiment becomes:

```text
dataset + trainer + network
  -> instantiate network from configs/networks/<name>.yaml
  -> instantiate trainer from configs/benchmarking.yaml (trainer entry)
  -> train for N epochs
  -> evaluate accuracy
  -> optionally run NeuroBench
  -> write results to benchmark_results/full_benchmark_<timestamp>/
```

---

## 1) `configs/benchmarking.yaml`

### Structure

```text
benchmarking.yaml
  experiment: run identity + output directory + seed
  execution:  global training knobs + runtime behavior
  data:       dataset list + per-dataset defaults
  trainers:   trainer registry + compatibility rules + trainer params
```

### `experiment`

```yaml
experiment:
  name: full_benchmark
  output_dir: ./benchmark_results
  seed: 42
  deterministic: true
```

- `name`: logical name (mostly informational; output folder is timestamped).
- `output_dir`: root directory where run artifacts are written.
- `seed`: seed reset before each experiment for comparability.
- `deterministic`: passed to `set_all_seeds(..., deterministic=...)`.

### `execution`

```yaml
execution:
  epochs: 1
  batch_size: 128
  learning_rate: 0.001
  timesteps: 25
  max_train_batches: null
  max_test_batches: null
  device: cuda
  single_process_data_loading: true
  continue_on_error: true
  run_neurobench: false
```

- `epochs`: epochs per experiment.
- `batch_size`: dataloader batch size.
- `learning_rate`: used for optimizers/manual updates (depending on trainer).
- `timesteps`: default rate-coding timesteps `T` (dataset can override).
- `max_train_batches`: if set, stops each epoch early after N batches (smoke tests).
- `max_test_batches`: if set, evaluates on only the first N test batches.
- `device`: `cpu`, `cuda`, `cuda:0`, or `auto` (falls back if unavailable).
- `single_process_data_loading`: when `true`, forces `num_workers=0` internally to
  avoid multiprocessing restrictions in some environments.
- `continue_on_error`: keep running remaining experiments even if one fails.
- `run_neurobench`: if `true`, runs NeuroBench at the end of each experiment.

### `data`

```yaml
data:
  datasets:
    - MNIST
  dataset_defaults:
    MNIST:
      input_shape: [1, 28, 28]
      input_size: 784
      num_classes: 10
      timesteps: 25
    ...
```

- `datasets`: list of datasets to include in the matrix.
- `dataset_defaults.<DATASET>`:
  - `input_shape`: canonical `(C, H, W)` for image-like datasets.
  - `input_size`: flattened feature size (used by FC/RNN-style configs).
  - `num_classes`: output classes (used to patch layer sizes / network kwargs).
  - `timesteps`: per-dataset default `T` (overrides `execution.timesteps`).

### `trainers`

Each key under `trainers:` defines one algorithm entry, for example:

```yaml
trainers:
  bptt:
    module: trainers.bptt_trainer
    class_name: BPTTTrainer
    enabled: true

    # Compatibility filters (runner uses these to skip invalid combos)
    requires_all_tags: []
    excludes_any_tags: []
    requires_network_attrs: [reset]
    allowed_architectures: []

    # Execution mode
    training:
      requires_grad: true
      use_optimizer: true
      optimizer: adam

    # Trainer-specific kwargs (passed to the trainer constructor)
    params:
      loss_type: ce_rate
```

#### Compatibility keys

These determine whether `(trainer, network, dataset)` becomes an experiment.

- `requires_all_tags`: list of BaseSNN semantic tags that must be true on the
  instantiated network (e.g. `fully_connected`, `convolutional`, `recurrent`).
- `excludes_any_tags`: tags that must *not* be present.
- `requires_network_attrs`: attribute names that must exist on the network object
  (simple `hasattr` checks; useful for structural contracts).
- `allowed_architectures`: optional allow-list for `model.architecture` coming
  from the network blueprint (if empty, any architecture is allowed).

#### `training` keys

- `requires_grad`: wraps the training loop in `torch.set_grad_enabled(...)`.
- `use_optimizer`: whether the runner should create and pass an optimizer.
- `optimizer`: optimizer name (`adam`, `sgd`, `nag`, `rmsprop`, or `null`).

#### `params` keys

Everything here is passed as kwargs to the trainer constructor (after a safety
filter that drops unsupported kwargs).

---

## 2) `configs/networks/*.yaml`

Each file in `configs/networks/` defines one network backbone *unambiguously*.
The runner reads all `*.yaml` here and uses them as the “networks” dimension of
the experiment matrix.

### Structure

```text
configs/networks/<network>.yaml
  name:             unique identifier used in experiment ids
  description:      human description
  enabled:          include/exclude from matrix

  model:            high-level model config (kept for bookkeeping)
  network_kwargs:   actual constructor kwargs for the benchmarking network class

  dataset_overrides:
    <DATASET>:
      model:         patch model fields for this dataset
      network_kwargs:patch ctor kwargs for this dataset

  tags:
    expected:       documentation-only tag list (not used by runner)
```

### `name` / `enabled`

- `name`: used in experiment ids like `MNIST__bptt__fc_snn`.
- `enabled`: if `false`, the file is ignored entirely.

### `model`

This section mirrors the general config style used elsewhere in the repo. The
runner mainly uses `model.architecture` for compatibility checks and for clarity
in saved manifests.

Important field:
- `model.architecture`: must be one of:
  - `fc_snn`, `r_snn`, `conv_snn`, `vg11_snn`

### `network_kwargs` (what actually builds the network)

This is the most important part. It is passed to the corresponding class:

```text
architecture -> class
  fc_snn     -> networks.benchmarking.fc_snn.FCSNN
  r_snn      -> networks.benchmarking.r_snn.RSNN
  conv_snn   -> networks.benchmarking.conv_snn.ConvSNN
  vg11_snn   -> networks.benchmarking.vg11_snn.VG11SNN
```

#### Shape conventions

The dataloaders used by the benchmark runner provide:

- FC/RNN-style inputs: one timestep shaped `(B, F)` (flattened features)
- Conv/VGG-style inputs: one timestep shaped `(B, C, H, W)`

So:

- For `fc_snn` / `r_snn`: `network_kwargs.in_shape` should typically be `[input_size]`
  (e.g. MNIST uses `[784]`).
- For `conv_snn` / `vg11_snn`: `network_kwargs.in_shape` should be `[C, H, W]`
  (e.g. CIFAR10 uses `[3, 32, 32]`).

### `dataset_overrides`

This allows one network blueprint to adapt to multiple datasets without copying
the whole file.

Example pattern:

```yaml
dataset_overrides:
  CIFAR10:
    network_kwargs:
      in_shape: [3, 32, 32]
      num_classes: 10
```

The runner deep-merges:

1. base `model` + override `model`
2. base `network_kwargs` + override `network_kwargs`

Then it also patches:
- last entry of `model.layer_sizes` to match `dataset_defaults.<DATASET>.num_classes`
- for FC/RNN, first entry of `model.layer_sizes` to match `dataset_defaults.<DATASET>.input_size`

---

## Minimal examples

### Run everything from config (Makefile)

```bash
make bench-full
```

### CLI smoke test (small, fast)

```bash
python3 benchmarking.py \\
  --device cpu \\
  --epochs 1 \\
  --timesteps 2 \\
  --max-train-batches 2 \\
  --max-test-batches 2 \\
  --datasets MNIST \\
  --algorithms bptt,ottt \\
  --networks fc_snn,conv_snn
```

---

## Common pitfalls

- `fc_snn`/`r_snn` using image-shaped `in_shape` (e.g. `[1, 28, 28]`) will fail
  if the loader is emitting flattened `(B, 784)` timesteps.
- Some trainers only work on specific semantics (e.g. feed-forward-only vs
  recurrent). Use the `requires_all_tags` / `excludes_any_tags` filters in
  `configs/benchmarking.yaml` to make those constraints explicit.
