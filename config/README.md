# Config Directory

This directory holds all configuration files for the benchmarking framework.

## Structure

```
config/
  benchmarking.yaml       — example benchmarking mode input
  experiments.yaml        — example custom mode input
  default/
    trainers/             — one YAML per trainer (default hyperparameters)
    models/               — one YAML per network (with per-dataset sections)
    datasets/             — one YAML per dataset (batch size, timesteps, etc.)
  custom/                 — user-created experiment files (not tracked by git)
```

---

## Modes

### Benchmarking mode

Run `run_exp_campaign.py --benchmarking config/benchmarking.yaml`.

The file lists which trainers, models and datasets to compare.
An empty list means "all available defaults".

```yaml
trainers: [bptt, stsf]   # empty = all in config/default/trainers/
models:   [fc_snn]
datasets: [MNIST, FashionMNIST]
runtime:
  epochs: 10
  device: cuda
  seed: 42
```

### Custom mode

Run `run_exp_campaign.py --custom config/experiments.yaml`.

Each top-level key is one named experiment. Values override the defaults.

```yaml
my_experiment:
  name: bptt_small_mnist        # human label (optional)
  opt: false                    # set true to enable Optuna
  trainer:
    name: bptt
    lr: 5.0e-4                  # overrides default lr
  model:
    name: fc_snn
    hidden_sizes: [128]         # overrides default hidden_sizes in default section
  dataset:
    name: MNIST
    T: 25
  runtime:
    epochs: 5
    device: cuda
```

---

## Model configs — per-dataset sections

A model YAML can have a `default` section plus dataset-specific sections.
When building an experiment, the dataset-specific section is merged on top of
`default` (override wins). Only specify the values that differ from the default.

```yaml
# config/default/models/fc_snn.yaml
default:
  name: fc_snn
  layer_sizes: [784, 256, 10]
  beta: 0.9
  threshold: 1.0

mnist:
  layer_sizes: [784, 128, 10]   # smaller for MNIST

fashionmnist:
  layer_sizes: [784, 800, 10]   # larger for FashionMNIST
```

---

## Hyper-parameter optimisation with Optuna

### 1 — Define the search space in YAML

Replace any plain scalar with a *tunable block*:

```yaml
# float with log-scale sampling
lr:
  value: 1.0e-3    # default for normal (opt: false) runs
  type: float
  min: 1.0e-5
  max: 1.0e-1
  log: true        # optional; omit for linear scale

# integer
batch_size:
  value: 256
  type: int
  min: 32
  max: 512
  step: 32         # optional; default 1

# categorical
loss_type:
  value: ce_rate
  type: categorical
  list: [ce_rate, mse_count, ce_count]

# not tunable — keeps the value, documents intent
hidden_sizes:
  value: [256]
  type: null
```

Tunable blocks can appear anywhere in the `trainer`, `model`, or `dataset`
sections of any config.  Blocks with `type: null` (or no `type` key) are
treated identically to plain values in both normal and Optuna runs.

### 2 — Enable Optuna

**Custom mode** (`experiments.yaml`): set `opt: true` on any experiment.

```yaml
my_tuned_exp:
  opt: true
  optuna:           # optional: override global Optuna settings
    n_trials: 50
    sampler: tpe
  trainer:
    name: bptt
    lr: { value: 1e-3, type: float, min: 1e-5, max: 1e-1, log: true }
  ...
```

**Benchmarking mode** (`benchmarking.yaml`): set top-level `opt: true` to run
an Optuna study for every generated `(trainer × model × dataset)` experiment.

```yaml
opt: true
optuna:
  n_trials: 20
  sampler: tpe
trainers: [bptt]
models: [fc_snn]
datasets: [MNIST]
```

### 3 — Configure the study (`optuna:` block)

| Key         | Default      | Description                                     |
|-------------|------------- |-------------------------------------------------|
| `n_trials`  | `20`         | Number of trials to run per experiment          |
| `direction` | `maximize`   | `maximize` or `minimize`                        |
| `sampler`   | `tpe`        | `tpe` \| `random` \| `cmaes`                   |
| `pruner`    | `null`       | `median` \| `hyperband` \| `null` (no pruning) |
| `timeout`   | `null`       | Max seconds per study (`null` = unlimited)      |
| `storage`   | `null`       | SQLite URL, e.g. `sqlite:///path/to/optuna.db`  |

The global `optuna:` block in `benchmarking.yaml` applies to all experiments.
A per-experiment `optuna:` block in `experiments.yaml` overrides it.

### 4 — Output structure

```
experiments/<campaign>/<exp_name>/
  config.yaml          ← best trial's resolved config
  metrics.json         ← best trial's metrics
  log.txt
  trials/
    trial_0000/        ← per-trial artefacts (config, metrics, log)
    trial_0001/
    ...
  optuna/
    trials.csv         ← all trials: params + objective value
    best_params.yaml   ← params of the best trial
    study.db           ← SQLite storage (only if storage: is set)
```

In normal (`opt: false`) runs the `trials/` and `optuna/` directories are not
created; output is identical to the pre-Optuna behaviour.
