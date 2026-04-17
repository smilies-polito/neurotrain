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

## Optuna-tunable attributes

Any scalar attribute can be replaced with a tunable block:

```yaml
lr:
  value: 1.0e-3    # used in normal runs and as the Optuna starting point
  type: float      # float | int | categorical
  min: 1.0e-5
  max: 1.0e-1
  log: true        # optional: log-scale sampling

batch_size:
  value: 256
  type: int
  min: 32
  max: 512

loss_type:
  value: ce_rate
  type: categorical
  list: [ce_rate, mse_count, ce_count]
```

Set `opt: true` in the custom experiment config to activate Optuna.
In normal runs (`opt: false`) the plain `value` is used.
