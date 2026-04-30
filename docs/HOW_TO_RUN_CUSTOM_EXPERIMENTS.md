# How to Run Custom Experiments with NeuroTrain

This guide covers all the ways you can customise what NeuroTrain runs and how. "Custom" can mean several different things:

- Select a specific subset of algorithms, architectures, or datasets for a benchmarking campaign
- Override hyperparameters of existing components without touching their default configs
- Add entirely new components (trainers, models, datasets) that you have implemented yourself
- Combine any of the above

All of these are supported through two operating modes launched via `run_exp_campaign.py`:

| Mode | Entry point | Use when |
|---|---|---|
| **Benchmarking** | `--benchmarking <yaml>` | You want a systematic matrix of combinations |
| **Custom** | `--custom <yaml>` | You want named, individually-configured experiments |

Both modes can launch Optuna HPO runs, but custom mode gives you finer control over the search space for individual experiments. See [Hyperparameter Optimisation](#hyperparameter-optimisation-optional) for details.

Benchmarking mode is designed for matrix generation: it selects combinations and applies shared runtime/HPO settings. To override trainer, model, or dataset parameters for individual experiments, use custom mode.

---

## Configuration Override Logic

Before diving into specific use cases, it helps to understand how NeuroTrain merges configurations. You only ever need to specify what changes — everything else is inherited from the component defaults in `config/default/`.

### Priority order

Configuration resolution depends on the component type.

For **trainers** and **datasets**, the merge order is simple:

| Priority | Source |
|---|---|
| 1 — highest | Your overrides in the custom experiment YAML |
| 2 — lowest | Component default in `config/default/` |

For **models**, NeuroTrain first resolves the model recipe and then applies flat user overrides:

| Priority | Source |
|---|---|
| 1 — highest | Flat model overrides in your custom experiment YAML |
| 2 | Trainer-specific section in the model YAML, e.g. `tp:` or `ottt:` |
| 3 | Dataset-specific section in the model YAML, e.g. `mnist:` or `cifar10:` |
| 4 — lowest | `default:` section in the model YAML |

### Model dataset specialisation

Model YAMLs have a `default:` section plus optional dataset-specific sections. The dataset section is merged on top — only specify the keys that differ:

```yaml
# config/default/models/fc_snn.yaml
default:
  beta: 0.9
  hidden_sizes:
    value: [256]

mnist:
  hidden_sizes:
    value: [128]    # overrides default for MNIST only

fashionmnist:
  hidden_sizes:
    value: [800]    # overrides default for FashionMNIST only
```

A flat experiment override has the highest priority and is the recommended way to force a value regardless of dataset or trainer.

---

## Contents

1. [Custom benchmarking campaigns](#1-custom-benchmarking-campaigns)
2. [Custom parameters on existing components](#2-custom-parameters-on-existing-components)
3. [Adding new components](#3-adding-new-components)
4. [Combining everything](#4-combining-everything)
5. [Hyperparameter optimisation (optional)](#hyperparameter-optimisation-optional)
6. [Inspecting results](#inspecting-results)

---

## 1. Custom Benchmarking Campaigns

Benchmarking mode reads a YAML file and runs all valid combinations of the listed trainers, models, and datasets. Create your own file under `config/` (or `config/custom/`) to define any subset you want.

```yaml
# config/custom/my_campaign.yaml

trainers: [bptt, ostl, etlp]        # subset of available trainers
models:   [fc_snn, r_snn]           # subset of available models
datasets: [MNIST, NMNIST]           # subset of available datasets

runtime:
  epochs: 100
  device: cuda
  seed: 42
  neurobench: false

opt: false
```

An **empty list** means "all available defaults":

```yaml
trainers: []    # runs all trainers in config/default/trainers/
models:   []    # runs all models   in config/default/models/
datasets: []    # runs all datasets in config/default/datasets/
```

Run it:

```bash
python3 run_exp_campaign.py --benchmarking config/custom/my_campaign.yaml \
    --name my_campaign

# Dry-run: print the list of experiments that would be generated, without running
python3 run_exp_campaign.py --benchmarking config/custom/my_campaign.yaml --dry-run
```

Incompatible combinations (e.g. a trainer that does not support convolutional networks) are skipped automatically based on `supported_net_types` declared in each component's default config.

### Shared runtime overrides

Any key in the `runtime:` block applies to all experiments in the campaign. You can override epochs, device, seed, and NeuroBench evaluation globally here without touching per-component defaults. Set `runtime.neurobench: true` when you want to include NeuroBench efficiency metrics.

```yaml
runtime:
  epochs: 50
  device: cuda
  seed: 123
  neurobench: true    # run NeuroBench evaluation for every combination
```

---

## 2. Custom Parameters on Existing Components

Use **custom mode** to run named experiments where any default parameter is overridden with a value of your choice. You only need to specify the keys that change — everything else inherits from the component's default YAML in `config/default/`.

### Overriding trainer hyperparameters

```yaml
# config/custom/my_experiments.yaml

bptt_low_lr:
  trainer:
    name: bptt
    lr: 1.0e-5          # override: lower learning rate
    loss_type: mse_count # override: different loss
  model:
    name: fc_snn
  dataset:
    name: MNIST
  runtime:
    epochs: 100
    device: cuda
    seed: 42
```

### Overriding model architecture

```yaml
bptt_deep_fc:
  trainer:
    name: bptt
  model:
    name: fc_snn
    hidden_sizes: [512, 256, 128]  # override: deeper network
    beta: 0.95                     # override: slower leak
    threshold: 0.8                 # override: lower threshold
  dataset:
    name: MNIST
  runtime:
    epochs: 100
    device: cuda
```

### Overriding dataset settings

```yaml
bptt_long_sequences:
  trainer:
    name: bptt
  model:
    name: r_snn
  dataset:
    name: NMNIST
    T: 50              # override: longer time window (default is 10)
    num_workers: 8     # override: more dataloader workers
    direct_coding: true  # override: use direct/analog coding instead of rate
  runtime:
    epochs: 50
    device: cuda
```

### Multiple experiments in one file

A single `experiments.yaml` file can contain as many named experiments as needed:

```yaml
# config/custom/ablation.yaml

bptt_small:
  trainer: { name: bptt, lr: 1e-3 }
  model:   { name: fc_snn, hidden_sizes: [128] }
  dataset: { name: MNIST }
  runtime: { epochs: 50, device: cuda }

bptt_medium:
  trainer: { name: bptt, lr: 1e-3 }
  model:   { name: fc_snn, hidden_sizes: [256] }
  dataset: { name: MNIST }
  runtime: { epochs: 50, device: cuda }

bptt_large:
  trainer: { name: bptt, lr: 1e-3 }
  model:   { name: fc_snn, hidden_sizes: [512, 256] }
  dataset: { name: MNIST }
  runtime: { epochs: 50, device: cuda }
```

```bash
python3 run_exp_campaign.py --custom config/custom/ablation.yaml --name ablation_fc_size
```

---

## 3. Adding New Components

You are not limited to the components that ship with NeuroTrain. You can add your own trainers, models, and datasets and use them in any campaign or experiment.

### New trainer

See [`docs/HOW_TO_BENCHMARK_YOUR_TRAINER.md`](HOW_TO_BENCHMARK_YOUR_TRAINER.md) for the full guide. In short:

1. Create `src/trainers/my_trainer.py` extending `BaseTrainer`
2. Register it in `src/trainers/__init__.py`
3. Create `config/default/trainers/my_trainer.yaml` with defaults and `supported_net_types`

Once registered, `my_trainer` is available in both benchmarking and custom mode exactly like any built-in trainer.

### New model

1. Create `src/networks/my_model.py` extending `BaseSNN` — implement `forward()` and `reset()`
2. Register it in the network factory (`src/networks/__init__.py` or equivalent)
3. Create `config/default/models/my_model.yaml`:

```yaml
# config/default/models/my_model.yaml

default:
  name: my_model
  net_type: fc          # fc | rec | conv — used for compatibility resolution

  # architecture parameters
  hidden_sizes:
    value: [256]
    type: null
  beta:
    value: 0.9
    type: float
    min: 0.8
    max: 0.99

mnist:
  in_shape: [784]
  num_classes: 10

cifar10:
  in_shape: [3072]
  num_classes: 10
  hidden_sizes:
    value: [1024, 512]
```

The `default:` section provides universal defaults. Dataset-specific sections (e.g. `mnist:`, `cifar10:`) are merged on top when the model is paired with that dataset — only specify the keys that differ.

### New dataset

1. Create `src/datasets/my_dataset_loader.py` implementing the standard loader interface (returns a `(train_loader, test_loader)` tuple)
2. Register it in the dataset factory
3. Create `config/default/datasets/my_dataset.yaml`:

```yaml
# config/default/datasets/my_dataset.yaml

name: my_dataset
supported_net_types: [fc, conv]   # which net_types this dataset works with

T:
  value: 20
  type: int
  min: 10
  max: 50
  step: 5

num_workers: 4
data_root: null       # define the fallback behaviour in your dataset loader
download: true
```

Once registered, `my_dataset` is available in both modes:

```yaml
# in benchmarking.yaml
datasets: [MNIST, my_dataset]

# in experiments.yaml
my_exp:
  dataset:
    name: my_dataset
    T: 30
```

---

## 4. Combining Everything

You can freely combine custom campaigns, parameter overrides, and new components. A few realistic examples:

### Custom campaign with parameter overrides

```yaml
# config/custom/local_rules_on_my_model.yaml  (benchmarking mode)

trainers: [ostl, etlp, stsf, my_trainer]
models:   [my_model]
datasets: [MNIST, NMNIST]

runtime:
  epochs: 75
  device: cuda
  seed: 42
  neurobench: false
```

```bash
python3 run_exp_campaign.py \
    --benchmarking config/custom/local_rules_on_my_model.yaml \
    --name local_rules_my_model
```

### Named experiments mixing built-in and custom components

```yaml
# config/custom/cross_comparison.yaml  (custom mode)

# Built-in trainer on your custom model
bptt_my_model_mnist:
  trainer:
    name: bptt
    lr: 2.0e-3
  model:
    name: my_model
    hidden_sizes: [512, 256]
  dataset:
    name: MNIST
  runtime:
    epochs: 100
    device: cuda

# Your custom trainer on a built-in model
my_trainer_fc_snn_nmnist:
  trainer:
    name: my_trainer
    my_param: 0.85
  model:
    name: fc_snn
  dataset:
    name: NMNIST
    T: 30
  runtime:
    epochs: 100
    device: cuda

# Your custom trainer on your custom dataset
my_trainer_my_dataset:
  trainer:
    name: my_trainer
  model:
    name: fc_snn
  dataset:
    name: my_dataset
    T: 20
  runtime:
    epochs: 50
    device: cuda
```

```bash
python3 run_exp_campaign.py --custom config/custom/cross_comparison.yaml \
    --name cross_comparison
```

---

## Hyperparameter Optimisation (Optional)

HPO via Optuna works identically in all the scenarios above. Add `opt: true` and replace any scalar with a tunable block.

### In benchmarking mode — tune all combinations

```yaml
# config/custom/my_campaign.yaml

trainers: [ostl, etlp]
models:   [fc_snn]
datasets: [MNIST]

opt: true          # enables Optuna for every generated experiment
optuna:
  n_trials: 30
  sampler: tpe
```

### In custom mode — tune selected experiments

```yaml
bptt_tuned:
  opt: true
  optuna:
    n_trials: 50
  trainer:
    name: bptt
    lr:
      value: 1.0e-3
      type: float
      min: 1.0e-5
      max: 1.0e-1
      log: true
  model:
    name: fc_snn
    beta:
      value: 0.9
      type: float
      min: 0.8
      max: 0.99
  dataset:
    name: MNIST
  runtime:
    epochs: 20    # short runs during HPO; increase in paper.yaml for final run

bptt_fixed:
  opt: false       # this experiment runs with fixed values only
  trainer:
    name: bptt
    lr: 5.0e-4
  model:
    name: fc_snn
  dataset:
    name: MNIST
  runtime:
    epochs: 100
```

### Tunable block syntax reference

```yaml
# Float (log-scale)
lr:
  value: 1.0e-3    # used when opt: false
  type: float
  min: 1.0e-5
  max: 1.0e-1
  log: true

# Integer with step
T:
  value: 25
  type: int
  min: 10
  max: 50
  step: 5

# Categorical
loss_type:
  value: ce_rate
  type: categorical
  list: [ce_rate, mse_count, ce_count]

# Fixed (not tunable — documents intent without enabling search)
hidden_sizes:
  value: [256]
  type: null
```

When `opt: false`, all blocks are automatically flattened to their `value`.

---

## Inspecting Results

```bash
python3 run_exp_campaign.py --benchmarking config/custom/my_campaign.yaml \
    --name my_campaign
```

Outputs:

```
experiments/my_campaign/
  campaign.yaml              # copy of the input config
  summary.json               # all experiments, one dict per run
  summary.csv                # flat table — trainer, model, dataset,
                             # test_accuracy, train_loss, elapsed_s, nb_* columns
  experiments/
    <exp_name>/
      config.yaml            # resolved config for this run
      metrics.json           # test_accuracy, train_loss, neurobench{}
      log.txt
      optuna/                # only when opt: true
        trials.csv
        best_params.yaml
        study.db
```

### Generate tables and heatmap

```bash
python3 scripts/generate_results.py experiments/my_campaign/

# With NeuroBench metrics
python3 scripts/generate_results.py experiments/my_campaign/ --neurobench

# Inject tables and heatmap into this guide
python3 scripts/generate_results.py experiments/my_campaign/ \
    --readme docs/HOW_TO_RUN_CUSTOM_EXPERIMENTS.md
```

<!-- RESULTS_START -->

*Results from campaign `full_bench`. Generated by `scripts/generate_results.py`.*

![NeuroTrain benchmarking results — test accuracy heatmap, algorithms (rows) × architecture–dataset combinations (cols). Campaign: full_bench — 2026-04-30 22:02](../experiments/full_bench/results_heatmap.png)
*NeuroTrain — Benchmarking Results · Campaign: `full_bench` · 2026-04-30 22:02*

### CIFAR-10

*Test accuracy (mean ± std where multiple seeds available).*

| Algorithm | Conv-SNN | FC-SNN | R-SNN |
|---|---|---|---|
| BPTT | 34.5% | 32.1% | 30.4% |
| DECOLLE | 28.1% | 31.8% | — |

### DVSGesture

*Test accuracy (mean ± std where multiple seeds available).*

| Algorithm | Conv-SNN | FC-SNN | R-SNN |
|---|---|---|---|
| BPTT | 46.6% | 52.7% | 48.9% |
| DECOLLE | 51.5% | 63.6% | — |

### Fashion-MNIST

*Test accuracy (mean ± std where multiple seeds available).*

| Algorithm | Conv-SNN | FC-SNN | R-SNN |
|---|---|---|---|
| BPTT | 75.8% | 79.6% | 79.5% |
| DECOLLE | 65.7% | 65.2% | — |

### MNIST

*Test accuracy (mean ± std where multiple seeds available).*

| Algorithm | Conv-SNN | FC-SNN | R-SNN |
|---|---|---|---|
| BPTT | 97.8% | 95.2% | 94.9% |
| DECOLLE | 80.3% | 83.2% | — |

### N-MNIST

*Test accuracy (mean ± std where multiple seeds available).*

| Algorithm | Conv-SNN | FC-SNN | R-SNN |
|---|---|---|---|
| BPTT | 97.1% | 93.7% | 93.2% |
| DECOLLE | 82.3% | 77.8% | — |

### SVHN

*Test accuracy (mean ± std where multiple seeds available).*

| Algorithm | Conv-SNN | FC-SNN | R-SNN |
|---|---|---|---|
| BPTT | 74.3% | 37.2% | 35.6% |
| DECOLLE | 49.2% | — | — |

<!-- RESULTS_END -->

### Visualise HPO trials

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///experiments/my_campaign/experiments/<exp_name>/optuna/study.db
# → opens at http://localhost:8080
```

The parallel coordinate view is particularly useful for understanding which parameters drive accuracy and where the search converged.