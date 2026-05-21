# Advanced Reference

- [Configuration System](#configuration-system)
- [NeuroBench Evaluation](#neurobench-evaluation)
- [Output Structure and Results Generation](#output-structure-and-results-generation)
- [Visualising HPO Results](#visualising-hpo-results)
- [Singularity / Apptainer](#singularity--apptainer)
- [Dependencies](#dependencies)

---

## Configuration System

NeuroTrain has two operating modes, both launched via `run_exp_campaign.py`. Internally, it spawns `experiment.py` once per resolved `ExperimentSpec` — you can also call `experiment.py <spec.json> <output_dir>` directly to debug a single run.

### Benchmarking mode

Declares which trainers, models, and datasets to compare. The campaign builder generates all valid combinations automatically.

```yaml
# config/benchmarking/my_bench.yaml
trainers: [bptt, ostl]   # empty list = all in config/default/trainers/
models:   [fc_snn]
datasets: [MNIST, FashionMNIST]
runtime:
  epochs: 50
  device: cuda
  seed: 42
  neurobench: false      # set true to run NeuroBench evaluation after training
opt: false               # set true to run Optuna for every combination
optuna:
  n_trials: 50
  sampler: tpe
```

```bash
python3 run_exp_campaign.py --benchmarking config/benchmarking/my_bench.yaml --name my_bench
```

### Custom mode

Defines named experiments with explicit overrides. Supports per-experiment Optuna HPO.

```yaml
# config/paper_examples.yaml
my_experiment:
  trainer:
    name: bptt
    lr: 5e-4
  model:
    name: fc_snn
    hidden_sizes: [128]
  dataset:
    name: MNIST
    T: 25
  runtime:
    epochs: 50
    device: cuda

my_tuned_experiment:
  opt: true
  optuna:
    n_trials: 50
  trainer:
    name: stsf
    lr:
      value: 1e-3
      type: float
      min: 1e-5
      max: 1e-1
      log: true
  model:
    name: fc_snn
    beta:
      value: 0.9
      type: float
      min: 0.5
      max: 0.99
  dataset:
    name: FashionMNIST
  runtime:
    epochs: 20
```

```bash
python3 run_exp_campaign.py --custom config/paper_examples.yaml --name my_exp
```

### Tunable parameter blocks

Any scalar in a trainer, model, or dataset config can be made tunable by replacing it with a block:

```yaml
lr:
  value: 1e-3        # used when opt: false
  type: float        # float | int | categorical
  min: 1e-5
  max: 1e-1
  log: true          # log-scale sampling

loss_type:
  value: ce_rate
  type: categorical
  list: [ce_rate, mse_count, ce_count]
```

### Model Specialization

Model YAMLs support trainer-specific and dataset-specific sections that are merged on top of the `default` section at runtime. This lets you keep a single model file while varying architecture details per trainer or dataset.

```yaml
# config/default/models/vgg9.yaml
default:
  beta: 0.9
  threshold: 1.0
tp:
  head_type: leaky_integrator  # applied when trainer is "tp"
mnist:
  input_shape: [1, 28, 28]     # applied when dataset is "mnist"
```

The merge order, from lowest to highest priority:

| Priority | Source | Example |
|---|---|---|
| 4 (lowest) | `default` block in model YAML | `default: { beta: 0.9 }` |
| 3 | Dataset-specific section in model YAML | `mnist: { input_shape: ... }` |
| 2 | Trainer-specific section in model YAML | `tp: { head_type: ... }` |
| 1 (highest) | User overrides in experiment file | `model: { beta: 0.95 }` |

Trainer-specific sections override dataset sections, so a trainer can enforce algorithm constraints (e.g. head type) regardless of the dataset. User flat overrides in the experiment file always win over all sections.

---

## NeuroBench Evaluation

NeuroTrain integrates [NeuroBench](https://github.com/NeuroBench/neurobench) for standardised neuromorphic evaluation. When `neurobench: true` is set in the `runtime` block, a full NeuroBench benchmark is run on the trained model after each experiment and its results are written to `metrics.json` alongside the standard training metrics.

Enable it in either mode:

```yaml
runtime:
  neurobench: true
```

The following metrics are computed automatically:

**Static metrics** (computed once on the model):

| Metric | Description |
|---|---|
| `Footprint` | Memory footprint of the model weights |
| `ConnectionSparsity` | Fraction of zero-valued synaptic connections |
| `ParameterCount` | Total number of trainable parameters |

**Workload metrics** (computed during inference on the test set):

| Metric | Description |
|---|---|
| `ClassificationAccuracy` | Test accuracy via NeuroBench harness |
| `ActivationSparsity` | Overall spike sparsity across all layers |
| `ActivationSparsityByLayer` | Per-layer spike sparsity breakdown |
| `MembraneUpdates` | Number of membrane potential updates per inference |

All NeuroBench results are stored under a `neurobench` key in `metrics.json` and included in the campaign-level `summary.csv` with `nb_` prefix columns, enabling direct comparison of efficiency metrics across algorithms alongside accuracy.

---

## Output Structure and Results Generation

### Campaign outputs

Each campaign produces per-experiment outputs plus a campaign-level summary:

```
experiments/<campaign>/
  campaign.yaml                ← copy of the input config
  summary.json                 ← all experiments, one dict per run
  summary.csv                  ← flat table: trainer, model, dataset,
                                  test_accuracy, train_loss, elapsed_s,
                                  epochs, nb_* (NeuroBench columns)
  experiments/
    <exp_name>/
      config.yaml              ← resolved config for this run
      metrics.json             ← per-epoch metrics + NeuroBench results
      log.txt
      optuna/                  ← only when opt: true
        trials.csv
        best_params.yaml
        study.db               ← SQLite, open with optuna-dashboard
```

### Generating tables and heatmaps

```bash
python3 src/generate_results.py experiments/my_bench/
```

This produces a Markdown results table and an accuracy heatmap from `summary.csv`.

---

## Visualising HPO Results

When `opt: true` is set, NeuroTrain saves an Optuna SQLite study database for each experiment:

```text
experiments/<campaign>/experiments/<exp_name>/optuna/study.db
```

Use [optuna-dashboard](https://github.com/optuna/optuna-dashboard) to inspect trial history, hyperparameter importances, parallel coordinate plots, and convergence behaviour in the browser.

Launch the dashboard using the **absolute path** to the study database:

```bash
optuna-dashboard "sqlite:////absolute/path/to/study.db"
```

For example:

```bash
optuna-dashboard "sqlite:////home/user/neurotrain/experiments/<campaign>/experiments/<exp_name>/optuna/study.db"
```

The dashboard is served at `http://localhost:8080`. Using an absolute path avoids SQLite path-resolution issues, especially inside Singularity or cluster shells.

![optuna-dashboard — trial history, hyperparameter importance, and parallel coordinate plots for a NeuroTrain HPO study](figures/optuna-dashboard-screenshot.png)

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| [torch](https://pytorch.org) | ≥ 2.0 | Core deep learning |
| [snntorch](https://github.com/jeshraghian/snntorch) | ≥ 0.7 | LIF neuron models, surrogate gradients — core SNN engine |
| [tonic](https://github.com/neuromorphs/tonic) | ≥ 1.0 | Event-based dataset loading (N-MNIST, DVSGesture, SHD, DVS-CIFAR10) |
| [neurobench](https://github.com/NeuroBench/neurobench) | latest | Neuromorphic benchmarking metrics and datasets |
| [optuna](https://optuna.org) | ≥ 3.0 | Built-in hyperparameter optimisation |
| `pandas` | ≥ 2.0 | Results table generation (`src/generate_results.py`) |
| `matplotlib` | ≥ 3.7 | Heatmap generation (`src/generate_results.py`) |
| `pyyaml` | ≥ 6.0 | Config parsing |

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, CUDA optional (MPS and CPU supported via `device: auto`).
