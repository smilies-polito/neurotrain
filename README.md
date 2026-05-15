# NeuroTrain

**An open benchmarking framework for SNN training algorithms.**


![NeuroTrain framework overview — trainers, networks, and datasets are independently registered components wired at runtime; run_exp_campaign.py enumerates all valid combinations and aggregates results into per-dataset accuracy tables.](docs/figures/snn_framework.png)

NeuroTrain is an open framework for implementing, comparing, and benchmarking SNN training algorithms under shared, controlled conditions. Trainers, networks, and datasets are fully decoupled — any combination can be benchmarked from a single config file. Hyperparameter optimization via [Optuna](https://optuna.org) is built-in. This makes it possible to separate algorithmic contributions from experimental choices — something that is typically hard when algorithms live in heterogeneous, one-off codebases.

NeuroTrain accompanies a survey paper providing a comprehensive taxonomy of SNN training algorithms — see the [paper](#citation) for background and the full algorithmic landscape.

Built on **[snnTorch](https://github.com/jeshraghian/snntorch)** (≥ 0.7), **[Tonic](https://github.com/neuromorphs/tonic)** (≥ 1.0), and **[NeuroBench](https://github.com/NeuroBench/neurobench)**.


> <a name="citation"></a>**If you use NeuroTrain in your research, please cite:**
>
> Caviglia, A., Marostica, F., Bardini, R., Savino, A., & Di Carlo, S. (2026). *NeuroTrain: Surveying Local Learning Rules for Spiking Neural Networks with an Open Benchmarking Framework*. arXiv:2605.15058. https://arxiv.org/abs/2605.15058
>
> ```bibtex
> @misc{caviglia2026neurotrainsurveyinglocallearning,
>   title   = {NeuroTrain: Surveying Local Learning Rules for Spiking Neural Networks with an Open Benchmarking Framework},
>   author  = {Alessio Caviglia and Filippo Marostica and Roberta Bardini and Alessandro Savino and Stefano Di Carlo},
>   year    = {2026},
>   eprint  = {2605.15058},
>   archivePrefix = {arXiv},
>   primaryClass  = {cs.NE},
>   url     = {https://arxiv.org/abs/2605.15058},
> }
> ```


<p align="center">
  <img src="docs/figures/LOGO_WEB.png" alt="SMILIES logo" height="40"/>
  <br/>
  Developed at the <strong>SMILIES Research Group</strong>, Dept. of Control and Computer Engineering, Politecnico di Torino.
  <br/>
  <a href="https://www.smilies.polito.it/">Website</a> · <a href="https://www.linkedin.com/company/smilies-polito">LinkedIn</a> · <a href="https://x.com/smiliespolito">𝕏</a>
</p>


## Quickstart

```bash
git clone https://github.com/smilies-polito/neurotrain
cd neurotrain
pip install -r requirements.txt

# Single-trainer benchmark (BPTT × all compatible models × datasets)
python3 run_exp_campaign.py --benchmarking config/benchmarking/bptt.yaml --name my_bench

# Custom experiment with optional Optuna HPO
python3 run_exp_campaign.py --custom config/paper_examples.yaml --name my_exp

# Reproduce paper results (all trainers, Optuna HPO, 10 trials × 20 epochs)
python3 run_exp_campaign.py --benchmarking config/paper_benchmarking.yaml --name paper
```

---
## Contents

- [Quickstart](#quickstart)
- [Benchmark Your Algorithm](#benchmark-your-snn-training-algorithm)
- [Run Custom Experiments](#run-custom-experiments)
- [How NeuroTrain Works](#design-principles)

<details>
<summary>Advanced / Development</summary>

- [Configuration System](#configuration-system)
- [NeuroBench Evaluation](#neurobench-evaluation)
- [Output Structure and Results Generation](#output-structure-and-results-generation)
- [Visualising HPO Results](#visualising-hpo-results)
- [Singularity / Apptainer](#singularity--apptainer)
- [Dependencies](#dependencies)
- [Implemented Algorithms](#implemented-algorithms)

</details>

---

# Benchmark Your SNN Training Algorithm

> NeuroTrain is designed for researchers who have developed a new SNN training algorithm and want to benchmark it systematically under fair, controlled conditions. For a complete step-by-step guide see **[`docs/HOW_TO_BENCHMARK_YOUR_TRAINER.md`](docs/HOW_TO_BENCHMARK_YOUR_TRAINER.md)**.

The typical workflow proceeds in four steps:

**Step 1 — Implement your trainer.**
Create `src/trainers/<name>_trainer.py` extending `BaseTrainer`, implement `train_sample()` and `reset()`, then register it in `src/trainers/__init__.py`:

```python
from trainers.my_trainer import MyTrainer
TRAINER_REGISTRY["my_trainer"] = MyTrainer
```

**Step 2 — Add default config and compatibility.**
Create `config/default/trainers/my_trainer.yaml` with default hyperparameters and the `supported_net_types` list. Compatibility with models and datasets is resolved automatically from these YAML fields — no separate registration needed.

**Step 3 — Run HPO.**
Add your experiment to `config/paper_examples.yaml` (or a new custom YAML) with `opt: true` and tunable parameter blocks. Run:

```bash
python3 run_exp_campaign.py --custom config/paper_examples.yaml --name my_trainer_hpo
```

Optuna runs a study for each experiment and writes the best config to `experiments/<name>/<exp>/optuna/best_params.yaml`.

**Step 4 — Run your benchmark and generate results.**

```bash
# Option A — your trainer only, against all compatible combinations
python3 run_exp_campaign.py --benchmarking config/benchmarking/my_trainer.yaml --name my_bench

# Option B — rerun the full matrix including your trainer
python3 run_exp_campaign.py --benchmarking config/paper_benchmarking.yaml --name full_bench

# Generate tables and heatmap from the campaign output
python3 src/generate_results.py experiments/my_bench/
```

---

# Run Custom Experiments

NeuroTrain supports a wide range of customisation — from selecting a specific subset of algorithms for a campaign, to overriding individual hyperparameters, to adding entirely new trainers, models, and datasets. All combinations are supported.

For a complete guide see **[`docs/HOW_TO_RUN_CUSTOM_EXPERIMENTS.md`](docs/HOW_TO_RUN_CUSTOM_EXPERIMENTS.md)**. A quick overview:

- **Custom benchmarking campaign** — create your own benchmarking YAML selecting any subset of trainers, models, and datasets:

```bash
python3 run_exp_campaign.py --benchmarking config/custom/my_campaign.yaml --name my_bench
```

- **Custom parameters** — override any default hyperparameter (trainer, model, or dataset) in `config/paper_examples.yaml` without touching the default configs:

```bash
python3 run_exp_campaign.py --custom config/paper_examples.yaml --name my_exp
```

- **New components** — add your own trainer ([guide](docs/HOW_TO_BENCHMARK_YOUR_TRAINER.md)), model, or dataset and use them in any campaign immediately after registration.

- **HPO on any of the above** — add `opt: true` and tunable blocks to any experiment in either mode.

---

## Design Principles

NeuroTrain separates three orthogonal concerns — *how to train*, *what to train*, and *what data to use* — and wires them together at runtime from a YAML config.

```
Input YAML (benchmarking or custom)
    │
    ▼
campaign_builder  ──→  list[ExperimentSpec]
                               │
           run_exp_campaign.py spawns experiment.py per spec
                               │
                               ▼
                    experiment.py
                      ├─ get_loader(dataset)
                      ├─ get_network(model)
                      ├─ TRAINER_REGISTRY[trainer]
                      ├─ train_one_epoch × epochs
                      ├─ evaluate
                      └─ neurobench_eval  (if enabled)
                               │
                               ▼
             experiments/<campaign>/
               summary.json   summary.csv
               <exp_name>/  config.yaml  metrics.json  log.txt
```

Every trainer implements `train_sample()` and `reset()` from `BaseTrainer`. Every network implements `forward()` and `reset()` from `BaseSNN`. Compatibility between trainers, models, and datasets is declared via `supported_net_types` in each component's default YAML and resolved automatically by `src/campaign/compatibility.py`.

## Repository Structure

```
neurotrain/
│
├── run_exp_campaign.py            # Main entry point (benchmarking + custom modes)
├── experiment.py                  # Single-experiment runner (called per spec)
│
├── config/
│   ├── paper_benchmarking.yaml    # Full benchmark matrix (all trainers, paper settings, HPO)
│   ├── paper_examples.yaml        # Custom mode: named example experiments with overrides + HPO
│   ├── default/
│   │   ├── trainers/              # One YAML per trainer (defaults + tunable blocks)
│   │   ├── models/                # One YAML per network (with per-dataset sections)
│   │   └── datasets/              # One YAML per dataset (timesteps, batch size, etc.)
│   ├── benchmarking/              # Per-trainer benchmarking configs
│   ├── custom/                    # User-created experiment files (not tracked by git)
│   └── vgg9/                      # VGG9-specific experiment configs
│
├── src/
│   ├── trainers/                  # Learning algorithm implementations
│   │   ├── base_trainer.py        # Abstract interface: train_sample() + reset()
│   │   ├── __init__.py            # TRAINER_REGISTRY — maps name → class
│   │   └── …                      # One file per algorithm
│   ├── networks/                  # SNN architectures (fc_snn, r_snn, conv_snn, vgg9*)
│   ├── datasets/                  # Dataset loaders
│   ├── campaign/                  # Orchestration layer
│   │   ├── campaign_builder.py    # Builds ExperimentSpec list from YAML
│   │   ├── compatibility.py       # Trainer × model × dataset compatibility
│   │   ├── config_loader.py       # YAML loading and deep-merge logic
│   │   ├── training_loop.py       # train_one_epoch, evaluate
│   │   ├── neurobench_eval.py     # NeuroBench integration
│   │   ├── optuna_helpers.py      # Tunable block resolution
│   │   └── results.py             # Output writing (summary.csv, summary.json)
│   └── generate_results.py        # Generate Markdown tables + heatmap from summary.csv
│
├── docs/
│   ├── HOW_TO_BENCHMARK_YOUR_TRAINER.md
│   ├── HOW_TO_RUN_CUSTOM_EXPERIMENTS.md
│   └── figures/                   # framework.png, results_heatmap.png, LOGO_WEB.png
└── pyproject.toml
```

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
python run_exp_campaign.py --benchmarking config/benchmarking/my_bench.yaml --name my_bench
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
python run_exp_campaign.py --custom config/paper_examples.yaml --name my_exp
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
  <exp_name>/
    config.yaml                ← resolved config for this run
    metrics.json               ← per-epoch metrics + NeuroBench results
    log.txt
    optuna/                    ← only when opt: true
      trials.csv
      best_params.yaml
      study.db                 ← SQLite, open with optuna-dashboard
```


### Visualising HPO Results

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

The dashboard is served at:

```text
http://localhost:8080
```

Using an absolute path avoids SQLite path-resolution issues, especially inside Singularity or cluster shells.

![optuna-dashboard — trial history, hyperparameter importance, and parallel coordinate plots for a NeuroTrain HPO study](docs/figures/optuna-dashboard-screenshot.png)
*optuna-dashboard for a trainer × model × dataset combination.*

## Singularity / Apptainer

For reproducible execution on HPC clusters, NeuroTrain can be containerised with Singularity (Apptainer).

**1. Build:**

```bash
sudo singularity build neurotrain.sif src/snn-training-benchmarking.def
# without root:
singularity build --fakeroot neurotrain.sif src/snn-training-benchmarking.def
```

**2. Run:**

```bash
singularity exec --nv \
    --bind /path/to/neurotrain:/workspace \
    neurotrain.sif \
    bash -c "cd /workspace && python run_exp_campaign.py \
        --benchmarking config/benchmarking/bptt.yaml --name my_bench"
```

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

---

## Implemented Algorithms

| Algorithm | Key | Networks |
|---|---|---|
| Backpropagation Through Time | `bptt` | FC · RC · Conv |
| Deep Continuous Local Learning | `decolle` | FC · Conv |
| Eligibility Propagation | `eprop` | RC |
| Event-Driven Symmetric RTRL | `esd_rtrl` | FC · RC · Conv |
| Event-based Three-factor Local Plasticity | `etlp` | FC · RC |
| Online Spatio-Temporal Learning | `ostl` | FC · RC |
| OSTL with Target Projection | `osttp` | FC · RC |
| Online Training Through Time | `ottt` | FC · RC · Conv · VGG9 |
| Spiking Time Sparse Feedback | `stsf` | FC |
| Trace Propagation | `tp` | FC · RC · Conv · VGG9 |

Each algorithm is registered in `src/trainers/__init__.py` via `TRAINER_REGISTRY`. Network compatibility is declared in `config/default/trainers/<name>.yaml` via the `supported_net_types` field and resolved automatically — no manual wiring needed.

---

# Current Results

In this section there are the results of a complete campaign on the components of the framework to showcase its capabilities. All results are obtained with an HPO with Optuna on the default hyperparameter search space defined in `config/default/` (see the paper for details on the search space). Each experiments has been run for 10 trials per 20 epochs and the best final test accuracy for each experiment is reported in the tables below. For some experiments, due to GPU memory constraints, the batch size has been reduced from the default 256.

## Legend

| Symbol | Meaning |
| :----: | ------- |
| 🟢 | Experiment successful |
| 🟡 | Results but with problems |
| 🔴 | Error while running |
| ⚫ | Not supported |

> **Dataset groups** — Frame-based: `MNIST` `F-MNIST` `CIFAR10` `SVHN`
> · Neuromorphic: `NMNIST` `DVSGest.` `DVSCifar10` `SHD`
>
> **Network abbreviations** — `FC` = Fully Connected · `RC` = Recurrent · `Conv` = Convolutional

**Default network architectures** (defined in [`config/default/models/`](config/default/models/)):

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC | 784-256-10 | 784-800-10 | 3072-1024-512-10 | 3072-1024-512-10 | 2312-512-10 | 32768-2048-11 | 32768-1024-512-10 | 700-512-20 |
| RC | 784-256-10 | 784-256-10 | 3072-512-256-10 | 3072-512-256-10 | 2312-256-10 | 32768-1024-11 | 32768-512-10 | 700-512-20 |

Conv is always: 12C5-MP2-32C5-MP2-FC

---

## BPTT

All results obtained with the default campaign configuration.

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC | 0.978 🟢 | 0.837 🟢 | 0.359 🟢 | 0.536 🟢 | 0.968 🟢 | 0.689 🟢 | 0.337 🟢 | 0.496 🟢 |
| RC | 0.969 🟢 | 0.828 🟢 | 0.343 🟢 | 0.561 🟢 | 0.958 🟢 | 0.712 🟢 | 0.340 🟢 | 0.696 🟢 |
| Conv | 0.989 🟢 | 0.829 🟢 | 0.449 🟢 | 0.838 🟢 | 0.982 🟢 | 0.636 🟢 | 0.379 🟢 | ⚫ |

---

## DECOLLE

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC | 0.957 🟢 | 0.801 🟢 | 0.399 🟢 | 0.749 🟢 | 0.931 🟢 | 0.708 🟢 | 0.361 🟢 | 0.375 🟢 |
| RC | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |
| Conv | 0.971 🟢 | 0.783 🟢 | 0.352 🟢 | 0.559 🟢 | 0.955 🟢 | 0.784 🟢 [^d1] | 0.394 🟢 [^d1] | ⚫ |

[^d1]: Results obtained with a lower batch size of 64 due to GPU memory constraints.

---

## EPROP

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |
| RC | 0.9783 🟢 | 0.8550 🟢 | 0.425 🟢 [^e1] | 0.596 🟢 [^e1] | 0.960 🟢 [^e1] | 0.667 🟢 [^e2] | 0.253 🟢 [^e2] | 0.6913 🟢 |
| Conv | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |

[^e1]: Results obtained with 512 hidden units instead of the standard 256 for a fair comparison, run in a separate campaign.
[^e2]: Results obtained with a lower batch size of 32 due to GPU memory constraints.

---

## ESD_RTRL

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC | 0.956 🟢 | 0.864 🟢 | 0.426 🟢 | 0.732 🟢 | 0.958 🟢 | 0.731 🟢 | 0.379 🟢 | 0.510 🟢 |
| RC | 0.827 🟢 | 0.678 🟢 | 0.311 🟢 | 0.253 🟢 | 0.801* 🟢 | 0.708 🟢 [^r1] | 0.191 🟢 [^r1] | 0.451 🟢 |
| Conv | 0.963 🟢 | 0.808 🟢 | 0.569 🟢 | 0.661 🟢 | 0.949 🟢 | 0.474 🟢 | 0.233 🟢 | ⚫ |

[^r1]: Batch size of 32.


---

## ETLP

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC | 0.932 🟢 | 0.822 🟢 | 0.249 🟢 | 0.361 🟢 | 0.888 🟢 | 0.636 🟢 [^t1] | 0.264 🟢 [^t1] | 0.260 🟢 |
| RC | 0.913 🟢 | 0.807 🟢 | 0.261 🟢 | 0.308 🟢 | 0.901 🟢 | 0.689 🟢 [^t1] | 0.308 🟢 | 0.269 🟢 |
| Conv | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |

[^t1]: Results obtained with a lower batch size of 64 due to GPU memory constraints.

---

## OSTL

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC | 0.963 🟢 | 0.837 🟢 | 0.379 🟢 | 0.618 🟢 | 0.932 🟢 | 0.712 🟢 [^o1] | 0.324 🟢 [^o1] | 0.236 🟢 |
| RC | 0.965 🟢 | 0.832 🟢 | 0.237 🟢 | 0.279 🟢 | 0.941 🟢 | 0.712 🟢 [^o1] | 0.323 🟢 [^o1] | 0.308 🟢 |
| Conv | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |

[^o1]: Results obtained with a lower batch size of 32 due to GPU memory constraints.

---

## OSTTP

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC | 0.925 🟢 | 0.823 🟢 | 0.315* 🟢 | 0.300 🟢 | 0.910 🟢 | 0.693 🟢 [^p1] | 0.253 🟢 [^p1] | 0.280 🟢 |
| RC | 0.921 🟢 | 0.810 🟢 | 0.215 🟢 | 0.217 🟢 | 0.918 🟢 | 0.655 🟢 [^p1] | 0.100 🟡 | 0.057 🟡 |
| Conv | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |

[^p1]: Results obtained with a lower batch size of 16 due to GPU memory constraints.

---

## OTTT

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC | 0.932 🟢 | 0.807 🟢 | 0.349 🟢 | 0.609 🟢 | 0.870 🟢 | 0.572 🟢 | 0.297 🟢 | 0.264 🟢 |
| RC | 0.930 🟢 | 0.810 🟢 | 0.350 🟢 | 0.583 🟢 | 0.882 🟢 | 0.606 🟢 | 0.357 🟢 | 0.412 🟢 |
| Conv | 0.954 🟢 | 0.738 🟢 | 0.492 🟢 | 0.795 🟢 | 0.802 🟢 | 0.576 🟢 | 0.357 🟢 | ⚫ |

---

## STSF

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC | 0.933 🟢 | 0.800 🟢 | 0.277 🟢 | 0.276 🟢 | 0.904 🟢 | 0.708 🟢 | 0.199 🟢 | 0.221 🟢 |
| RC | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |
| Conv | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ | ⚫ |

---

## TP

| Network | MNIST | F-MNIST | CIFAR10 | SVHN | NMNIST | DVSGest. | DVSCifar10 | SHD |
| ------- | :---: | :-----: | :-----: | :--: | :----: | :------: | :--------: | :-: |
| FC | 0.975 🟢 | 0.862 🟢 | 0.339 🟢 | 0.457 🟢 | 0.963 🟢 | 0.686 🟢 | 0.307 🟢 | 0.498 🟢 |
| RC | 0.974 🟢 | 0.859 🟢 | 0.348 🟢 | 0.562 🟢 | 0.962 🟢 | 0.701 🟢 | 0.338 🟢 | 0.576 🟢 |
| Conv | 0.982 🟢 | 0.846 🟢 | 0.547 🟢 | 0.785 🟢 | 0.972 🟢 | 0.629 🟢 | 0.333 🟢 | ⚫ |

---

## VGG9 Networks

The framework also supports VGG9 architectures. They are tested separately from the standard experiments given the increased training time and complexity. Results below cover the trainers that support VGG9 (BPTT, OTTT, TP) on the more complex datasets (CIFAR10, SVHN, DVSCifar10, DVSGesture), using two VGG9 variants. VGG9 Optuna HPO configs are stored in [`config/vgg9/`](config/vgg9/) following the naming convention `{trainer}_vgg9{v1|v2}_{dataset}.yaml`.

### VGG9 Architecture Variants

Trainers that support VGG9 are benchmarked with two variants. The results are on 100 epochs of training with a minimal exploration. Results will be updated as we explore more hyperparameters and run more epochs.
Both variants share the same base convolutional structure with 8 layers (64, 128, 256, 256, 512, 512, 512, 512 channels).

**VGG9v1**
- **Head**: Global linear classifier (no temporal integration)
- **Pooling**: Average pooling 2×2 after blocks 2 and 4
- **Fixed params**: Sigmoid surrogate function, Conv gain = 1.0, scaling after LIF

**VGG9v2**
- **Head**: Leaky integrator with temporal pooling integration (2×2 spatial, leak 1.0)
- **Pooling**: MaxPool 2×2 after blocks 2, 4, 6; AdaptiveAvgPool 2×2 after block 8
- **Fixed params**: Arctangent surrogate function, Conv gain = 1.8, no scaling after LIF

### VGG9 Results

| Trainer | Network | CIFAR10 | SVHN | DVSCifar10 | DVSGesture |
| ------- | ------- | :-----: | :--: | :--------: | :--------: |
| BPTT | VGG9v1 | 0.100 🟡  | 0.067 🟡  | 0.085 🟡  | 0.091 🟡  |
| BPTT | VGG9v2 | 0.910 🟢 | 0.960 🟢 | 0.626 🟡  | 0.894 🟢 |
| OTTT | VGG9v1 | 0.525 🟡  | 0.485 🟡 | 0.100 🟡 | 0.091 🟡 |
| OTTT | VGG9v2 | 0.666 🟡  | 0.224 🟡  | 0.587 🟡 | 0.091 🟡 |
| TP | VGG9v1 | 0.534 🟡  | 0.311 🟡 | 0.375 🟡 | 0.920 🟢 |
| TP | VGG9v2 | 0.750 🟡  | 0.944 🟢 | 0.311 🟡 | 0.882 🟢 |

### VGG9 HPO Config Files

All HPO configs use Optuna (50 trials, TPE sampler). Run any single config with:

```bash
python run_exp_campaign.py --custom config/vgg9/<config>.yaml --name vgg9_run
```

| Trainer | Network | CIFAR10 | SVHN | DVSCifar10 | DVSGesture |
| ------- | ------- | ------- | ---- | ---------- | ---------- |
| BPTT | VGG9v1 | `bptt_vgg9v1_cifar10.yaml` | `bptt_vgg9v1_svhn.yaml` | `bptt_vgg9v1_dvscifar10.yaml` | `bptt_vgg9v1_dvsgesture.yaml` |
| BPTT | VGG9v2 | `bptt_vgg9v2_cifar10.yaml` | `bptt_vgg9v2_svhn.yaml` | `bptt_vgg9v2_dvscifar10.yaml` | `bptt_vgg9v2_dvsgesture.yaml` |
| OTTT | VGG9v1 | `ottt_vgg9v1_cifar10.yaml` | `ottt_vgg9v1_svhn.yaml` | `ottt_vgg9v1_dvscifar10.yaml` | `ottt_vgg9v1_dvsgesture.yaml` |
| OTTT | VGG9v2 | `ottt_vgg9v2_cifar10.yaml` | `ottt_vgg9v2_svhn.yaml` | `ottt_vgg9v2_dvscifar10.yaml` | `ottt_vgg9v2_dvsgesture.yaml` |
| TP | VGG9v1 | `tp_vgg9v1_cifar10.yaml` | `tp_vgg9v1_svhn.yaml` | `tp_vgg9v1_dvscifar10.yaml` | `tp_vgg9v1_dvsgesture.yaml` |
| TP | VGG9v2 | `tp_vgg9v2_cifar10.yaml` | `tp_vgg9v2_svhn.yaml` | `tp_vgg9v2_dvscifar10.yaml` | `tp_vgg9v2_dvsgesture.yaml` |

---

# License

*To be added.*