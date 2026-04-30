# NeuroTrain

**An open benchmarking framework for SNN training algorithms.**

NeuroTrain is an open framework for implementing, comparing, and benchmarking SNN training algorithms under shared, controlled conditions. Trainers, networks, and datasets are fully decoupled — any combination can be benchmarked from a single config file. Hyperparameter optimization via [Optuna](https://optuna.org) is built-in: mark any parameter as tunable directly in YAML and set `opt: true` to run a study. This makes it possible to separate algorithmic contributions from experimental choices — something that is typically hard when algorithms live in heterogeneous, one-off codebases.

NeuroTrain accompanies a survey paper providing a comprehensive taxonomy of SNN training algorithms — see the [paper](#citation) for background and the full algorithmic landscape.

Built on **[snnTorch](https://github.com/jeshraghian/snntorch)** (≥ 0.7), **[Tonic](https://github.com/neuromorphs/tonic)** (≥ 1.0), and **[NeuroBench](https://github.com/NeuroBench/neurobench)**.

> Developed at the [SMILIES Research Group](https://placeholder-website.polito.it), Dept. of Control and Computer Engineering, Politecnico di Torino. &nbsp;·&nbsp; [🌐 Website](https://placeholder-website.polito.it) &nbsp;·&nbsp; [LinkedIn](https://linkedin.com/company/placeholder) &nbsp;·&nbsp; [𝕏](https://x.com/placeholder)

> <a name="citation"></a>**If you use NeuroTrain in your research, please cite:**
>
> Caviglia, A., Marostica, F., Bardini, R., Savino, A., & Di Carlo, S. (2026). *NeuroTrain: surveying Local Learning Rules for Spiking Neural Networks with an Open Benchmarking Framework*. arXiv preprint. https://arxiv.org/abs/PLACEHOLDER

![NeuroTrain framework overview — trainers, networks, and datasets are independently registered components wired at runtime; run_exp_campaign.py enumerates all valid combinations and aggregates results into per-dataset accuracy tables.](docs/figures/framework.png)

```bash
git clone https://gitlabtsgroup.polito.it/neuromorphic/software/snn-training-benchmarking
cd snn-training-benchmarking
pip install -r requirements.txt

# Benchmark matrix: all trainers × models × datasets declared in config/benchmarking.yaml
python run_exp_campaign.py --benchmarking config/benchmarking.yaml --name my_bench

# Custom experiment with optional Optuna HPO
python run_exp_campaign.py --custom config/experiments.yaml --name my_exp

# Reproduce paper results (fill config/paper.yaml with best Optuna outputs first)
make paper
```

| Script | Mode | Use when |
|---|---|---|
| `run_exp_campaign.py --benchmarking <yaml>` | Benchmarking | Systematic matrix of trainer × model × dataset |
| `run_exp_campaign.py --custom <yaml>` | Custom | Named experiments with per-experiment overrides and HPO |
| `experiment.py <spec.json> <output_dir>` | Single run | Called internally; useful for debugging a single spec |

---

## Contents

- [SNN Training Algorithms Benchmarking Results](#snn-training-algorithms-benchmarking-results)
- [Benchmark Your SNN Training Algorithm](#benchmark-your-snn-training-algorithm)
- [Under the Hood — How NeuroTrain Works](#under-the-hood--how-neurotrain-works)
  - [Design Principles](#design-principles)
  - [Repository Structure](#repository-structure)
  - [Configuration System](#configuration-system)
  - [Singularity / Apptainer](#singularity--apptainer)
  - [Dependencies](#dependencies)
- [Reproduce the Paper Benchmarking Results](#reproduce-the-paper-benchmarking-results)
- [Development Status](#development-status)
  - [Implemented Algorithms](#implemented-algorithms)
  - [Networks and Datasets](#networks-and-datasets)
  - [Validated Integration Results](#validated-integration-results)
  - [Testing](#testing)
  - [HPC / SLURM](#hpc--slurm)
  - [Trainer Notes](#trainer-notes)
- [License](#license)

---

# SNN Training Algorithms Benchmarking Results

*Last updated: PLACEHOLDER DATE — updated with each major release. Full per-dataset tables and reproduction instructions: [Reproduce the Paper Benchmarking Results](#reproduce-the-paper-benchmarking-results).*

> 📊 **[PLACEHOLDER — Figure: accuracy heatmap, algorithms (rows) × dataset–architecture combinations (cols), colour-coded from low (light) to high (dark). Metric: test accuracy, mean ± std over 3 seeds with HPO-optimised hyperparameters. Empty cells indicate incompatible combinations.]**

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
Add your experiment to `config/experiments.yaml` with `opt: true` and tunable parameter blocks. Run:

```bash
python run_exp_campaign.py --custom config/experiments.yaml --name my_trainer_hpo
```

Optuna runs a study for each experiment and writes the best config to `experiments/<name>/<exp>/optuna/best_params.yaml`.

**Step 4 — Run your benchmark.**

```bash
# Option A — your trainer only, against all compatible combinations
python run_exp_campaign.py --benchmarking config/benchmarking.yaml \
    --name my_bench
# (set trainers: [my_trainer] in benchmarking.yaml, or leave empty for all)

# Option B — rerun the full matrix including your trainer
python run_exp_campaign.py --benchmarking config/benchmarking.yaml \
    --name full_bench
```

---

# Under the Hood — How NeuroTrain Works

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
             experiments/<campaign>/<exp_name>/
               config.yaml   metrics.json   log.txt
               optuna/       (if opt: true)
```

Every trainer implements `train_sample()` and `reset()` from `BaseTrainer`. Every network implements `forward()` and `reset()` from `BaseSNN`. Compatibility between trainers, models, and datasets is declared via `supported_net_types` in each component's default YAML and resolved automatically by `src/campaign/compatibility.py`.

## Repository Structure

```
snn-training-benchmarking/
│
├── run_exp_campaign.py            # Main entry point (benchmarking + custom modes)
├── experiment.py                  # Single-experiment runner (called per spec)
│
├── config/
│   ├── benchmarking.yaml          # Benchmarking mode: lists trainers, models, datasets
│   ├── experiments.yaml           # Custom mode: named experiments with overrides + HPO
│   ├── paper.yaml                 # Post-HPO best configs for paper results
│   ├── default/
│   │   ├── trainers/              # One YAML per trainer (defaults + tunable blocks)
│   │   ├── models/                # One YAML per network (with per-dataset sections)
│   │   └── datasets/              # One YAML per dataset (timesteps, batch size, etc.)
│   ├── benchmarking/              # Per-trainer benchmarking configs (trainer × dataset)
│   ├── custom/                    # User-created experiment files (not tracked by git)
│   └── vgg9/                      # VGG9-specific experiment configs
│
├── src/
│   ├── trainers/                  # Learning algorithm implementations
│   │   ├── base_trainer.py        # Abstract interface: train_sample() + reset()
│   │   ├── __init__.py            # TRAINER_REGISTRY — maps name → class
│   │   ├── bptt_trainer.py
│   │   └── …                      # One file per algorithm
│   ├── networks/
│   │   ├── base_snn.py            # Abstract interface all networks must implement
│   │   ├── fc_snn.py
│   │   ├── r_snn.py
│   │   ├── conv_snn.py
│   │   └── vgg9*.py               # VGG9 variants
│   ├── datasets/                  # Dataset loaders
│   └── campaign/                  # Orchestration layer
│       ├── campaign_builder.py    # Builds ExperimentSpec list from YAML
│       ├── compatibility.py       # Trainer × model × dataset compatibility
│       ├── config_loader.py       # YAML loading and deep-merge logic
│       ├── experiment_spec.py     # ExperimentSpec dataclass
│       ├── training_loop.py       # train_one_epoch, evaluate
│       ├── optuna_helpers.py      # Tunable block resolution
│       └── results.py             # Output writing
│
├── tests/                         # Integration and dataloader tests
├── hpc/                           # SLURM sbatch scripts
├── docs/                          # Extended documentation
│   ├── HOW_TO_BENCHMARK_YOUR_TRAINER.md
│   └── configs_guide.md
├── Makefile                       # Convenience targets
└── pyproject.toml
```

## Configuration System

NeuroTrain has two operating modes, both launched via `run_exp_campaign.py`.

### Benchmarking mode

Declares which trainers, models, and datasets to compare. The campaign builder generates all valid combinations automatically.

```yaml
# config/benchmarking.yaml
trainers: [bptt, ostl]   # empty list = all in config/default/trainers/
models:   [fc_snn]
datasets: [MNIST, FashionMNIST]
runtime:
  epochs: 50
  device: cuda
  seed: 42
  neurobench: false
opt: false               # set true to run Optuna for every combination
optuna:
  n_trials: 50
  sampler: tpe
```

```bash
python run_exp_campaign.py --benchmarking config/benchmarking.yaml --name my_bench
make bench                  # uses config/benchmarking.yaml by default
make dry-bench              # print experiment list without running
```

### Custom mode

Defines named experiments with explicit overrides. Supports per-experiment Optuna HPO.

```yaml
# config/experiments.yaml
my_experiment:
  trainer:
    name: bptt
    lr: 5e-4                    # plain value (no tuning)
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
  opt: true                     # enable Optuna for this experiment
  optuna:
    n_trials: 50
  trainer:
    name: stsf
    lr:
      value: 1e-3
      type: float
      min: 1e-5
      max: 1e-1
      log: true                 # log-scale sampling
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
python run_exp_campaign.py --custom config/experiments.yaml --name my_exp
make custom
make dry-custom
```

### Tunable parameter blocks

Any scalar in a trainer, model, or dataset config can be made tunable by replacing it with a block:

```yaml
lr:
  value: 1e-3        # used when opt: false
  type: float        # float | int | categorical
  min: 1e-5
  max: 1e-1
  log: true          # log-scale sampling (floats only)

batch_size:
  value: 128
  type: int
  min: 32
  max: 512
  step: 32

loss_type:
  value: ce_rate
  type: categorical
  list: [ce_rate, mse_count, ce_count]
```

When `opt: false`, the loader flattens blocks to their `value` automatically. When `opt: true`, Optuna uses the full block as a search space definition.

### Output structure

```
experiments/<campaign>/<exp_name>/
  config.yaml          ← resolved config for this run
  metrics.json         ← training and evaluation metrics
  log.txt
  optuna/              ← only when opt: true
    trials.csv
    best_params.yaml
    study.db           ← SQLite, open with optuna-dashboard
```

### Visualising HPO Results

When `opt: true` is set, NeuroTrain writes an SQLite study database to `experiments/<campaign>/<exp_name>/optuna/study.db`. [optuna-dashboard](https://github.com/optuna/optuna-dashboard) provides a real-time web UI to inspect trial history, parameter importances, and convergence plots.

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///experiments/<campaign>/<exp_name>/optuna/study.db
# → opens at http://localhost:8080
```

![optuna-dashboard — trial history, hyperparameter importance, and parallel coordinate plots for a NeuroTrain HPO study](docs/figures/optuna_dashboard_screenshot.png)
*optuna-dashboard showing trial accuracy across 50 Optuna TPE trials for a trainer × model × dataset combination. Use the parallel coordinate view to identify which hyperparameters drive accuracy and where the search converged.*

> **Note:** `docs/figures/optuna_dashboard_screenshot.png` is a placeholder — replace it with a screenshot from your own HPO campaign. The optuna-dashboard interface is shown at the [official repository](https://github.com/optuna/optuna-dashboard).

For the complete config reference see [`docs/configs_guide.md`](docs/configs_guide.md).

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
    --bind /path/to/snn-training-benchmarking:/workspace \
    neurotrain.sif \
    bash -c "cd /workspace && python run_exp_campaign.py \
        --benchmarking config/benchmarking.yaml --name my_bench"
```

**3. SLURM + Singularity** — set `APPTAINER_IMAGE` before submitting; the sbatch scripts in `hpc/` handle the rest:

```bash
export APPTAINER_IMAGE=/path/to/neurotrain.sif
sbatch hpc/bench_bptt_mnist.sbatch
```

## Dependencies

| Package | Version | Role |
|---|---|---|
| [torch](https://pytorch.org) | ≥ 2.0 | Core deep learning |
| [snntorch](https://github.com/jeshraghian/snntorch) | ≥ 0.7 | LIF neuron models, surrogate gradients — core SNN engine |
| [tonic](https://github.com/neuromorphs/tonic) | ≥ 1.0 | Event-based dataset loading (N-MNIST, DVSGesture, SHD, DVS-CIFAR10) |
| [neurobench](https://github.com/NeuroBench/neurobench) | latest | Neuromorphic benchmarking metrics and datasets |
| [optuna](https://optuna.org) | ≥ 3.0 | Built-in hyperparameter optimisation |
| `pyyaml` | ≥ 6.0 | Config parsing |
| `tensorboard` | ≥ 2.13 | Experiment tracking |

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, CUDA optional (MPS and CPU supported via `device: auto`).

---

# Reproduce the Paper Benchmarking Results

Paper results are stored in `config/paper.yaml` — one named experiment per trainer × model × dataset combination, with HPO-optimised hyperparameters as plain scalar values.

```bash
# Reproduce all paper results
make paper
# equivalent to:
python run_exp_campaign.py --custom config/paper.yaml --name paper
```

Each run logs the full config, seed, and git commit hash to `experiments/paper/<exp_name>/`, ensuring every result is traceable.

### MNIST

*Test accuracy, rate-coded input, 10 classes.*

| Algorithm | FC-SNN | Conv-SNN | R-SNN |
|---|---|---|---|
| BPTT | — | — | — |
| OSTL | — | — | — |
| E-prop | — | — | — |
| ESD-RTRL | — | — | — |
| ETLP | — | — | — |
| STSF | — | — | — |
| DRTP | — | — | — |
| OTTT | — | — | — |

### Fashion-MNIST

*Test accuracy, rate-coded input, 10 classes.*

| Algorithm | FC-SNN | Conv-SNN | R-SNN |
|---|---|---|---|
| BPTT | — | — | — |
| OSTL | — | — | — |
| E-prop | — | — | — |
| ESD-RTRL | — | — | — |
| ETLP | — | — | — |
| STSF | — | — | — |
| DRTP | — | — | — |
| OTTT | — | — | — |

### CIFAR-10

*Test accuracy, rate-coded input, 10 classes.*

| Algorithm | Conv-SNN | VGG9 | R-SNN |
|---|---|---|---|
| BPTT | — | — | — |
| E-prop | — | — | — |
| ESD-RTRL | — | — | — |
| DRTP | — | — | — |
| OTTT | — | — | — |
| STOP | — | — | — |

### SVHN

*Test accuracy, rate-coded input, 10 classes.*

| Algorithm | Conv-SNN | VGG9 | R-SNN |
|---|---|---|---|
| BPTT | — | — | — |
| E-prop | — | — | — |
| ESD-RTRL | — | — | — |
| OTTT | — | — | — |

### N-MNIST

*Test accuracy, event-based neuromorphic input, 10 classes.*

| Algorithm | FC-SNN | R-SNN |
|---|---|---|
| BPTT | — | — |
| OSTL | — | — |
| OSTTP | — | — |
| E-prop | — | — |
| ESD-RTRL | — | — |
| ETLP | — | — |
| OTTT | — | — |

### DVSGesture

*Test accuracy, event-based neuromorphic input, 11 gesture classes.*

| Algorithm | Conv-SNN | VGG9 | R-SNN |
|---|---|---|---|
| BPTT | — | — | — |
| E-prop | — | — | — |
| ESD-RTRL | — | — | — |
| DECOLLE | — | — | — |
| OTTT | — | — | — |

---

# Development Status

## Implemented Algorithms

🟢 implemented · 🟡 under development · 🔴 planned

| Algorithm | File | Category | Status | Compatible Net Types |
|---|---|---|---|---|
| BPTT | `bptt_trainer.py` | Backprop-through-time | 🟢 | fc, rec, conv |
| OSTL | `ostl_trainer.py` | Online / local | 🟢 | fc |
| OSTTP | `osttp_trainer.py` | Online / local | 🟢 | rec |
| E-prop | `eprop_trainer.py` | Online / local | 🟢 | rec |
| ESD-RTRL | `es_d_rtrl_trainer.py` | Online / local | 🟢 | fc, rec, conv |
| ETLP | `etlp_trainer.py` | Online / local | 🟢 | fc, rec, conv |
| STSF | `stsf_trainer.py` | Online / local | 🟢 | fc, rec, conv |
| DRTP | `drtp_trainer.py` | Feedback alignment | 🟢 | fc |
| OTTT | `ottt_trainer.py` | Online / global | 🟡 | conv |
| OTPE | `otpe_trainer.py` | Online / global | 🟡 | fc, rec, conv |
| DECOLLE | `decolle_trainer.py` | Local losses | 🟡 | fc, rec, conv |
| STOP | `stop_trainer.py` | Local losses | 🟡 | fc, rec, conv |
| TP | `tp_trainer.py` | Target propagation | 🟡 | fc, rec, conv |
| ELL | `ell_trainer.py` | Eligibility traces | 🟡 | fc |
| FELL | `fell_trainer.py` | Eligibility traces | 🟡 | fc |
| BELL | `bell_trainer.py` | Eligibility traces | 🟡 | fc |
| STLLR | `stllr_trainer.py` | Spike-timing | 🔴 | fc, rec, conv |

## Networks and Datasets

### Architectures

| Architecture | `net_type` | Notes |
|---|---|---|
| FC-SNN | `fc` | Fully-connected, baseline for rate-coded data |
| Conv-SNN | `conv` | Convolutional, general image tasks |
| R-SNN | `rec` | Recurrent, temporal / event-based tasks |
| VGG9 | `conv` | Multiple dataset-specific variants |

### Datasets

| Dataset | Type | Input | Classes | Status |
|---|---|---|---|---|
| MNIST | Rate-coded | 28×28 | 10 | 🟢 |
| Fashion-MNIST | Rate-coded | 28×28 | 10 | 🟢 |
| CIFAR-10 | Rate-coded | 32×32×3 | 10 | 🟢 |
| SVHN | Rate-coded | 32×32×3 | 10 | 🟢 |
| NMNIST | Event-based | 34×34 | 10 | 🟢 |
| DVS-CIFAR10 | Event-based | 128×128×2 | 10 | 🟢 |
| DVSGesture | Event-based | 128×128×2 | 11 | 🟢 works with caching |
| SHD | Event-based | audio | 20 | 🟡 occasional issues |

## Validated Integration Results

*Full training loop results used to validate trainer × network × dataset combinations. Each result is pinned to a specific commit for traceability.*

| Test | Network | Dataset | Result (train / test) | Commit |
|---|---|---|---|---|
| `bptt_cifar10_vgg9.py` | VGG9 | CIFAR-10 | epoch 10: 0.8685 / 0.7212 | `bcae958` |
| `bptt_dvsgest_vgg9.py` | VGG9 | DVSGesture | epoch 40: 1.000 / 0.9356 | `bcae958` |
| `ostl_mnist_fc.py` | FC-SNN | MNIST | epoch 25: 0.9984 / 0.9765 | `71a774a` |
| `ostl_nmnist_r.py` | R-SNN | NMNIST | epoch 10: 0.9137 / 0.8961 | `71a774a` |
| `stsf_mnist_fc.py` | FC-SNN | MNIST | epoch 50: 0.9657 / 0.9627 | `f788ee2` |
| `stsf_nmnist_fc.py` | FC-SNN | NMNIST | epoch 10: 0.9128 / 0.9076 | — |

*Extended results on VGG9 networks — in progress, not yet HPO-optimised:*

| Config | Trainer | Dataset | Epoch | Train Acc | Test Acc | Commit | Notes |
|---|---|---|---|---|---|---|---|
| `ottt_vgg9_svhn.yaml` | OTTT | SVHN | 70 | 0.80 | 0.78 | `07ef17d` | Oscillating; peak test 0.92 |
| `ottt_vgg9_cifar10.yaml` | OTTT | CIFAR-10 | 70 | 0.42 | 0.42 | `07ef17d` | Oscillating; still low |
| `tp_vgg9_dvsgesture.yaml` | TP | DVSGesture | 70 | 0.85 | 0.80 | `07ef17d` | Oscillating; peak test 0.88 |
| `tp_vgg9_dvscifar10.yaml` | TP | DVS-CIFAR10 | 70 | 0.40 | 0.23 | `07ef17d` | Suspected overfitting |

## Testing

```bash
python -m pytest tests/
python tests/dataloaders/test_mnist_loader.py     # dataloader smoke test
python tests/bptt_cifar10_vgg9.py                 # single integration test
```

## HPC / SLURM

Per-trainer, per-dataset sbatch scripts live in `hpc/`. Key Makefile targets:

```bash
make all-opt             # submit HPO jobs for all trainers across all datasets
make opt-bptt            # submit HPO jobs for BPTT only
sbatch hpc/bench_bptt_mnist.sbatch   # submit a single job directly
```

Adapt `--partition`, `--gres`, and `--time` in each sbatch script to your cluster. Set `APPTAINER_IMAGE` to run inside a Singularity container (see above).

## Trainer Notes

**TP (`tp_trainer.py`):** `batch_size ≥ 2` is required — the contrastive loss degenerates to zero gradient with B=1. Set `out_integrator: true` in the model config when pairing with FC-SNN or R-SNN for faithful evaluation. VGG9 already has a correct leaky-integrator head.

---

## License

MIT — see `LICENSE`.