# How to Benchmark Your SNN Training Algorithm with NeuroTrain

This guide walks through the full process of integrating a new SNN training algorithm into NeuroTrain and running a systematic benchmark comparison against existing algorithms.

---

## Framework Design Principles

Before implementing, understand the two core principles that shape every trainer in NeuroTrain.

**1. Networks are standard snnTorch SNNs — your trainer drives everything else.**

NeuroTrain networks are plain snnTorch models: LIF neurons, standard PyTorch layers, no custom hooks, no algorithm-specific modifications. The network's job is to define the architecture and hold state. All training logic — how gradients are computed or approximated, how weights are updated, how temporal credit is assigned, how traces are maintained — lives exclusively in the trainer. This enforces the separation that makes fair comparison possible: the same network runs unchanged under any compatible trainer.

**2. All temporal state is reset between batches by the trainer.**

snnTorch neurons carry membrane potential across timesteps within each input sequence. At the start of every batch, `reset()` is called. Your trainer is responsible for resetting both the network's neuron states and any algorithm-specific state (eligibility traces, local error signals, running averages). This keeps batches independent and ensures that benchmark results are not affected by hidden state leakage across data samples.

If you find yourself modifying a network file only to make your learning rule work, or carrying state across the `reset()` boundary without explicit intent, the decomposition is likely wrong.

---

## Workflow Overview

1. [Implement your trainer](#step-1--implement-your-trainer)
2. [Add default config and register compatibility](#step-2--add-default-config-and-register)
3. [Define and run HPO](#step-3--define-and-run-hpo-optional)
4. [Run your benchmark](#step-4--run-your-benchmark)
5. [Open a pull request](#contributing)

---

## Step 1 — Implement Your Trainer

Create `src/trainers/my_trainer.py`. Your trainer must extend `BaseTrainer` and implement exactly two abstract methods:

```python
from trainers.base_trainer import BaseTrainer
import torch
import torch.nn as nn

class MyTrainer(BaseTrainer):

    def __init__(self, network: nn.Module, lr: float, batch_size: int, **kwargs):
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        # initialise algorithm-specific state here (traces, local errors, etc.)

    def train_sample(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Train on a single batch. All training logic goes here.
        Returns: (loss, predictions)
        """
        self.optimizer.zero_grad()
        # forward pass through the snnTorch network
        # compute loss and weight update (backprop, traces, local rules, etc.)
        loss = ...
        pred = ...
        loss.backward()
        self.optimizer.step()
        return loss, pred

    def reset(self) -> None:
        """
        Reset all temporal state. Called between every batch.
        Must reset both network neuron states and any algorithm-specific state.
        """
        self.network.reset()       # resets snnTorch LIF membrane potentials
        # reset eligibility traces, local error accumulators, etc.
```

`BaseTrainer` extends `nn.Module`, so your trainer can hold submodules, buffers, and parameters. `**kwargs` not consumed by `__init__` are silently ignored — this keeps your trainer compatible with the config-passing mechanism.

### Abstract method specification

| Method | Called by | Expected behaviour |
|---|---|---|
| `train_sample(data, target)` | `src/campaign/training_loop.py` | One weight update per call; returns `(loss, pred)` |
| `reset()` | `src/campaign/training_loop.py` at the start of each batch | Clears all temporal state — neuron potentials, traces, accumulators |

---

## Step 2 — Add Default Config and Register

### Register in `src/trainers/__init__.py`

Add two lines:

```python
from trainers.my_trainer import MyTrainer
TRAINER_REGISTRY["my_trainer"] = MyTrainer
```

### Create `config/default/trainers/my_trainer.yaml`

This file defines default hyperparameters and declares which network types your trainer supports. Compatibility with models and datasets is resolved automatically from `supported_net_types` — no separate registration needed.

```yaml
# config/default/trainers/my_trainer.yaml

name: my_trainer
supported_net_types: [fc, rec]   # options include: fc | rec | conv | vgg9

# Hyperparameters — plain values or tunable blocks (for Optuna)
lr:
  value: 1.0e-3
  type: float
  min: 1.0e-5
  max: 1.0e-1
  log: true

my_param:
  value: 0.9
  type: float
  min: 0.5
  max: 0.99

loss_type:
  value: ce_rate
  type: null    # not tunable — keep fixed; options: ce_rate, mse_count, ce_count
```

Parameters marked `type: null` (or with no `type` key) are treated as plain values in both normal and Optuna runs. Parameters with a `type` are treated as tunable when `opt: true`.

### Verify compatibility

Run a dry-run to confirm your trainer appears and matches the expected combinations:

```bash
# Create config/benchmarking/my_trainer.yaml, then:
python3 run_exp_campaign.py --benchmarking config/benchmarking/my_trainer.yaml --dry-run
```

---

## Step 3 — Define and Run HPO (Optional)

Hyperparameter optimisation (HPO) via Optuna is built into NeuroTrain and is **optional**. You can run a single training experiment with fixed hyperparameters, or run an HPO study to find the best configuration before the final benchmark. Both paths use a custom YAML file (e.g. `config/paper_examples.yaml` or your own `config/custom/my_experiments.yaml`) — the only difference is setting `opt: true` or `opt: false`.

### Running without HPO (fixed hyperparameters)

Set `opt: false` (or omit it — false is the default). All parameters are used as plain values.

```yaml
# config/custom/my_experiments.yaml
my_trainer_mnist_fc:
  opt: false
  trainer:
    name: my_trainer
    lr: 1.0e-3          # plain value, used as-is
    my_param: 0.9
  model:
    name: fc_snn
  dataset:
    name: MNIST
    T: 25
  runtime:
    epochs: 50
    device: cuda
    seed: 42
```

```bash
python3 run_exp_campaign.py --custom config/custom/my_experiments.yaml --name my_trainer_run
```

### Running with HPO

Set `opt: true` and replace any parameter you want to tune with a tunable block. Optuna samples values from the defined search space across `n_trials` trials.

### Add your HPO experiment to `config/custom/my_experiments.yaml`

```yaml
my_trainer_mnist_fc:
  opt: true
  optuna:
    n_trials: 50
    sampler: tpe          # tpe | random | cmaes
    direction: maximize
  trainer:
    name: my_trainer
    lr:
      value: 1.0e-3
      type: float
      min: 1.0e-5
      max: 1.0e-1
      log: true
    my_param:
      value: 0.9
      type: float
      min: 0.5
      max: 0.99
  model:
    name: fc_snn
    beta:
      value: 0.9
      type: float
      min: 0.8
      max: 0.99
  dataset:
    name: MNIST
    T:
      value: 25
      type: int
      min: 10
      max: 30
      step: 5
  runtime:
    epochs: 20            # short runs during HPO
    device: cuda
    seed: 42
```

Run the study:

```bash
python3 run_exp_campaign.py --custom config/custom/my_experiments.yaml \
    --name my_trainer_hpo
```

Optuna writes results to `experiments/my_trainer_hpo/experiments/my_trainer_mnist_fc/optuna/`. The best config is in `best_params.yaml`.

### Save best params to `config/paper_examples.yaml`

Copy the best hyperparameters from `best_params.yaml` into `config/paper_examples.yaml` (or your own custom YAML) as plain scalar values (no tunable blocks needed):

```yaml
# config/paper_examples.yaml

my_trainer_mnist_fc:
  name: my_trainer_mnist_fc
  opt: false
  trainer:
    name: my_trainer
    lr: 3.7e-3           # best from Optuna
    my_param: 0.87       # best from Optuna
  model:
    name: fc_snn
    beta: 0.93
  dataset:
    name: MNIST
    T: 25
  runtime:
    epochs: 100          # full training run — longer than HPO
    device: cuda
    seed: 42
```

Repeat for each compatible trainer × model × dataset combination you want to include.

---

## Step 4 — Run Your Benchmark

With your trainer registered and your HPO-tuned final configs saved, you have two options.

**Option A — Benchmark your trainer only**, without rerunning existing algorithms:

```bash
# Create config/benchmarking/my_trainer.yaml, then:
python3 run_exp_campaign.py --benchmarking config/benchmarking/my_trainer.yaml \
    --name my_trainer_bench
```

Or use custom mode with your HPO-tuned configs:

```bash
python3 run_exp_campaign.py --custom config/paper_examples.yaml \
    --name my_trainer_paper
```

This is the recommended path when you want to add your algorithm to the comparison without rerunning all existing results.

**Option B — Rerun the full benchmark**, including your trainer alongside all existing algorithms:

```bash
# Leave trainers: [] (empty = all) in your benchmarking YAML:
python3 run_exp_campaign.py --benchmarking config/paper_benchmarking.yaml \
    --name full_bench
```

Use this when you want a complete, fresh comparison — for example after updating shared components (network architectures, dataloaders) that affect all algorithms.

### Inspecting results

Campaign-level outputs:

```text
experiments/<campaign>/
  campaign.yaml
  summary.json
  summary.csv
  experiments/
    <exp_name>/
      config.yaml
      metrics.json
      log.txt
      optuna/           # only when opt: true
        trials.csv
        best_params.yaml
        study.db        # SQLite — open with optuna-dashboard
```

**NeuroBench evaluation:** set `neurobench: true` in the `runtime` block to automatically run a NeuroBench benchmark after training. Results are written under a `neurobench` key in `metrics.json` and included as `nb_*` columns in the campaign-level `summary.csv`. Metrics include activation sparsity, membrane updates, memory footprint, connection sparsity, and parameter count — enabling direct comparison of efficiency alongside accuracy across all algorithms.

```yaml
runtime:
  epochs: 50
  device: cuda
  neurobench: true    # adds nb_* columns to summary.csv
```

When `opt: true`, NeuroTrain writes an SQLite study database. Use [optuna-dashboard](https://github.com/optuna/optuna-dashboard) to inspect trial history, hyperparameter importances, and convergence plots in real time:

```bash
optuna-dashboard "sqlite:////absolute/path/to/experiments/<campaign>/experiments/<exp_name>/optuna/study.db"
# → opens at http://localhost:8080
```

![optuna-dashboard — trial history, hyperparameter importance, and parallel coordinate plots](figures/optuna-dashboard-screenshot.png)

The parallel coordinate view is particularly useful for identifying which hyperparameters drive accuracy and where the search has converged — use it to decide whether to extend the study or commit the best config to your custom YAML.

### Generating results tables and heatmap

Once your campaign has run, use `src/generate_results.py` to produce per-dataset Markdown accuracy tables and a heatmap PNG from `summary.csv`:

```bash
# Tables + heatmap
python3 src/generate_results.py experiments/my_trainer_bench/

# Also include NeuroBench metrics table
python3 src/generate_results.py experiments/my_trainer_bench/ --neurobench
```

## Contributing

Open a pull request including:
- `src/trainers/my_trainer.py`
- `config/default/trainers/my_trainer.yaml`
- Your HPO-tuned final entries in `config/paper_examples.yaml`
- Integration test with pinned result
- A one-row summary for the algorithm table in `README.md`

If you have questions or run into issues, open an issue in the repository.