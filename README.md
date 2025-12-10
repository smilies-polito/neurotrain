# SNN Training Benchmarking

A modular, scalable benchmarking platform for spiking neural network (SNN) learning algorithms. Supports plug-and-play algorithms, advanced logging, reproducibility, and systematic research analysis with **NeuroBench integration**.

## Features

- **Reproducible Experiments**: Comprehensive seed management, environment logging, and git commit tracking
- **Configuration System**: YAML/JSON config files with CLI override support
- **Checkpointing**: Automatic checkpoint saving with resume capability
- **Plug-and-Play Trainers**: Easily add new learning algorithms (BPTT, STSF)
- **TensorBoard Integration**: Real-time training visualization
- **NeuroBench Integration**: Official neuromorphic benchmark datasets and metrics
- **Automated Benchmarking**: Compare algorithms across multiple datasets

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://gitlabtsgroup.polito.it/neuromorphic/software/snn-training-benchmarking.git
cd snn-training-benchmarking

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Dependencies Only

```bash
pip install -r requirements.txt
```

## Quick Start

### Using Config File (Recommended)

```bash
# Run with default MNIST config
python main.py --config configs/mnist_default.yaml

# Override specific parameters via CLI
python main.py --config configs/mnist_default.yaml --lr 0.001 --epochs 50
```

### Using CLI Only (Legacy)

```bash
python main.py --dataset MNIST --epochs 100 --batch-size 256 --lr 0.01
```

### Resume Training

```bash
# Auto-resume from latest checkpoint
python main.py --config configs/mnist_default.yaml --resume

# Resume from specific checkpoint
python main.py --resume-from experiments/STSF_MNIST/20231128_120000/checkpoints/checkpoint_latest.pt
```

## Comprehensive Benchmarking

This platform supports systematic comparison of SNN learning algorithms across multiple datasets with NeuroBench integration.

### Available Learning Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| **STSF** | Local Learning | Spiking Time Sparse Feedback - bio-plausible, no backprop |
| **BPTT** | Gradient-based | Backpropagation Through Time with surrogate gradients |
| **OTTT** | Local Learning | Online Training Through Time with eligibility traces |

### Available Datasets

#### Standard Image Classification
| Dataset | Input Size | Classes | Description |
|---------|------------|---------|-------------|
| MNIST | 784 | 10 | Handwritten digits |
| FashionMNIST | 784 | 10 | Fashion items |
| CIFAR10 | 3072 | 10 | Natural images |
| SVHN | 3072 | 10 | Street view house numbers |
| DVSGesture | 1156 | 11 | Event-based gestures |

#### NeuroBench Official Benchmarks
| Dataset | Task | Description |
|---------|------|-------------|
| SpeechCommands | Classification | Google Speech Commands (12 keywords) |
| WISDM | Classification | Human Activity Recognition (6 activities) |
| PrimateReaching | Regression | Motor prediction from neural recordings |
| MackeyGlass | Regression | Chaotic time series prediction |

### Run Full Benchmark Suite

Compare BPTT vs STSF vs OTTT across all classification datasets:

```bash
# Run comprehensive benchmark (all datasets, both algorithms)
python run_all_benchmarks.py --epochs 50 --device cuda

# With custom settings
python run_all_benchmarks.py \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.001 \
    --device cuda \
    --output-dir ./benchmark_results
```

This will:
1. Train both **BPTT** and **STSF** on each dataset
2. Profile CPU/GPU timing with PyTorch profiler
3. Run **NeuroBench evaluation** for SNN-specific metrics
4. Save comprehensive results to JSON
5. Print comparison summary tables

### Single Algorithm Comparison

```bash
# Run benchmark on specific config
python src/benchmark_runner.py --config configs/benchmark_comparison.yaml

# Override output directory
python src/benchmark_runner.py --config configs/benchmark_comparison.yaml --output-dir ./my_results
```

### NeuroBench Metrics

The benchmark automatically computes these NeuroBench metrics:

**Static Metrics** (model properties):
- `Footprint` - Memory footprint
- `ConnectionSparsity` - Weight sparsity
- `ParameterCount` - Total parameters

**Workload Metrics** (inference efficiency):
- `ActivationSparsity` - Overall spike sparsity
- `ActivationSparsityByLayer` - Per-layer breakdown
- `SynapticOperations` - Computational cost (like MACs)
- `MembraneUpdates` - Neuron update overhead
- `ClassificationAccuracy` - Test accuracy

### Benchmark Output

Results are saved to `benchmark_results/full_benchmark_<timestamp>.json`:

```json
{
  "MNIST": {
    "bptt": {
      "final_accuracy": 0.9823,
      "total_wall_time_s": 245.3,
      "avg_epoch_cpu_ms": 4521.2,
      "neurobench": {
        "ActivationSparsity": 0.92,
        "SynapticOperations": 156000,
        ...
      }
    },
    "stsf": {
      "final_accuracy": 0.9172,
      "total_wall_time_s": 89.1,
      ...
    }
  },
  ...
}
```

### Using Singularity Container

For HPC environments, use the provided Singularity container:

```bash
# Build container (requires fakeroot or sudo)
cd src
singularity build --fakeroot snn-training-benchmarking.sif snn-training-benchmarking.def

# Run training
singularity exec snn-training-benchmarking.sif python3 main.py --config configs/mnist_default.yaml

# Run full benchmark
singularity exec snn-training-benchmarking.sif python3 run_all_benchmarks.py --epochs 50 --device cuda

# Interactive shell
singularity shell snn-training-benchmarking.sif
```

### TensorBoard Visualization

Monitor training in real-time:

```bash
# Inside container
singularity exec snn-training-benchmarking.sif tensorboard --logdir=experiments --bind_all

# Then open http://localhost:6006 in browser
# For remote servers, use SSH port forwarding:
# ssh -L 6006:localhost:6006 user@server
```

## Configuration

### Config File Structure

```yaml
experiment:
  name: "STSF_MNIST"
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
  optimizer: null  # null for manual updates, "adam" for optimizer

trainer:
  name: "stsf"
  update_last: false
  update_every: 1
  seq_batch: 1

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

### Available Configs

| Config | Description |
|--------|-------------|
| `mnist_default.yaml` | Standard MNIST training with STSF |
| `mnist_quantized.yaml` | Quantized training for hardware deployment |
| `fashionmnist_default.yaml` | FashionMNIST training with STSF |
| `cifar10_default.yaml` | CIFAR10 training with STSF |
| `benchmark_comparison.yaml` | Algorithm comparison config (BPTT vs STSF) |

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML/JSON config file | None |
| `--resume` | Resume from latest checkpoint | False |
| `--resume-from` | Resume from specific checkpoint | None |
| `--dataset` | Dataset name | MNIST |
| `--epochs` | Number of epochs | 100 |
| `--batch-size` | Batch size | 256 |
| `--lr` | Learning rate | 0.01 |
| `--T` | Number of timesteps | 10 |
| `--seed` | Random seed | 42 |
| `--quantization` | Enable quantization | False |

## Project Structure

```
snn-training-benchmarking/
├── main.py                 # Entry point
├── configs/                # YAML configuration files
│   ├── mnist_default.yaml
│   ├── mnist_quantized.yaml
│   └── ...
├── src/
│   ├── datasets/          # Data loaders
│   ├── networks/          # Network architectures
│   ├── trainers/          # Learning algorithms
│   ├── utils/             # Utilities
│   │   ├── config.py      # Configuration system
│   │   ├── checkpoint.py  # Checkpointing
│   │   └── experiment_logger.py  # Reproducibility logging
│   └── LearningAlgorithms.py
├── tests/                  # Test suite
├── experiments/            # Experiment outputs (gitignored)
├── requirements.txt
└── pyproject.toml
```

## Reproducibility

Every experiment automatically logs:

- **Seeds**: Python, NumPy, PyTorch, CUDA seeds
- **Environment**: Python version, PyTorch version, CUDA version, device info
- **Code Version**: Git commit hash, branch, dirty state
- **Configuration**: Complete config used for the run
- **RNG State**: Full random state for exact reproducibility on resume

All this information is saved to `experiment_context.json` in the experiment directory.

## TensorBoard

View training progress in TensorBoard:

```bash
tensorboard --logdir experiments/
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_config.py
```

## Code Quality

```bash
# Format code
black src/ tests/ main.py

# Check linting
flake8 src/ tests/ main.py

# Run all pre-commit hooks
pre-commit run --all-files
```

## Adding New Learning Algorithms

1. Create a new trainer in `src/trainers/`:

```python
from trainers.base_trainer import BaseTrainer

class MyTrainer(BaseTrainer):
    def __init__(self, network, lr, ...):
        super().__init__()
        self.network = network
        # ...

    def train_sample(self, data, target):
        # Implement training logic
        return loss, pred

    def reset(self):
        self.network.reset()
```

2. Register in `main.py`:

```python
def get_trainer(trainer_name):
    trainers = {
        "stsf": STSFTrainer,
        "my_trainer": MyTrainer,  # Add here
    }
    return trainers[trainer_name]
```

3. Use in config:

```yaml
trainer:
  name: "my_trainer"
```

## Roadmap

- [x] BPTT (Backpropagation Through Time) trainer
- [x] NeuroBench integration (datasets + metrics)
- [x] Automated benchmarking campaigns
- [ ] Additional learning algorithms (e-prop, STDP)
- [ ] Convolutional SNN architectures
- [ ] Regression task support for NeuroBench
- [ ] Interactive visualization dashboard
- [ ] MLflow integration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make changes and ensure tests pass
4. Format code with `black` and check with `flake8`
5. Commit and push
6. Create a merge request

## License

MIT License

## Acknowledgments

- [snntorch](https://github.com/jeshraghian/snntorch) - Spiking neural network library
- [NeuroBench](https://neurobench.ai) - Neuromorphic computing benchmarks and metrics
- [Tonic](https://github.com/neuromorphs/tonic) - Neuromorphic datasets
