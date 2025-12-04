# SNN Training Benchmarking

A modular, scalable benchmarking platform for spiking neural network (SNN) learning algorithms. Supports plug-and-play algorithms, advanced logging, reproducibility, and systematic research analysis.

## Features

- **Reproducible Experiments**: Comprehensive seed management, environment logging, and git commit tracking
- **Configuration System**: YAML/JSON config files with CLI override support
- **Checkpointing**: Automatic checkpoint saving with resume capability
- **Plug-and-Play Trainers**: Easily add new learning algorithms
- **TensorBoard Integration**: Real-time training visualization
- **Multiple Datasets**: MNIST, FashionMNIST, CIFAR10, SVHN, DVSGesture

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
| `mnist_default.yaml` | Standard MNIST training |
| `mnist_quantized.yaml` | Quantized training for hardware |
| `fashionmnist_default.yaml` | FashionMNIST training |
| `cifar10_default.yaml` | CIFAR10 training |

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

- [ ] Additional learning algorithms (BPTT, e-prop, STDP)
- [ ] Convolutional SNN architectures
- [ ] NeuroBench integration
- [ ] Benchmarking campaigns
- [ ] Interactive visualization
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
- [Tonic](https://github.com/neuromorphs/tonic) - Neuromorphic datasets
