# Project State Log
> Last Updated: 2025-12-03
> Version: 0.1.0

## 1. Project Identity

**Name:** SNN Training Benchmarking  
**Purpose:** Modular platform for benchmarking **SNN learning algorithms** (not architectures)  
**Repository:** `https://gitlabtsgroup.polito.it/neuromorphic/software/snn-training-benchmarking`

---

## 2. Architecture Overview

```
snn-training-benchmarking/
├── main.py                      # Entry point
├── configs/                     # YAML experiment configs
│   ├── mnist_default.yaml
│   ├── mnist_quantized.yaml
│   ├── fashionmnist_default.yaml
│   └── cifar10_default.yaml
├── src/
│   ├── trainers/                # Learning algorithms
│   │   ├── base_trainer.py      # Abstract interface
│   │   └── stsf_trainer.py      # STSF implementation
│   ├── networks/                # SNN architectures (controlled variable)
│   │   ├── base_network.py
│   │   └── fc_network.py        # Fully-connected LIF network
│   ├── datasets/                # Data loaders
│   │   ├── get_loader.py        # Factory function
│   │   ├── mnist_loader.py
│   │   ├── fashionmnist_loader.py
│   │   ├── cifar10_loader.py
│   │   ├── svhn_loader.py
│   │   └── dvsgesture_loader.py
│   ├── utils/
│   │   ├── config.py            # Dataclass-based config system
│   │   ├── checkpoint.py        # Checkpointing with graceful exit
│   │   ├── experiment_logger.py # Reproducibility logging
│   │   ├── parameters.py        # CLI argument parsing
│   │   ├── quantizer.py         # Fixed-point utilities
│   │   └── helpers.py           # Device selection, seeding
│   └── LearningAlgorithms.py    # Train/eval loops
├── tests/                       # Pytest suite
└── experiments/                 # Output directory (gitignored)
```

---

## 3. Implementation Status

### 3.1 Learning Algorithms (Trainers)

| Algorithm | Status | File | Notes |
|-----------|--------|------|-------|
| STSF (Spiking Time Sparse Feedback) | ✅ Complete | `stsf_trainer.py` | Local, hardware-friendly |
| BPTT (Backpropagation Through Time) | 🟡 Available via snnTorch | — | Use `snntorch.backprop.BPTT`; needs wrapper for `BaseTrainer` interface |
| e-prop (Eligibility Propagation) | 🔲 Not Started | — | Biologically plausible |
| STDP (Spike-Timing-Dependent Plasticity) | 🔲 Not Started | — | Unsupervised |

> **Note**: CLI flag `--bptt` exists in `parameters.py` but trainer is commented out in `main.py` (line 205)

### 3.2 Network Architectures

| Architecture | Status | File | Input Shape |
|--------------|--------|------|-------------|
| FCNetwork (Fully-Connected LIF) | ✅ Complete | `fc_network.py` | `[T, B, features]` |
| ConvNetwork (Convolutional LIF) | 🔲 Not Started | — | `[T, B, C, H, W]` |

### 3.3 Datasets

| Dataset | Status | Loader | Input Size | Classes |
|---------|--------|--------|------------|---------|
| MNIST | ✅ Complete | `mnist_loader.py` | 784 | 10 |
| FashionMNIST | ✅ Complete | `fashionmnist_loader.py` | 784 | 10 |
| CIFAR10 | ✅ Complete | `cifar10_loader.py` | 3072 | 10 |
| SVHN | ✅ Complete | `svhn_loader.py` | 3072 | 10 |
| DVSGesture | ✅ Partial | `dvsgesture_loader.py` | Event-based | 11 |

### 3.4 Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| Config system (dataclasses + YAML) | ✅ Complete | CLI override support |
| Checkpointing | ✅ Complete | Best/latest/periodic saving |
| Experiment logging | ✅ Complete | Seeds, git, environment |
| TensorBoard integration | ✅ Complete | Metrics logging |
| Test suite (pytest) | ✅ Complete | ~90% coverage |
| Pre-commit hooks | ✅ Complete | Black, flake8, isort |

### 3.5 NeuroBench Integration

| Component | Status | Library | Notes |
|-----------|--------|---------|-------|
| NeuroBench in container | ✅ Available | — | v2.1.0 in `.def` |
| SNNTorchModel wrapper | 🔲 Use directly | `neurobench.models.SNNTorchModel` | No custom code needed |
| Static metrics | 🔲 Use directly | `neurobench.metrics.static.*` | FootprintMemory, ConnectionSparsity |
| Workload metrics | 🔲 Use directly | `neurobench.metrics.workload.*` | ActivationSparsity, SynapticOperations |
| Benchmark harness | 🔲 Use directly | `neurobench.benchmarks.Benchmark` | No custom code needed |

### 3.6 Benchmarking Metrics

| Metric | Status | Library | Notes |
|--------|--------|---------|-------|
| Accuracy at N epochs | 🔲 Not Started | Custom (minimal) | Track during training loop |
| CPU/CUDA timing | 🔲 Use directly | `torch.profiler` | No custom timing code |
| GPU events | 🔲 Use directly | `torch.cuda.Event` | Built-in CUDA timing |
| Memory profiling | 🔲 Use directly | `torch.profiler` | profile_memory=True |
| Power consumption | 🔲 Deferred | `pynvml` (optional) | Not critical for MVP |

### 3.7 snnTorch Built-ins to Reuse

| Feature | Library Function | Notes |
|---------|-----------------|-------|
| BPTT Training | `snntorch.backprop.BPTT` | Complete training loop |
| Loss Functions | `snntorch.functional.mse_count_loss`, `ce_rate_loss`, etc. | All standard losses |
| Surrogate Gradients | `snntorch.surrogate.*` | Built into neurons |
| Spike Encoding | `snntorch.spikegen.*` | Rate, latency encoding |

---

## 4. Dependencies

### 4.1 Core (requirements.txt)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
snntorch>=0.7.0
tonic>=1.0.0
pyyaml>=6.0
tensorboard>=2.13.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

### 4.2 Container Versions (snn-training-benchmarking.def)

```
snntorch==0.9.4
neurobench==2.1.0
ray[all] (latest)
```

### 4.3 Development

```
pytest>=7.4.0
pytest-cov>=4.1.0
flake8>=6.1.0
black>=23.0.0
pre-commit>=3.3.0
isort>=5.12.0
```

---

## 5. Key Design Decisions

1. **Learning algorithm in Trainer, not Network**: `FCNetwork.forward()` is pure inference; weight updates happen in `Trainer.train_sample()`

2. **Input shape convention**: `[timesteps, batch_size, features]` for all temporal data

3. **Spike accumulation for classification**: Output = sum of spikes over time, not final membrane

4. **Quantization support**: `quant=True` enables fixed-point arithmetic for hardware deployment

5. **snnTorch native**: Use `snn.Leaky`, `snn.Synaptic` directly, no custom neurons

---

## 6. Known Limitations

1. Only `FCNetwork` implemented (no convolutional)
2. Only STSF algorithm implemented
3. No NeuroBench integration yet
4. No power measurement infrastructure
5. No algorithmic complexity metrics
6. DVSGesture loader may need updates for NeuroBench compatibility

---

## 7. Next Priority: NeuroBench + BPTT Baseline

**Goal**: Establish benchmarking infrastructure with:
- BPTT as baseline algorithm
- NeuroBench metrics integration
- Selected benchmark datasets
- Comprehensive timing/power metrics

---

## 8. Environment

- **Python**: 3.9+
- **Container**: Singularity (`.sif` files available)
- **Hardware targets**: CPU, CUDA GPU, Apple MPS
- **Code style**: Black (88 chars), flake8, isort

