# SNN Training Benchmarking

This repository contains code for benchmarking the training of Spiking Neural Networks (SNNs). The code is structured in a modular way, clearly separating network architectures and training algorithms which can be mixed and matched to evaluate different combinations of architectures and training methods.

## Repository Structure

- `📂 configs/`: Contains configuration files for different experiments.
- `📂 docs/`: Contains documentation for the framework.
- `📂 hpc/`: Contains scripts for running benchmarks on High-Performance Computing (HPC) clusters.
- `📂 src/`: Contains the source code of the framework.
- `📂 test/`: Test programs used to test different components.
  - `📂 src/networks/`: Contains different SNN architectures.
  - `📂 src/trainers/`: Contains different trainer objects that implement various training algorithms.
  - `📂 src/datasets/`: Contains code for loading and preprocessing datasets.
  - `📂 src/utils/`: Contains utility functions for training and evaluation.
- `Makefile`: Contains commands for setting up the environment and running benchmarks.
- `main.py`: The main entry point for running single experiments.
- `benchmarking.py`: The main entry point for running benchmarking experiments.
- `reproducibility.py`: Script to launch reproducibility experiments on implemented trainers.

## Unit Tests & Validation

### Dataloaders

| Dataset | Type | Status | Notes |
|---------|------|--------|-------|
| MNIST | Rate-coded | 🟢 |  |
| Fashion-MNIST | Rate-coded | 🟢 |  |
| CIFAR-10 | Rate-coded | 🟢 |  |
| SVHN | Rate-coded | 🟢 |  |
| NMNIST | Event-based | 🟢 |  |
| DVSGesture | Event-based | 🟢 | Works with caching |
| SHD | Event-based | 🟡 | Noticed a few issues that seems related to dataset itself, need to check them |
| DVS-CIFAR10 | Event-based | 🔴 | Not implemented yet |


### Networks

| Architecture | Type | Location | Status | Notes |
|-------------|------|----------|--------|-------|
| FC-SNN | Feedforward | `benchmarking/fc_snn.py` | 🟢 | Fully-connected, baseline for rate-coded data |
| Conv-SNN | Convolutional | `benchmarking/conv_snn.py` | 🟢 | Convolutional layers, for image tasks |
| R-SNN | Recurrent | `benchmarking/r_snn.py` | 🟢 | Recurrent SNN, for temporal tasks |
| VGG9 (CIFAR-10) | Convolutional | `benchmarking/vgg9_cifar10.py` | 🟢 | VGG-inspired, tuned for CIFAR-10 |
| VGG9 (DVSGesture) | Convolutional | `benchmarking/vgg9_dvsgest.py` | 🟢 | VGG-inspired, for event-based gestures |
| VGG9 (SVHN) | Convolutional | `benchmarking/vgg9_svhn.py` | 🟢 | VGG-inspired, for street view data |
| Paper-specific | Various | `reproducibility/` | 🟢 | Algorithm-specific implementations (DECOLLE, DRTP, E-prop, etc.) |

### Trainers

| Algorithm | Trainer File | Status | Supported Networks |
|-----------|--------------|--------|-------------------|
| BPTT | `bptt_trainer.py` | 🟢 | All |
| OSTL | `ostl_trainer.py` | 🟢 | Recurrent |
| OTTT | `ottt_trainer.py` | 🔴 | Convolutional |
| OSTTP | `osttp_trainer.py` | 🟢 | Recurrent (SHD-tuned) |
| OTPE | `otpe_trainer.py` | 🔴 | All |
| DRTP | `drtp_trainer.py` | 🔴 | All |
| DECOLLE | `decolle_trainer.py` | 🔴 | All |
| E-prop | `eprop_trainer.py` | 🟢 | Recurrent |
| ESD-RTRL | `esd_rtrl_trainer.py` | 🟢 | Recurrent |
| ETLP | `etlp_trainer.py` | 🔴 | All |
| STOP | `stop_trainer.py` | 🔴 | All |
| TP | `tp_trainer.py` | 🔴 | All |
| ELL | `ell_trainer.py` | 🔴 | Feedforward |
| FELL | `fell_trainer.py` | 🔴 | Feedforward |
| BELL | `bell_trainer.py` | 🔴 | Feedforward |
| STLLR | `stllr_trainer.py` | 🔴 | All |
| STSF | `stsf_trainer.py` | 🔴 | All |

### Test Programs

**Integration & Trainer Validation** — Full training loops testing trainer × network × dataset combinations:

##### BPTT
| Test | Network | Dataset | Result (Epoch: train/test) | Commit |
|------|---------|---------|--------|--------|
| `bptt_mnist_fc.py`      | FC-SNN    | MNIST         |  |  |
| `bptt_fmnist_conv.py`   | Conv-SNN  | Fashion-MNIST |  |  |
| `bptt_cifar10_vgg9.py`  | VGG9      | CIFAR-10      | 10: 0.8685/0.7212                   | bcae958cdc4487399a3f08c02615d87d7faca6d0 |
| `bptt_dvsgest_vgg9.py`  | VGG9      | DVSGesture    | 10: 0.9179/0.8864 40: 1.0/0.9356    | bcae958cdc4487399a3f08c02615d87d7faca6d0 |
| `bptt_dvsgest_r.py`     | R-SNN     | DVSGesture    | |  |
| `bptt_svhn_vgg9.py`     | VGG9      | SVHN          |  |  |
| `bptt_svhn_r.py`        | R-SNN     | SVHN          |  |  |
| `bptt_nmnist_r.py`      | R-SNN     | NMNIST        |  |  |

##### OSTL
| Test | Network | Dataset | Result (Epoch: train/test) | Commit |
|------|---------|---------|--------|--------|
| `ostl_mnist_fc.py`      | FC-SNN    | MNIST         | 10: 0.9960/0.9753 25: 0.9984/0.9765 | 71a774acd6e601a527e119f4b9d3f2c2b48b44da |
| `ostl_nmnist_r.py`      | R-SNN     | NMNIST        | 10: 0.9137/0.8961                   | 71a774acd6e601a527e119f4b9d3f2c2b48b44da |

##### ETLP
| Test | Network | Dataset | Result (Epoch: train/test) | Commit |
|------|---------|---------|--------|--------|
| `etlp_mnist_fc.py`      | FC-SNN    | MNIST         | 10: 0.8247/0.8321 100: 0.8512/0.8507 | |
| `etlp_nmnist_r.py`      | FC-SNN    | MNIST         | 2: 0.3636/0.4436 | |

**Dataset Smoke Tests** — Minimal tests verifying dataloaders work correctly:

| Test | Dataset | Command |
|------|---------|---------|
| `test_mnist_loader.py` | MNIST | `python tests/dataloaders/test_mnist_loader.py` |
| `test_fashionmnist_loader.py` | Fashion-MNIST | `python tests/dataloaders/test_fashionmnist_loader.py` |
| `test_cifar10_loader.py` | CIFAR-10 | `python tests/dataloaders/test_cifar10_loader.py` |
| `test_svhn_loader.py` | SVHN | `python tests/dataloaders/test_svhn_loader.py` |
| `test_nmnist_loader.py` | NMNIST | `python tests/dataloaders/test_nmnist_loader.py` |
| `test_dvsgesture_loader.py` | DVSGesture | `python tests/dataloaders/test_dvsgesture_loader.py` |
| `test_shd_loader.py` | SHD | `python tests/dataloaders/test_shd_loader.py` |

