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

## Supported Components

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
| DVS-CIFAR10 | Event-based | 🟢 |  |


### Networks

| Architecture | Type | Location | Status | Notes |
|-------------|------|----------|--------|-------|
| FC-SNN | Feedforward | `benchmarking/fc_snn.py` | 🟢 | Fully-connected, baseline for rate-coded data |
| Conv-SNN | Convolutional | `benchmarking/conv_snn.py` | 🟢 | Convolutional layers, for image tasks |
| R-SNN | Recurrent | `benchmarking/r_snn.py` | 🟢 | Recurrent SNN, for temporal tasks |
| VGG9 (CIFAR-10) | Convolutional | `benchmarking/vgg9_cifar10.py` | 🟢 | VGG-inspired, tuned for CIFAR-10 |
| VGG9 (DVSGesture) | Convolutional | `benchmarking/vgg9_dvsgest.py` | 🟢 | VGG-inspired, for event-based gestures |
| VGG9 (SVHN) | Convolutional | `benchmarking/vgg9_svhn.py` | 🟢 | VGG-inspired, for street view data |

### Trainers

| Algorithm | Trainer File | Status | Supported Networks | Notes |
|-----------|--------------|--------|--------------------|-------|
| BPTT | `bptt_trainer.py` | 🟢 | All | |
| OSTL | `ostl_trainer.py` | 🟢 | Recurrent |  |
| OTTT | `ottt_trainer.py` | 🔴 | Convolutional | |
| OSTTP | `osttp_trainer.py` | 🟢 | Recurrent (SHD-tuned) | |
| OTPE | `otpe_trainer.py` | 🔴 | All | |
| DECOLLE | `decolle_trainer.py` | 🔴 | All | |
| E-prop | `eprop_trainer.py` | 🟢 | Recurrent | |
| ESD-RTRL | `esd_rtrl_trainer.py` | 🟢 | Recurrent | |
| ETLP | `etlp_trainer.py` | 🟢 | All | |
| STOP | `stop_trainer.py` | 🔴 | All | |
| TP | `tp_trainer.py` | 🔴 | All | **batch_size ≥ 2 required** (contrastive loss degenerates to zero gradient with B=1). **Use `out_integrator=True`** on FCSNN/RSNN for faithful eval: the algorithm trains a pure integrator at the output; with `out_integrator=True` eval uses `mem_rec[-1]` at the final timestep. VGG9 already has a correct LI head. |
| ELL | `ell_trainer.py` | 🔴 | Feedforward | |
| FELL | `fell_trainer.py` | 🔴 | Feedforward | |
| BELL | `bell_trainer.py` | 🔴 | Feedforward | |
| STLLR | `stllr_trainer.py` | 🔴 | All | |
| STSF | `stsf_trainer.py` | 🔴 | All | |

### Test Results

Tests on VGG9 network:

| Cofig File | Trainer | Dataset | Epoch | Train Accuracy | Test Accuracy | Commit | Comment |
|---------|-------|---------------|----------|------|--------|--------|------------|
| `ottt_vgg9_svhn.yaml` | OTTT | SVHN | 70 | 0.80 | 0.78 | 07ef17df78dee26957e0bb18cf7492b9471bcf34 | Growing well but overshooting with accuracy drops every once in a while. Reached peaks of 0.92 test accuracy |
| `ottt_vgg9_cifar10.yaml` | OTTT | CIFAR10 | 70 | 0.42 | 0.42 | 07ef17df78dee26957e0bb18cf7492b9471bcf34 | Growing well but overshooting with accuracy drops every once in a while. Still not high enough. |
| `tp_vgg9_dvsgesture.yaml` | OTTT | DVSGesture | 70 | 0.85 | 0.80 | 07ef17df78dee26957e0bb18cf7492b9471bcf34 | Growing well but overshooting with accuracy drops every once in a while. Reached peaks of 0.88 test accuracy |
| `tp_vgg9_dvscifar10.yaml` | OTTT | DVSCIFAR10 | 70 | 0.40 | 0.23 | 07ef17df78dee26957e0bb18cf7492b9471bcf34 | It seems like there is overfitting here |



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

