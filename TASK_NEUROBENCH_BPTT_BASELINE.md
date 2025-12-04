# Task Prompt: NeuroBench Integration + BPTT Baseline Benchmarking

## Role

Act as a **senior neuromorphic computing engineer** implementing a standardized benchmarking system for SNN learning algorithms, **maximizing reuse of existing libraries**.

---

## Core Principle: REUSE EVERYTHING

> **DO NOT reimplement what already exists in NeuroBench, snnTorch, or PyTorch.**

| Functionality | Use This | NOT This |
|--------------|----------|----------|
| BPTT Training | `snntorch.backprop.BPTT` | Custom training loop |
| Loss Functions | `snntorch.functional.*` | Custom loss implementations |
| Timing | `torch.profiler` / `torch.utils.benchmark` | Custom timers |
| GPU Time | `torch.cuda.Event` | Manual timestamps |
| SNN Metrics | `neurobench.metrics.*` | Custom metric calculations |
| Model Wrapping | `neurobench.models.SNNTorchModel` | Custom wrappers |
| Benchmarking | `neurobench.benchmarks.Benchmark` | Custom benchmark loops |

---

## Part 1: What Each Library Provides

### 1.1 snnTorch (Training)

```python
# BPTT - Use directly, don't reimplement
from snntorch import backprop
loss = backprop.BPTT(net, data, targets, num_steps, optimizer, criterion)

# Loss functions - Use directly
import snntorch.functional as SF
loss_fn = SF.mse_count_loss()      # MSE on spike counts
loss_fn = SF.ce_count_loss()       # Cross-entropy on spike counts  
loss_fn = SF.ce_rate_loss()        # Cross-entropy on spike rates
loss_fn = SF.mse_membrane_loss()   # MSE on membrane potential

# Surrogate gradients - Built into neurons
import snntorch as snn
lif = snn.Leaky(beta=0.9, spike_grad=snn.surrogate.fast_sigmoid())

# Spike encoding - Use directly
from snntorch import spikegen
spikes = spikegen.rate(data, num_steps=25)
spikes = spikegen.latency(data, num_steps=25)
```

### 1.2 NeuroBench (Benchmarking)

```python
# Model wrapper - Use directly
from neurobench.models import SNNTorchModel
wrapped_model = SNNTorchModel(my_snntorch_model)

# Metrics - Use directly, don't reimplement
from neurobench.metrics.static import (
    FootprintMemory,      # Model memory in bytes
    FootprintMACs,        # Multiply-accumulate operations
    ConnectionSparsity,   # Weight sparsity
)
from neurobench.metrics.workload import (
    ActivationSparsity,   # Spike sparsity during inference
    SynapticOperations,   # Total synaptic ops (AC + MAC)
    ClassificationAccuracy,
    MSE,
)

# Benchmark harness - Use directly
from neurobench.benchmarks import Benchmark
benchmark = Benchmark(
    model=wrapped_model,
    dataloader=test_loader,
    static_metrics=[FootprintMemory(), ConnectionSparsity()],
    workload_metrics=[ActivationSparsity(), SynapticOperations(), ClassificationAccuracy()],
)
results = benchmark.run()
```

### 1.3 PyTorch (Profiling)

```python
# Timing - Use built-in profiler
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Training code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))

# GPU timing - Use CUDA events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
# ... code ...
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)

# Simple benchmarking - Use torch.utils.benchmark
from torch.utils.benchmark import Timer
timer = Timer(stmt="model(x)", globals={"model": model, "x": x})
result = timer.blocked_autorange()
print(f"Mean: {result.mean * 1e3:.2f} ms")
```

---

## Part 2: Minimal Implementation Required

### 2.1 BPTT Trainer (Thin Wrapper Only)

The only custom code needed is a thin wrapper to match your `BaseTrainer` interface:

```python
# src/trainers/bptt_trainer.py
"""BPTT trainer wrapping snnTorch's built-in backprop.BPTT."""

from snntorch import backprop
import snntorch.functional as SF
import torch

from trainers.base_trainer import BaseTrainer


class BPTTTrainer(BaseTrainer):
    """
    BPTT using snnTorch's built-in implementation.
    
    This is a thin wrapper around snntorch.backprop.BPTT to match
    the BaseTrainer interface for fair algorithm comparison.
    """

    def __init__(
        self,
        network,
        lr: float,
        batch_size: int,
        loss_type: str = "ce_rate",
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.lr = lr
        self.batch_size = batch_size
        
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        
        # Use snnTorch's built-in loss functions
        loss_functions = {
            "mse_count": SF.mse_count_loss(),
            "ce_count": SF.ce_count_loss(),
            "ce_rate": SF.ce_rate_loss(),
        }
        self.loss_fn = loss_functions.get(loss_type, SF.ce_rate_loss())

    def train_sample(self, data: torch.Tensor, target: torch.Tensor):
        """Use snnTorch's BPTT directly."""
        num_steps = data.shape[0]
        
        # snnTorch handles everything: forward, backward, optimizer step
        loss = backprop.BPTT(
            self.network, data, target, num_steps,
            self.optimizer, self.loss_fn
        )
        
        # Get predictions (separate forward for metrics)
        with torch.no_grad():
            self.network.reset()
            spk_sum = sum(self.network(data[t])[0][-1] for t in range(num_steps))
            pred = spk_sum.argmax(dim=1, keepdim=True)
        
        return loss, pred

    def reset(self):
        self.network.reset()
```

### 2.2 NeuroBench Evaluation (Direct Library Use)

```python
# src/utils/neurobench_eval.py
"""NeuroBench integration - just wrapping their API."""

from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark
from neurobench.metrics.static import FootprintMemory, ConnectionSparsity
from neurobench.metrics.workload import (
    ActivationSparsity, 
    SynapticOperations, 
    ClassificationAccuracy,
)


def run_neurobench(network, test_loader, device="cpu"):
    """
    Run NeuroBench evaluation using their built-in harness.
    
    Returns all metrics computed by NeuroBench directly.
    """
    network.to(device).eval()
    
    # Wrap with NeuroBench's SNNTorchModel
    wrapped = SNNTorchModel(network)
    
    # Use NeuroBench's Benchmark harness directly
    benchmark = Benchmark(
        model=wrapped,
        dataloader=test_loader,
        static_metrics=[FootprintMemory(), ConnectionSparsity()],
        workload_metrics=[
            ActivationSparsity(),
            SynapticOperations(),
            ClassificationAccuracy(),
        ],
    )
    
    return benchmark.run()
```

### 2.3 Timing with PyTorch Profiler

```python
# src/utils/benchmark_metrics.py
"""Benchmarking metrics using PyTorch's built-in profiler."""

import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.benchmark import Timer
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int
    accuracy: float
    loss: float
    wall_time_ms: float
    cpu_time_ms: float
    cuda_time_ms: Optional[float]
    memory_mb: Optional[float]


class ProfiledTrainer:
    """
    Wrapper that adds PyTorch profiling to any trainer.
    
    Uses torch.profiler - don't reimplement timing!
    """
    
    def __init__(self, trainer, use_cuda=True):
        self.trainer = trainer
        self.use_cuda = use_cuda and torch.cuda.is_available()
    
    def train_epoch_profiled(self, train_loader, device):
        """Train one epoch with PyTorch profiling."""
        
        activities = [ProfilerActivity.CPU]
        if self.use_cuda:
            activities.append(ProfilerActivity.CUDA)
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            # Your existing training loop
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            for data, target in train_loader:
                data = data.transpose(0, 1).to(device)
                target = target.to(device)
                
                self.trainer.reset()
                loss, pred = self.trainer.train_sample(data, target)
                
                total_loss += loss.item() * data.size(1)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(1)
        
        # Extract timing from profiler
        key_avg = prof.key_averages()
        cpu_time = sum(e.cpu_time_total for e in key_avg) / 1000  # ms
        cuda_time = sum(e.cuda_time_total for e in key_avg) / 1000 if self.use_cuda else None
        
        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "cpu_time_ms": cpu_time,
            "cuda_time_ms": cuda_time,
            "profiler": prof,  # Full profiler for detailed analysis
        }
```

---

## Part 3: Complete Benchmark Runner

```python
# src/benchmark_runner.py
"""
Benchmark runner using NeuroBench, snnTorch, and PyTorch profiler.

Minimal custom code - maximum library reuse.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Type

import torch
from torch.profiler import profile, ProfilerActivity

from trainers.base_trainer import BaseTrainer
from trainers.bptt_trainer import BPTTTrainer
from trainers.stsf_trainer import STSFTrainer
from networks.fc_network import FCNetwork
from datasets.get_loader import get_loader
from utils.neurobench_eval import run_neurobench


# Algorithm metadata (not computed, just documented)
ALGORITHM_INFO = {
    "bptt": {
        "name": "Backpropagation Through Time",
        "is_local": False,
        "requires_backprop": True,
        "source": "snntorch.backprop.BPTT",
    },
    "stsf": {
        "name": "Spiking Time Sparse Feedback", 
        "is_local": True,
        "requires_backprop": False,
        "source": "custom (stsf_trainer.py)",
    },
}


@dataclass
class BenchmarkResult:
    """Results from one benchmark run."""
    algorithm: str
    dataset: str
    architecture: List[int]
    
    # Training results
    final_accuracy: float
    final_loss: float
    epochs_trained: int
    
    # Timing (from PyTorch profiler)
    total_wall_time_s: float
    avg_epoch_cpu_ms: float
    avg_epoch_cuda_ms: float
    
    # NeuroBench metrics (from their harness)
    neurobench: Dict
    
    # Algorithm info
    algorithm_info: Dict


def benchmark_algorithm(
    algorithm_name: str,
    trainer_class: Type[BaseTrainer],
    dataset: str,
    layer_sizes: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    timesteps: int,
    checkpoint_epochs: List[int],
    device: str,
) -> BenchmarkResult:
    """
    Run benchmark for a single algorithm.
    
    Uses:
    - snnTorch BPTT (for bptt trainer)
    - PyTorch profiler (for timing)
    - NeuroBench (for SNN metrics)
    """
    
    # Setup
    train_loader, test_loader = get_loader(dataset, batch_size, timesteps)
    network = FCNetwork(layer_sizes=layer_sizes, beta=0.9)
    trainer = trainer_class(network=network, lr=lr, batch_size=batch_size)
    trainer = trainer.to(device)
    
    # Training with PyTorch profiler
    epoch_times_cpu = []
    epoch_times_cuda = []
    accuracies = {}
    
    import time
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        # Profile this epoch
        activities = [ProfilerActivity.CPU]
        if device != "cpu":
            activities.append(ProfilerActivity.CUDA)
        
        with profile(activities=activities) as prof:
            train_one_epoch(trainer, train_loader, device)
        
        # Extract timing from profiler
        key_avg = prof.key_averages()
        cpu_ms = sum(e.cpu_time_total for e in key_avg) / 1000
        cuda_ms = sum(e.cuda_time_total for e in key_avg) / 1000 if device != "cpu" else 0
        epoch_times_cpu.append(cpu_ms)
        epoch_times_cuda.append(cuda_ms)
        
        # Evaluate at checkpoints
        if (epoch + 1) in checkpoint_epochs:
            acc = evaluate(network, test_loader, device)
            accuracies[epoch + 1] = acc
            print(f"  Epoch {epoch+1}: accuracy={acc:.4f}")
    
    total_time = time.perf_counter() - start_time
    
    # Final evaluation
    final_acc = evaluate(network, test_loader, device)
    
    # NeuroBench evaluation (uses their harness directly)
    neurobench_results = run_neurobench(network, test_loader, device)
    
    return BenchmarkResult(
        algorithm=algorithm_name,
        dataset=dataset,
        architecture=layer_sizes,
        final_accuracy=final_acc,
        final_loss=0.0,  # TODO: track
        epochs_trained=epochs,
        total_wall_time_s=total_time,
        avg_epoch_cpu_ms=sum(epoch_times_cpu) / len(epoch_times_cpu),
        avg_epoch_cuda_ms=sum(epoch_times_cuda) / len(epoch_times_cuda),
        neurobench=neurobench_results,
        algorithm_info=ALGORITHM_INFO.get(algorithm_name, {}),
    )


def train_one_epoch(trainer, train_loader, device):
    """Simple training loop."""
    trainer.network.train()
    for data, target in train_loader:
        data = data.transpose(0, 1).to(device)
        target = target.to(device)
        trainer.reset()
        trainer.train_sample(data, target)


def evaluate(network, test_loader, device):
    """Simple evaluation."""
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.transpose(0, 1).to(device)
            target = target.to(device)
            network.reset()
            spk_sum = sum(network(data[t])[0][-1] for t in range(data.size(0)))
            correct += (spk_sum.argmax(1) == target).sum().item()
            total += target.size(0)
    return correct / total


def run_comparison(config: dict, output_dir: Path):
    """Run comparison benchmark."""
    algorithms = {
        "bptt": BPTTTrainer,
        "stsf": STSFTrainer,
    }
    
    results = {}
    for algo_name in config["algorithms"]:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {algo_name.upper()}")
        print(f"{'='*60}")
        
        result = benchmark_algorithm(
            algorithm_name=algo_name,
            trainer_class=algorithms[algo_name],
            dataset=config["dataset"],
            layer_sizes=config["layer_sizes"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            timesteps=config["timesteps"],
            checkpoint_epochs=config["checkpoint_epochs"],
            device=config["device"],
        )
        results[algo_name] = result
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2, default=str)
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Algorithm':<12} | {'Accuracy':<10} | {'Time (s)':<10} | {'Local':<6}")
    print("-"*80)
    for name, r in results.items():
        is_local = r.algorithm_info.get("is_local", "?")
        print(f"{name:<12} | {r.final_accuracy:<10.4f} | {r.total_wall_time_s:<10.1f} | {is_local}")
    
    return results
```

---

## Part 4: Configuration

```yaml
# configs/benchmark_comparison.yaml

# Algorithms to compare
algorithms:
  - bptt
  - stsf

# Dataset
dataset: "MNIST"
timesteps: 25

# Architecture (same for all algorithms - fair comparison)
layer_sizes: [784, 256, 10]

# Training
epochs: 50
batch_size: 128
lr: 0.001

# When to record accuracy
checkpoint_epochs: [1, 5, 10, 25, 50]

# Hardware  
device: "cuda"  # or "cpu"

# Output
output_dir: "./benchmark_results"
```

---

## Part 5: Deliverables (Minimal)

### Files to Create

Only **3 small files** needed:

- [ ] `src/trainers/bptt_trainer.py` — ~40 lines wrapping `snntorch.backprop.BPTT`
- [ ] `src/utils/neurobench_eval.py` — ~30 lines wrapping NeuroBench harness
- [ ] `src/benchmark_runner.py` — ~150 lines orchestration

### Files to Modify

- [ ] `main.py` — Uncomment line 205 to register BPTTTrainer
- [ ] `configs/` — Add benchmark config

### What We DON'T Need to Implement

| Feature | Why Not Needed |
|---------|----------------|
| Custom loss functions | Use `snntorch.functional.*` |
| Custom BPTT algorithm | Use `snntorch.backprop.BPTT` |
| Custom timing code | Use `torch.profiler` |
| Custom SNN metrics | Use `neurobench.metrics.*` |
| Custom benchmark harness | Use `neurobench.benchmarks.Benchmark` |
| Custom model wrapper | Use `neurobench.models.SNNTorchModel` |

---

## Part 6: Validation

1. **BPTT works**: `snntorch.backprop.BPTT` runs without errors
2. **NeuroBench integrates**: `SNNTorchModel` wraps FCNetwork correctly
3. **Profiler works**: `torch.profiler` captures CPU/CUDA time
4. **Comparison runs**: Both STSF and BPTT produce results

---

## Questions Before Implementation

1. **Network compatibility**: Does `FCNetwork` work with `snntorch.backprop.BPTT`? (May need `init_hidden=True` on neurons)

2. **NeuroBench version**: API may differ between versions. Should I check 2.1.0 compatibility?

3. **Output format**: JSON sufficient, or also CSV/TensorBoard?

---

## Execution Order

1. **Test snnTorch BPTT** with existing FCNetwork
2. **Test NeuroBench wrapper** on trained network
3. **Create thin BPTTTrainer** wrapper
4. **Create benchmark_runner.py** using all libraries
5. **Run comparison**: BPTT vs STSF on MNIST
