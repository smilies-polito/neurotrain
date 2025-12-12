"""
Benchmark runner for comparing SNN learning algorithms.

Orchestrates algorithm comparison using:
- snnTorch for training (BPTT uses snntorch.functional losses)
- time.perf_counter() for lightweight timing (with CUDA sync for GPU)
- NeuroBench for SNN-specific metrics

Usage:
    python src/benchmark_runner.py --config configs/benchmark_comparison.yaml
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Type, Optional

import torch
import yaml
# Note: We use simple time.perf_counter() instead of torch.profiler
# to avoid the ~10x overhead that the profiler adds

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainers.base_trainer import BaseTrainer
from trainers.bptt_trainer import BPTTTrainer
from trainers.stsf_trainer import STSFTrainer
from trainers.eprop_trainer import EpropTrainer
from networks.fc_network import FCNetwork
from datasets.get_loader import get_loader
from utils.neurobench_eval import run_neurobench


# Algorithm metadata (not computed, just documented)
ALGORITHM_INFO = {
    "bptt": {
        "name": "Backpropagation Through Time",
        "is_local": False,
        "requires_backprop": True,
        "source": "snntorch.functional (surrogate gradients)",
    },
    "stsf": {
        "name": "Spiking Time Sparse Feedback",
        "is_local": True,
        "requires_backprop": False,
        "source": "custom (stsf_trainer.py)",
    },
    "eprop": {
        "name": "Eligibility Propagation",
        "is_local": True,
        "requires_backprop": False,
        "source": "custom (eprop_trainer.py) - Bellec et al. 2020",
    },
}


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    accuracy: float
    loss: float
    cpu_time_ms: float
    cuda_time_ms: Optional[float] = None


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
    
    # Accuracy at checkpoints
    checkpoint_accuracies: Dict[int, float] = field(default_factory=dict)
    
    # Timing (from PyTorch profiler)
    total_wall_time_s: float = 0.0
    avg_epoch_cpu_ms: float = 0.0
    avg_epoch_cuda_ms: float = 0.0
    
    # NeuroBench metrics (from their harness)
    neurobench: Dict[str, Any] = field(default_factory=dict)
    
    # Algorithm info
    algorithm_info: Dict[str, Any] = field(default_factory=dict)


def train_one_epoch(
    trainer: BaseTrainer,
    train_loader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        trainer: Trainer instance (BPTT or STSF)
        train_loader: Training data loader
        device: Torch device
        
    Returns:
        Dictionary with loss and accuracy
    """
    trainer.network.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for data, target in train_loader:
        # Data shape: [batch, features] -> transpose to [timesteps, batch, features]
        data = data.transpose(0, 1).to(device)
        target = target.to(device)
        batch_size = data.size(1)
        
        trainer.reset()
        loss, pred = trainer.train_sample(data, target)
        
        total_loss += loss.item() * batch_size
        total_correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += batch_size
    
    return {
        "loss": total_loss / total_samples if total_samples > 0 else 0.0,
        "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
    }


@torch.no_grad()
def evaluate(
    network: torch.nn.Module,
    test_loader,
    device: torch.device,
) -> float:
    """
    Evaluate network accuracy.
    
    Args:
        network: Network to evaluate
        test_loader: Test data loader
        device: Torch device
        
    Returns:
        Test accuracy as float
    """
    network.eval()
    correct = 0
    total = 0
    
    for data, target in test_loader:
        data = data.transpose(0, 1).to(device)
        target = target.to(device)
        
        network.reset()
        spk_sum = None
        for t in range(data.size(0)):
            spk, _ = network(data[t])
            if spk_sum is None:
                spk_sum = spk[-1]
            else:
                spk_sum = spk_sum + spk[-1]
        
        preds = spk_sum.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += target.size(0)
    
    return correct / total if total > 0 else 0.0


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
    beta: float = 0.9,
) -> BenchmarkResult:
    """
    Run benchmark for a single algorithm.
    
    Uses:
    - snnTorch loss functions (for BPTT trainer)
    - torch.profiler for timing
    - NeuroBench for SNN metrics
    
    Args:
        algorithm_name: Name of algorithm ("bptt", "stsf")
        trainer_class: Trainer class to instantiate
        dataset: Dataset name
        layer_sizes: Network architecture
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        timesteps: Number of timesteps for spike encoding
        checkpoint_epochs: Epochs at which to record accuracy
        device: Device string ("cpu", "cuda")
        beta: LIF neuron beta parameter
        
    Returns:
        BenchmarkResult with all metrics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {algorithm_name.upper()}")
    print(f"{'='*60}")
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    use_cuda = device.type == "cuda"
    
    # Get data loaders
    train_loader, test_loader = get_loader(dataset, batch_size, timesteps)
    
    # Create network
    network = FCNetwork(layer_sizes=layer_sizes, beta=beta)
    
    # Create trainer with appropriate settings
    if algorithm_name == "bptt":
        # BPTT uses gradient-based training
        torch.set_grad_enabled(True)
        trainer = trainer_class(
            network=network,
            lr=lr,
            batch_size=batch_size,
        )
    elif algorithm_name == "eprop":
        # E-prop uses local learning rules (no autograd)
        torch.set_grad_enabled(False)
        trainer = trainer_class(
            network=network,
            lr=lr,
            batch_size=batch_size,
            use_optimizer=False,
            optimizer=None,
        )
    else:
        # STSF uses local learning
        torch.set_grad_enabled(False)
        trainer = trainer_class(
            network=network,
            lr=lr,
            batch_size=batch_size,
            quant=False,
            use_optimizer=False,
            optimizer=None,
        )
    
    # Move to device
    trainer = trainer.to(device)
    
    # Training with lightweight timing
    # NOTE: We use simple time.perf_counter() for per-epoch timing instead of
    # torch.profiler (which has ~10x overhead). Profiler is only used optionally
    # on the last epoch if detailed breakdown is needed.
    epoch_times_ms = []
    checkpoint_accuracies = {}
    final_loss = 0.0
    
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        # Simple wall-clock timing per epoch
        epoch_start = time.perf_counter()
        
        # Synchronize before timing if using CUDA
        if use_cuda:
            torch.cuda.synchronize()
        
        metrics = train_one_epoch(trainer, train_loader, device)
        
        # Synchronize after to ensure all GPU ops are complete
        if use_cuda:
            torch.cuda.synchronize()
        
        epoch_end = time.perf_counter()
        epoch_ms = (epoch_end - epoch_start) * 1000.0
        epoch_times_ms.append(epoch_ms)
        
        final_loss = metrics["loss"]
        
        # Evaluate at checkpoints
        if (epoch + 1) in checkpoint_epochs:
            acc = evaluate(network, test_loader, device)
            checkpoint_accuracies[epoch + 1] = acc
            print(f"  Epoch {epoch + 1}: accuracy={acc:.4f}, loss={final_loss:.4f}, time={epoch_ms:.0f}ms")
    
    total_time = time.perf_counter() - start_time
    avg_epoch_ms = sum(epoch_times_ms) / len(epoch_times_ms) if epoch_times_ms else 0.0
    
    # Final evaluation
    final_accuracy = evaluate(network, test_loader, device)
    print(f"\n  Final accuracy: {final_accuracy:.4f}")
    print(f"  Avg epoch time: {avg_epoch_ms:.0f}ms")
    
    # NeuroBench evaluation
    print("  Running NeuroBench evaluation...")
    try:
        neurobench_results = run_neurobench(
            network, 
            test_loader, 
            device=str(device),
            num_timesteps=timesteps,
        )
        # Convert any non-serializable values
        neurobench_results = _make_serializable(neurobench_results)
    except Exception as e:
        print(f"  Warning: NeuroBench evaluation failed: {e}")
        neurobench_results = {"error": str(e)}
    
    return BenchmarkResult(
        algorithm=algorithm_name,
        dataset=dataset,
        architecture=layer_sizes,
        final_accuracy=final_accuracy,
        final_loss=final_loss,
        epochs_trained=epochs,
        checkpoint_accuracies=checkpoint_accuracies,
        total_wall_time_s=total_time,
        avg_epoch_cpu_ms=avg_epoch_ms,  # Now using simple wall-clock time
        avg_epoch_cuda_ms=avg_epoch_ms if use_cuda else 0.0,  # Same value if CUDA
        neurobench=neurobench_results,
        algorithm_info=ALGORITHM_INFO.get(algorithm_name, {}),
    )


def _make_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def run_comparison(config: Dict[str, Any], output_dir: Path) -> Dict[str, BenchmarkResult]:
    """
    Run comparison benchmark across multiple algorithms.
    
    Args:
        config: Configuration dictionary with benchmark settings
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping algorithm names to BenchmarkResult
    """
    # Available trainers
    trainers = {
        "bptt": BPTTTrainer,
        "stsf": STSFTrainer,
        "eprop": EpropTrainer,
    }
    
    results = {}
    
    for algo_name in config["algorithms"]:
        if algo_name not in trainers:
            print(f"Warning: Unknown algorithm '{algo_name}', skipping")
            continue
        
        result = benchmark_algorithm(
            algorithm_name=algo_name,
            trainer_class=trainers[algo_name],
            dataset=config["dataset"],
            layer_sizes=config["layer_sizes"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            timesteps=config["timesteps"],
            checkpoint_epochs=config["checkpoint_epochs"],
            device=config["device"],
            beta=config.get("beta", 0.9),
        )
        results[algo_name] = result
    
    # Save results to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "benchmark_results.json"
    
    with open(output_file, "w") as f:
        json.dump(
            {k: asdict(v) for k, v in results.items()},
            f,
            indent=2,
            default=str,
        )
    
    print(f"\nResults saved to: {output_file}")
    
    # Print comparison summary
    print_comparison_summary(results)
    
    return results


def print_comparison_summary(results: Dict[str, BenchmarkResult]) -> None:
    """Print a formatted comparison summary."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    header = f"{'Algorithm':<12} | {'Accuracy':<10} | {'Wall Time (s)':<13} | {'Local':<6} | {'Backprop':<8}"
    print(header)
    print("-" * 80)
    
    for name, r in results.items():
        is_local = r.algorithm_info.get("is_local", "?")
        requires_bp = r.algorithm_info.get("requires_backprop", "?")
        print(f"{name:<12} | {r.final_accuracy:<10.4f} | {r.total_wall_time_s:<13.2f} | {str(is_local):<6} | {str(requires_bp):<8}")
    
    print("=" * 80)
    
    # Print checkpoint accuracy comparison
    if len(results) > 1:
        print("\nCheckpoint Accuracy Comparison:")
        print("-" * 60)
        
        # Get all checkpoint epochs
        all_epochs = set()
        for r in results.values():
            all_epochs.update(r.checkpoint_accuracies.keys())
        
        for epoch in sorted(all_epochs):
            line = f"  Epoch {epoch:>3}: "
            for name, r in results.items():
                acc = r.checkpoint_accuracies.get(epoch, None)
                if acc is not None:
                    line += f"{name}={acc:.4f}  "
            print(line)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run SNN learning algorithm benchmarks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_comparison.yaml",
        help="Path to benchmark configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine output directory
    output_dir = Path(args.output_dir if args.output_dir else config.get("output_dir", "./benchmark_results"))
    
    print("\n" + "=" * 60)
    print("SNN LEARNING ALGORITHM BENCHMARK")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Algorithms: {config['algorithms']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Architecture: {config['layer_sizes']}")
    print(f"Epochs: {config['epochs']}")
    print("=" * 60)
    
    # Run comparison
    run_comparison(config, output_dir)


if __name__ == "__main__":
    main()

