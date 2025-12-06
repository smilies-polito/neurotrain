#!/usr/bin/env python3
"""
Run BPTT vs STSF benchmarks across all available datasets.

Usage:
    python run_all_benchmarks.py [--epochs 50] [--device cuda]
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from benchmark_runner import benchmark_algorithm, print_comparison_summary, BenchmarkResult
from trainers.bptt_trainer import BPTTTrainer
from trainers.stsf_trainer import STSFTrainer


# Dataset configurations: dataset_name -> (input_size, num_classes, layer_sizes)
# ============================================================================
# STANDARD IMAGE CLASSIFICATION DATASETS
# ============================================================================
STANDARD_DATASETS = {
    "MNIST": {
        "layer_sizes": [784, 256, 10],
        "timesteps": 25,
        "task": "classification",
    },
    "FashionMNIST": {
        "layer_sizes": [784, 256, 10],
        "timesteps": 25,
        "task": "classification",
    },
    "CIFAR10": {
        "layer_sizes": [3072, 512, 10],  # 32x32x3 = 3072
        "timesteps": 25,
        "task": "classification",
    },
    "SVHN": {
        "layer_sizes": [3072, 512, 10],
        "timesteps": 25,
        "task": "classification",
    },
    # "DVSGesture": {
    #     "layer_sizes": [1156, 256, 11],  # 34x34 = 1156
    #     "timesteps": 25,
    #     "task": "classification",
    # },
}

# ============================================================================
# NEUROBENCH OFFICIAL BENCHMARK DATASETS
# ============================================================================
NEUROBENCH_DATASETS = {
    # Classification tasks
    # SpeechCommands disabled - requires torchcodec not in container
    # "SpeechCommands": {
    #     "layer_sizes": [1600, 256, 12],  # Resampled audio features -> 12 keywords
    #     "timesteps": 100,
    #     "task": "classification",
    # },
    # WISDM disabled - requires pytorch_lightning not in container
    # "WISDM": {
    #     "layer_sizes": [3, 128, 6],  # 3-axis accel -> 6 activities
    #     "timesteps": 200,
    #     "task": "classification",
    # },
    # Regression tasks (require different loss function)
    # "PrimateReaching": {
    #     "layer_sizes": [96, 128, 2],  # Neural channels -> 2D velocity
    #     "timesteps": 50,
    #     "task": "regression",
    # },
    # "MackeyGlass": {
    #     "layer_sizes": [1, 64, 1],  # Time series prediction
    #     "timesteps": 50,
    #     "task": "regression",
    # },
}

# Combined datasets for benchmarking
DATASETS = {**STANDARD_DATASETS, **NEUROBENCH_DATASETS}

ALGORITHMS = {
    "bptt": BPTTTrainer,
    "stsf": STSFTrainer,
}


def run_all_benchmarks(
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001,
    device: str = "cuda",
    beta: float = 0.9,
    checkpoint_epochs: list = None,
    output_dir: str = "./benchmark_results",
):
    """Run benchmarks for all algorithms on all datasets."""
    
    if checkpoint_epochs is None:
        checkpoint_epochs = [1, 10, 25, epochs]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    print("\n" + "=" * 80)
    print("FULL BENCHMARK SUITE: BPTT vs STSF")
    print("=" * 80)
    print(f"Algorithms: {list(ALGORITHMS.keys())}")
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("=" * 80)
    
    for dataset_name, dataset_config in DATASETS.items():
        print(f"\n{'#' * 80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#' * 80}")
        
        dataset_results = {}
        
        for algo_name, trainer_class in ALGORITHMS.items():
            try:
                result = benchmark_algorithm(
                    algorithm_name=algo_name,
                    trainer_class=trainer_class,
                    dataset=dataset_name,
                    layer_sizes=dataset_config["layer_sizes"],
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    timesteps=dataset_config["timesteps"],
                    checkpoint_epochs=checkpoint_epochs,
                    device=device,
                    beta=beta,
                )
                dataset_results[algo_name] = result
            except Exception as e:
                print(f"ERROR benchmarking {algo_name} on {dataset_name}: {e}")
                continue
        
        all_results[dataset_name] = dataset_results
        
        # Print comparison for this dataset
        if dataset_results:
            print_comparison_summary(dataset_results)
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"full_benchmark_{timestamp}.json"
    
    # Convert to serializable format
    serializable_results = {}
    for dataset, algos in all_results.items():
        serializable_results[dataset] = {}
        for algo, result in algos.items():
            serializable_results[dataset][algo] = {
                "algorithm": result.algorithm,
                "dataset": result.dataset,
                "architecture": result.architecture,
                "final_accuracy": result.final_accuracy,
                "final_loss": result.final_loss,
                "epochs_trained": result.epochs_trained,
                "checkpoint_accuracies": result.checkpoint_accuracies,
                "total_wall_time_s": result.total_wall_time_s,
                "avg_epoch_cpu_ms": result.avg_epoch_cpu_ms,
                "avg_epoch_cuda_ms": result.avg_epoch_cuda_ms,
                "neurobench": result.neurobench,
                "algorithm_info": result.algorithm_info,
            }
    
    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n{'=' * 80}")
    print("FULL BENCHMARK COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results saved to: {results_file}")
    
    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<15} | {'BPTT Acc':<10} | {'STSF Acc':<10} | {'BPTT Time':<12} | {'STSF Time':<12}")
    print("-" * 80)
    
    for dataset, algos in all_results.items():
        bptt_acc = algos.get("bptt", {})
        stsf_acc = algos.get("stsf", {})
        
        bptt_acc_val = bptt_acc.final_accuracy if hasattr(bptt_acc, 'final_accuracy') else "N/A"
        stsf_acc_val = stsf_acc.final_accuracy if hasattr(stsf_acc, 'final_accuracy') else "N/A"
        bptt_time = bptt_acc.total_wall_time_s if hasattr(bptt_acc, 'total_wall_time_s') else "N/A"
        stsf_time = stsf_acc.total_wall_time_s if hasattr(stsf_acc, 'total_wall_time_s') else "N/A"
        
        if isinstance(bptt_acc_val, float):
            bptt_acc_val = f"{bptt_acc_val:.4f}"
        if isinstance(stsf_acc_val, float):
            stsf_acc_val = f"{stsf_acc_val:.4f}"
        if isinstance(bptt_time, float):
            bptt_time = f"{bptt_time:.2f}s"
        if isinstance(stsf_time, float):
            stsf_time = f"{stsf_time:.2f}s"
            
        print(f"{dataset:<15} | {bptt_acc_val:<10} | {stsf_acc_val:<10} | {bptt_time:<12} | {stsf_time:<12}")
    
    print("=" * 80)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark suite")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per benchmark")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Output directory")
    
    args = parser.parse_args()
    
    run_all_benchmarks(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

