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
from trainers.decolle_trainer import DECOLLETrainer


# Dataset configurations: dataset_name -> (input_size, num_classes, layer_sizes)
# ============================================================================
# RATE-CODED IMAGE CLASSIFICATION DATASETS
# ============================================================================
RATE_CODED_DATASETS = {
    "MNIST": {
        "layer_sizes": [784, 256, 10],
        "timesteps": 25,
        "task": "classification",
        "type": "rate-coded",
    },
    "FashionMNIST": {
        "layer_sizes": [784, 256, 10],
        "timesteps": 25,
        "task": "classification",
        "type": "rate-coded",
    },
    "CIFAR10": {
        "layer_sizes": [3072, 512, 10],  # 32x32x3 = 3072
        "timesteps": 25,
        "task": "classification",
        "type": "rate-coded",
    },
    "SVHN": {
        "layer_sizes": [3072, 512, 10],
        "timesteps": 25,
        "task": "classification",
        "type": "rate-coded",
    },
}

# ============================================================================
# EVENT-BASED NEUROMORPHIC DATASETS (ideal for DECOLLE)
# ============================================================================
EVENT_BASED_DATASETS = {
    "NMNIST": {
        "layer_sizes": [1156, 256, 10],  # 34x34 = 1156, 10 digits
        "timesteps": 25,
        "task": "classification",
        "type": "event-based",
    },
    "DVSGesture": {
        "layer_sizes": [16384, 512, 11],  # 128x128 = 16384, 11 gestures
        "timesteps": 50,
        "task": "classification",
        "type": "event-based",
    },
}

# Combined standard datasets
STANDARD_DATASETS = {**RATE_CODED_DATASETS, **EVENT_BASED_DATASETS}

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
    "decolle": DECOLLETrainer,
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
    print("FULL BENCHMARK SUITE: BPTT vs STSF vs DECOLLE")
    print("=" * 80)
    print(f"Algorithms: {list(ALGORITHMS.keys())}")
    print(f"Rate-coded datasets: {list(RATE_CODED_DATASETS.keys())}")
    print(f"Event-based datasets: {list(EVENT_BASED_DATASETS.keys())} (ideal for DECOLLE)")
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
    print("\n" + "=" * 160)
    print(f"FINAL SUMMARY ({epochs} epochs)")
    print("=" * 160)
    
    print(f"{'Dataset':<14} | {'BPTT Acc':<9} | {'STSF Acc':<9} | {'DECOLLE Acc':<11} | {'BPTT Wall Time':<15} | {'STSF Wall Time':<15} | {'DECOLLE Wall Time':<17} | {'BPTT Time/epoch':<16} | {'STSF Time/epoch':<16} | {'DECOLLE Time/epoch':<19}")
    print("-" * 210)
    
    for dataset, algos in all_results.items():
        bptt_res = algos.get("bptt", {})
        stsf_res = algos.get("stsf", {})
        decolle_res = algos.get("decolle", {})
        
        # Accuracy
        bptt_acc = bptt_res.final_accuracy if hasattr(bptt_res, 'final_accuracy') else None
        stsf_acc = stsf_res.final_accuracy if hasattr(stsf_res, 'final_accuracy') else None
        decolle_acc = decolle_res.final_accuracy if hasattr(decolle_res, 'final_accuracy') else None
        
        # Total wall time
        bptt_total = bptt_res.total_wall_time_s if hasattr(bptt_res, 'total_wall_time_s') else None
        stsf_total = stsf_res.total_wall_time_s if hasattr(stsf_res, 'total_wall_time_s') else None
        decolle_total = decolle_res.total_wall_time_s if hasattr(decolle_res, 'total_wall_time_s') else None
        
        # Per-epoch wall-clock time (ms)
        bptt_epoch = bptt_res.avg_epoch_cpu_ms if hasattr(bptt_res, 'avg_epoch_cpu_ms') else None
        stsf_epoch = stsf_res.avg_epoch_cpu_ms if hasattr(stsf_res, 'avg_epoch_cpu_ms') else None
        decolle_epoch = decolle_res.avg_epoch_cpu_ms if hasattr(decolle_res, 'avg_epoch_cpu_ms') else None
        
        # Format values
        bptt_acc_str = f"{bptt_acc:.4f}" if bptt_acc is not None else "N/A"
        stsf_acc_str = f"{stsf_acc:.4f}" if stsf_acc is not None else "N/A"
        decolle_acc_str = f"{decolle_acc:.4f}" if decolle_acc is not None else "N/A"
        bptt_total_str = f"{bptt_total:.1f}s" if bptt_total is not None else "N/A"
        stsf_total_str = f"{stsf_total:.1f}s" if stsf_total is not None else "N/A"
        decolle_total_str = f"{decolle_total:.1f}s" if decolle_total is not None else "N/A"
        bptt_epoch_str = f"{bptt_epoch:.0f}ms" if bptt_epoch is not None else "N/A"
        stsf_epoch_str = f"{stsf_epoch:.0f}ms" if stsf_epoch is not None else "N/A"
        decolle_epoch_str = f"{decolle_epoch:.0f}ms" if decolle_epoch is not None else "N/A"
            
        print(f"{dataset:<14} | {bptt_acc_str:<9} | {stsf_acc_str:<9} | {decolle_acc_str:<11} | {bptt_total_str:<15} | {stsf_total_str:<15} | {decolle_total_str:<17} | {bptt_epoch_str:<16} | {stsf_epoch_str:<16} | {decolle_epoch_str:<19}")
    
    print("=" * 160)
    
    # Print NeuroBench metrics summary
    print("\n" + "=" * 180)
    print("NEUROBENCH METRICS SUMMARY")
    print("=" * 180)
    print(f"{'Dataset':<14} | {'Algo':<6} | {'Params':<12} | {'Footprint':<12} | {'ActSpars':<10} | {'Eff. MACs':<14} | {'Dense MACs':<14} | {'Savings':<8} | {'MemUpdates':<12}")
    print("-" * 180)
    
    for dataset, algos in all_results.items():
        for algo_name in ["bptt", "stsf", "decolle"]:
            res = algos.get(algo_name, {})
            if not hasattr(res, 'neurobench'):
                continue
            
            nb = res.neurobench if hasattr(res, 'neurobench') else {}
            if not nb:
                continue
            
            # Extract NeuroBench metrics
            params = nb.get("ParameterCount", "N/A")
            footprint = nb.get("Footprint", "N/A")
            act_sparsity = nb.get("ActivationSparsity", "N/A")
            synops = nb.get("SynapticOperations", "N/A")
            mem_updates = nb.get("MembraneUpdates", "N/A")
            
            # Format values
            if isinstance(params, (int, float)):
                params_str = f"{int(params):,}"
            else:
                params_str = str(params)[:12]
            
            # Footprint in KB or MB
            if isinstance(footprint, (int, float)):
                if footprint >= 1024 * 1024:
                    footprint_str = f"{footprint / (1024*1024):.2f} MB"
                elif footprint >= 1024:
                    footprint_str = f"{footprint / 1024:.1f} KB"
                else:
                    footprint_str = f"{int(footprint)} B"
            else:
                footprint_str = str(footprint)[:12]
            
            if isinstance(act_sparsity, float):
                act_str = f"{act_sparsity:.4f}"
            else:
                act_str = str(act_sparsity)[:10]
            
            # SynapticOperations returns a dict like {'Effective_MACs': value, 'Dense': value}
            eff_macs = 0
            dense_macs = 0
            if isinstance(synops, dict):
                eff_macs = synops.get("Effective_MACs", 0)
                dense_macs = synops.get("Dense", 0)
                eff_str = f"{int(eff_macs):,}" if isinstance(eff_macs, (int, float)) else str(eff_macs)[:14]
                dense_str = f"{int(dense_macs):,}" if isinstance(dense_macs, (int, float)) else str(dense_macs)[:14]
            elif isinstance(synops, (int, float)):
                eff_str = f"{int(synops):,}"
                dense_str = "N/A"
            else:
                eff_str = str(synops)[:14]
                dense_str = "N/A"
            
            # Compute savings percentage
            if isinstance(eff_macs, (int, float)) and isinstance(dense_macs, (int, float)) and dense_macs > 0:
                savings = (1 - eff_macs / dense_macs) * 100
                savings_str = f"{savings:.1f}%"
            else:
                savings_str = "N/A"
            
            # MembraneUpdates may also be a dict
            if isinstance(mem_updates, dict):
                mem_val = list(mem_updates.values())[0] if mem_updates else 0
                mem_str = f"{int(mem_val):,}" if isinstance(mem_val, (int, float)) else str(mem_val)[:12]
            elif isinstance(mem_updates, (int, float)):
                mem_str = f"{int(mem_updates):,}"
            else:
                mem_str = str(mem_updates)[:12]
            
            print(f"{dataset:<14} | {algo_name.upper():<6} | {params_str:<12} | {footprint_str:<12} | {act_str:<10} | {eff_str:<14} | {dense_str:<14} | {savings_str:<8} | {mem_str:<12}")
    
    print("=" * 180)
    
    print("\nLegend:")
    print("  Training Summary:")
    print("    - Acc: Final test accuracy")
    print("    - Wall Time: Total wall-clock time for all epochs")
    print("    - Time/epoch: Average wall-clock time per epoch (includes compute + data loading)")
    print("  NeuroBench Metrics:")
    print("    - Params: Total number of model parameters")
    print("    - Footprint: Memory footprint of the model (weights + buffers)")
    print("    - ActSpars: Activation sparsity (fraction of zero spikes - higher = more efficient)")
    print("    - Eff. MACs: Effective MACs (actual ops with spike sparsity)")
    print("    - Dense MACs: Dense MACs (ops if all neurons fired)")
    print("    - Savings: Compute reduction from spike sparsity ((1 - Eff/Dense) * 100%)")
    print("    - MemUpdates: Number of membrane potential updates")
    
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

