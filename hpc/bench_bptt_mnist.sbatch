#!/bin/bash
#SBATCH --job-name=snn-bptt-mnist
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=hpc/slurm_outputs/bench_bptt_mnist-%j.out
#SBATCH --error=hpc/slurm_outputs/bench_bptt_mnist-%j.err

apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --benchmarking config/benchmarking/bptt_mnist.yaml --name bptt_mnist_opt
