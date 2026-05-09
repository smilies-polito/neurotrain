#!/bin/bash
#SBATCH --job-name=ottt-mnist
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=hpc/slurm_outputs/bench_ottt_mnist-%j.out
#SBATCH --error=hpc/slurm_outputs/bench_ottt_mnist-%j.err

apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --benchmarking config/benchmarking/ottt_mnist.yaml --no-progress --name ottt_mnist_opt
