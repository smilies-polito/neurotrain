#!/bin/bash
#SBATCH --job-name=snn-ostl-cifar10
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=hpc/slurm_outputs/bench_ostl_cifar10-%j.out
#SBATCH --error=hpc/slurm_outputs/bench_ostl_cifar10-%j.err

apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --benchmarking config/benchmarking/ostl_cifar10.yaml --no-progress --name ostl_cifar10_opt
