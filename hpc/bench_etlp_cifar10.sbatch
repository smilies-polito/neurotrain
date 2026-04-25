#!/bin/bash
#SBATCH --job-name=snn-etlp-cifar10
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=hpc/slurm_outputs/bench_etlp_cifar10-%j.out
#SBATCH --error=hpc/slurm_outputs/bench_etlp_cifar10-%j.err

apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --benchmarking config/benchmarking/etlp_cifar10.yaml --no-progress --name etlp_cifar10_opt
