#!/bin/bash
#SBATCH --job-name=snn-tp-fmnist
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=hpc/slurm_outputs/bench_tp_fmnist-%j.out
#SBATCH --error=hpc/slurm_outputs/bench_tp_fmnist-%j.err

apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --benchmarking config/benchmarking/tp_fmnist.yaml --no-progress --name tp_fmnist_opt
