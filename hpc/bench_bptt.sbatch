#!/bin/bash
#SBATCH --job-name=snn-bptt
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=snn-bptt-%j.out
#SBATCH --error=snn-bptt-%j.err

apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --benchmarking config/benchmarking/bptt.yaml --name bptt_opt
