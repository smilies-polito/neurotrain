#!/bin/bash
#SBATCH --job-name=osttp-dvscifar10
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=hpc/slurm_outputs/bench_osttp_dvscifar10-%j.out
#SBATCH --error=hpc/slurm_outputs/bench_osttp_dvscifar10-%j.err

apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --benchmarking config/benchmarking/osttp_dvscifar10.yaml --no-progress --name osttp_dvscifar10_opt
