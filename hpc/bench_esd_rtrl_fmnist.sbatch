#!/bin/bash
#SBATCH --job-name=snn-esd-rtrl
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=hpc/slurm_outputs/bench_esd_rtrl_fmnist-%j.out
#SBATCH --error=hpc/slurm_outputs/bench_esd_rtrl_fmnist-%j.err

apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --benchmarking config/benchmarking/esd_rtrl_fmnist.yaml --name esd_rtrl_fmnist_opt
