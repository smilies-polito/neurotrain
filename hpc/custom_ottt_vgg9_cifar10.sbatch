#!/bin/bash
#SBATCH --job-name=ottt-vgg9-cifar10
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=hpc/slurm_outputs/ottt_vgg9_cifar10-%j.out
#SBATCH --error=hpc/slurm_outputs/ottt_vgg9_cifar10-%j.err

apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --custom config/custom/ottt_vgg9_cifar10.yaml --name ottt_vgg9_cifar10_opt
