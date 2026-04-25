#!/bin/bash
#SBATCH --job-name=tp-vgg9-dvscifar10
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=hpc/slurm_outputs/train_tp_vgg9_dvscifar10-%j.out
#SBATCH --error=hpc/slurm_outputs/train_tp_vgg9_dvscifar10-%j.err

apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --custom config/custom/tp_vgg9_dvscifar10_300ep.yaml --no-progress --name train_tp_vgg9_dvscifar10
