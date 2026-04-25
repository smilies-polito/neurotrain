#!/bin/bash
#SBATCH --job-name=ottt-vgg9-svhn
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=hpc/slurm_outputs/train_ottt_vgg9_svhn-%j.out
#SBATCH --error=hpc/slurm_outputs/train_ottt_vgg9_svhn-%j.err

apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --custom config/custom/ottt_vgg9_svhn_300ep.yaml --no-progress --name train_ottt_vgg9_svhn
