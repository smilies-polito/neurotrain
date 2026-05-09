#!/bin/bash
#SBATCH --job-name=paper-vgg9-ottt-otttnet-svhn
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=hpc/slurm_outputs/paper_vgg9_ottt_otttnet_svhn-%j.out
#SBATCH --error=hpc/slurm_outputs/paper_vgg9_ottt_otttnet_svhn-%j.err

export PYTORCH_ALLOC_CONF=expandable_segments:True
apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --custom config/paper/vgg9/ottt_otttnet_svhn.yaml --no-progress --name paper_vgg9_ottt_otttnet_svhn
