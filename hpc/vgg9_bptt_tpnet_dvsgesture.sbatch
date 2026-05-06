#!/bin/bash
#SBATCH --job-name=snn-vgg9-bptt-tpnet-dvsgesture
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=hpc/slurm_outputs/vgg9_bptt_tpnet_dvsgesture-%j.out
#SBATCH --error=hpc/slurm_outputs/vgg9_bptt_tpnet_dvsgesture-%j.err

export PYTORCH_ALLOC_CONF=expandable_segments:True
apptainer exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --custom config/vgg9/bptt_tpnet_dvsgesture_opt.yaml --no-progress --name vgg9_bptt_tpnet_dvsgesture_opt
