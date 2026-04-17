#!/bin/bash
#SBATCH --job-name=snn-ell
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=snn-ell-%j.out
#SBATCH --error=snn-ell-%j.err

cd $SLURM_SUBMIT_DIR
singularity exec --nv --bind /home/users/caviglia $HOME/container-defs/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --benchmarking config/benchmarking/ell.yaml --name ell_opt
