#!/bin/bash
#SBATCH --job-name=snn-tp
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=snn-tp-%j.out
#SBATCH --error=snn-tp-%j.err

cd $SLURM_SUBMIT_DIR
singularity exec --nv --bind /home/users/caviglia $HOME/container-defs/snn-training-benchmarking.sif \
    python3 run_exp_campaign.py --benchmarking config/benchmarking/tp.yaml --name tp_opt
