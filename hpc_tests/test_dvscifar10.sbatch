#!/bin/bash
#SBATCH --job-name=snn-dvscifar10-test
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=hpc_tests/slurm_outputs/test_dvscifar10-%j.out
#SBATCH --error=hpc_tests/slurm_outputs/test_dvscifar10-%j.err

# Create output directory if needed
mkdir -p hpc_tests/slurm_outputs

singularity exec --nv --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    hpc/snn-training-benchmarking.sif \
    python3 hpc_tests/test_dvscifar10.py
