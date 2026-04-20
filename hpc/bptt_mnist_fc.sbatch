#!/bin/bash
# =============================================================================
# SLURM Batch Script: BPTT MNIST FCSNN Training
# =============================================================================

# ---------------------  SLURM Resource Directives  ---------------------------
#SBATCH --job-name=bptt-mnist-fc               
#SBATCH --partition=gpu_a40                  
#SBATCH --nodes=1                               
#SBATCH --ntasks=1                              
#SBATCH --cpus-per-task=8                       
#SBATCH --gres=gpu:1                            
#SBATCH --mem=48G                     
#SBATCH --output=hpc/slurm_outputs/bptt_mnist_fc-%j.out
#SBATCH --error=hpc/slurm_outputs/bptt_mnist_fc-%j.err

set -euo pipefail

# ---------------------  Directory Setup  -------------------------------------
# Hardcoded absolute paths
# @TODO: remove absolute paths
PROJECT_ROOT="/home/acaviglia/snn-training-benchmarking"
cd "${PROJECT_ROOT}"
mkdir -p hpc/slurm_outputs src/Data

# ---------------------  Threading & Environment Variables  -------------------
cpus="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${cpus}"        
export MKL_NUM_THREADS="${cpus}"        
export OPENBLAS_NUM_THREADS="${cpus}" 
export NUMEXPR_NUM_THREADS="${cpus}" 
export NUMEXPR_MAX_THREADS="${cpus}" 
export OMP_PLACES="cores"                     
export OMP_PROC_BIND="spread"              
export PYTHONUNBUFFERED=1                                    

# ---------------------  Informational Logging  -------------------------------
echo "[INFO] start=$(date -Iseconds) job_id=${SLURM_JOB_ID:-na} node=${SLURMD_NODENAME:-na}"
echo "[INFO] cpus_per_task=${cpus}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
fi

# ---------------------  Job Execution  ---------------------------------------
# Run directly via Apptainer using hardcoded parameters
srun --cpu-bind=cores --hint=nomultithread \
    apptainer exec --nv --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${PROJECT_ROOT}/hpc/snn-training-benchmarking.sif" \
    bash -lc "cd '${PROJECT_ROOT}' && python3 tests/bptt_mnist_fc.py \
        --epochs 50 \
        --batch-size 64 \
        --timesteps 10 \
        --lr 1e-3 \
        --beta 0.95 \
        --threshold 1.0 \
        --seed 42 \
        --device cuda \
        --optuna-trials 100 \
        --optuna-epochs 100 \
        --study-name bptt_mnist_fc \
        --hpc-prints"

echo "[INFO] done=$(date -Iseconds)"
