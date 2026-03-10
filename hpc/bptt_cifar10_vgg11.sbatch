#!/bin/bash
# =============================================================================
# SLURM Batch Script: BPTT CIFAR-10 VGG-11 Training
# =============================================================================

# ---------------------  SLURM Resource Directives  ---------------------------
#SBATCH --job-name=bptt-cifar10-vgg11          
#SBATCH --partition=gpu_a40                  
#SBATCH --nodes=1                               
#SBATCH --ntasks=1                              
#SBATCH --cpus-per-task=8                       
#SBATCH --gres=gpu:1                            
#SBATCH --mem=48G                     
#SBATCH --output=/home/acaviglia/snn-training-benchmarking/hpc/slurm_outputs/%x-%j.out   
#SBATCH --error=/home/acaviglia/snn-training-benchmarking/hpc/slurm_outputs/%x-%j.err    

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
    bash -lc "cd '${PROJECT_ROOT}' && python3 tests/bptt_cifar10_vgg11.py \
        --timesteps 10 \
        --seed 42 \
        --device cuda \
        --optuna-trials 100 \
        --optuna-epochs 20 \
        --study-name bptt_cifar10_vgg11 \
        --hpc-prints \
        --optuna-storage sqlite:///optuna_results.db"

echo "[INFO] done=$(date -Iseconds)"
