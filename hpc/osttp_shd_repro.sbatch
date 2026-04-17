#!/bin/bash
# =============================================================================
# SLURM Batch Script: OSTTP SHD Reproduction Run
# =============================================================================

# ---------------------  SLURM Resource Directives  ---------------------------
#SBATCH --job-name=osttp-shd-repro
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
python_bin="${PYTHON_BIN:-python3}"

cd "${project_root}"
mkdir -p hpc/slurm_outputs src/Data

cpus="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${cpus}"
export MKL_NUM_THREADS="${cpus}"
export OPENBLAS_NUM_THREADS="${cpus}"
export NUMEXPR_NUM_THREADS="${cpus}"
export NUMEXPR_MAX_THREADS="${cpus}"
export OMP_PLACES="cores"
export OMP_PROC_BIND="spread"
export PYTHONUNBUFFERED=1

echo "[INFO] start=$(date -Iseconds) job_id=${SLURM_JOB_ID:-na} node=${SLURMD_NODENAME:-na}"
echo "[INFO] cpus_per_task=${cpus} python=${python_bin}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
fi

run_cmd="cd '${project_root}' && ${python_bin} tests/osttp_shd_repro.py \
    --epochs 200 \
    --batch-size 64 \
    --timesteps 100 \
    --lr 2e-4 \
    --beta 0.95 \
    --threshold 1.0 \
    --seed 42 \
    --device cuda \
    --hpc-prints"

if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    srun --cpu-bind=cores --hint=nomultithread \
        apptainer exec --nv --bind "${project_root}:${project_root}" "${APPTAINER_IMAGE}" \
        bash -lc "${run_cmd}"
else
    srun --cpu-bind=cores --hint=nomultithread \
        bash -lc "${run_cmd}"
fi

echo "[INFO] done=$(date -Iseconds)"
