#!/bin/bash
#SBATCH --job-name=snn-tp
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=hpc/slurm_outputs/snn-tp-%j.out
#SBATCH --error=hpc/slurm_outputs/snn-tp-%j.err

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
python_bin="${PYTHON_BIN:-python3}"

cd "${project_root}"
mkdir -p hpc/slurm_outputs experiments src/Data

cpus="${SLURM_CPUS_PER_TASK:-1}"
export STSF_DATA="${STSF_DATA:-${project_root}/src/Data}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${cpus}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${cpus}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${cpus}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${cpus}}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${cpus}}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export PYTHONUNBUFFERED=1

echo "[INFO] start=$(date -Iseconds) job_id=${SLURM_JOB_ID:-na} node=${SLURMD_NODENAME:-na}"
echo "[INFO] trainer=tp cpus_per_task=${cpus} python=${python_bin}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
fi

cmd="${python_bin} run_exp_campaign.py --benchmarking config/benchmarking/tp.yaml --name tp_opt"

if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    if ! command -v apptainer >/dev/null 2>&1; then
        echo "[ERROR] APPTAINER_IMAGE is set but apptainer is not available." >&2
        exit 1
    fi
    srun --cpu-bind=cores --hint=nomultithread \
        apptainer exec --nv --bind "${project_root}:${project_root}" "${APPTAINER_IMAGE}" \
        bash -lc "cd '${project_root}' && ${cmd}"
else
    srun --cpu-bind=cores --hint=nomultithread \
        bash -lc "cd '${project_root}' && ${cmd}"
fi

echo "[INFO] done=$(date -Iseconds)"
