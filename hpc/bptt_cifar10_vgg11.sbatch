#!/bin/bash
#SBATCH --job-name=bptt-cifar10-vgg11
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/slurm_outputs/%x-%j.out
#SBATCH --error=hpc/slurm_outputs/%x-%j.err

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
python_bin="python3"

# Hardcoded experiment parameters (edit this file to change).
epochs="1"
batch_size="64"
timesteps="10"
lr="1e-3"
beta="0.95"
threshold="1.0"
seed="42"
device="cuda"

optuna_trials="0"
optuna_epochs="1"
study_name="bptt_cifar10_vgg11"
optuna_storage=""

cd "${project_root}"
mkdir -p hpc/slurm_outputs src/Data

cpus="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${cpus}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${cpus}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${cpus}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${cpus}}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${cpus}}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export PYTHONUNBUFFERED=1

cmd=(
    "${python_bin}" "tests/bptt_cifar10_vgg11.py"
    "--epochs" "${epochs}"
    "--batch-size" "${batch_size}"
    "--timesteps" "${timesteps}"
    "--lr" "${lr}"
    "--beta" "${beta}"
    "--threshold" "${threshold}"
    "--seed" "${seed}"
    "--device" "${device}"
    "--optuna-trials" "${optuna_trials}"
    "--optuna-epochs" "${optuna_epochs}"
    "--study-name" "${study_name}"
)

if [[ -n "${optuna_storage}" ]]; then
    cmd+=("--optuna-storage" "${optuna_storage}")
fi

echo "[INFO] start=$(date -Iseconds) job_id=${SLURM_JOB_ID:-na} node=${SLURMD_NODENAME:-na}"
echo "[INFO] project_root=${project_root} cpus_per_task=${cpus} python=${python_bin}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] command:"
printf ' %q' "${cmd[@]}"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
fi

if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    if ! command -v apptainer >/dev/null 2>&1; then
        echo "[ERROR] APPTAINER_IMAGE is set but apptainer is not available." >&2
        exit 1
    fi

    cmd_escaped="$(printf '%q ' "${cmd[@]}")"
    srun --cpu-bind=cores --hint=nomultithread \
        apptainer exec --nv --bind "${project_root}:${project_root}" "${APPTAINER_IMAGE}" \
        bash -lc "cd '${project_root}' && ${cmd_escaped}"
else
    srun --cpu-bind=cores --hint=nomultithread "${cmd[@]}"
fi

echo "[INFO] done=$(date -Iseconds)"
