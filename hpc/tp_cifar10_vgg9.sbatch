# =============================================================================
# SLURM Batch Script: TP CIFAR-10 VGG9 Optuna
# =============================================================================

# ---------------------  SLURM Resource Directives  ---------------------------
#SBATCH --job-name=tp-cifar10-vgg9
#SBATCH --partition=serics_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=/home/users/caviglia/snn-training-benchmarking/hpc/slurm_outputs/%x-%j.out
#SBATCH --error=/home/users/caviglia/snn-training-benchmarking/hpc/slurm_outputs/%x-%j.err

set -euo pipefail

# ---------------------  Directory Setup  -------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/home/users/caviglia/snn-training-benchmarking}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$HOME/container-defs/snn-training-benchmarking.sif}"

cd "${PROJECT_ROOT}"
mkdir -p hpc/slurm_outputs optuna_results src/Data

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

# ---------------------  Optuna / Run Parameters  -----------------------------
OPTUNA_TRIALS="${OPTUNA_TRIALS:-100}"
OPTUNA_EPOCHS="${OPTUNA_EPOCHS:-20}"
TIMESTEPS="${TIMESTEPS:-20}"
SEED="${SEED:-42}"
STUDY_NAME="${STUDY_NAME:-tp_cifar10_vgg9}"
OPTUNA_DB_PATH="${OPTUNA_DB_PATH:-${PROJECT_ROOT}/optuna_results/tp_cifar10_vgg9.db}"
OPTUNA_STORAGE="sqlite:////${OPTUNA_DB_PATH#/}"

# ---------------------  Informational Logging  -------------------------------
echo "[INFO] start=$(date -Iseconds) job_id=${SLURM_JOB_ID:-na} node=${SLURMD_NODENAME:-na}"
echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] image=${APPTAINER_IMAGE}"
echo "[INFO] cpus_per_task=${cpus}"
echo "[INFO] study_name=${STUDY_NAME}"
echo "[INFO] optuna_trials=${OPTUNA_TRIALS} optuna_epochs=${OPTUNA_EPOCHS}"
echo "[INFO] optuna_storage=${OPTUNA_STORAGE}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
fi

# ---------------------  Job Execution  ---------------------------------------
srun --cpu-bind=cores --hint=nomultithread \
  apptainer exec --nv --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${APPTAINER_IMAGE}" \
  bash -lc "cd '${PROJECT_ROOT}' && python3 tests/tp_cifar10_vgg9.py \
    --timesteps ${TIMESTEPS} \
    --seed ${SEED} \
    --device cuda \
    --optuna-trials ${OPTUNA_TRIALS} \
    --optuna-epochs ${OPTUNA_EPOCHS} \
    --study-name ${STUDY_NAME} \
    --hpc-prints \
    --optuna-storage ${OPTUNA_STORAGE}"

echo "[INFO] done=$(date -Iseconds)"