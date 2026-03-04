#!/bin/bash
# =============================================================================
# SLURM Batch Script: BPTT CIFAR-10 VGG-11 Training
# =============================================================================
# This script submits a Spiking Neural Network (SNN) training job using
# Backpropagation Through Time (BPTT) on the CIFAR-10 dataset with a VGG-11
# architecture. It is designed to run on a SLURM-managed HPC cluster with
# GPU resources. Optionally supports Optuna hyperparameter tuning and
# Apptainer (Singularity) container execution.
# =============================================================================

# ---------------------  SLURM Resource Directives  ---------------------------
#SBATCH --job-name=bptt-cifar10-vgg11          # Job name shown in squeue
#SBATCH --partition=gpu_a40                  # GPU partition to submit to
#SBATCH --nodes=1                               # Number of compute nodes
#SBATCH --ntasks=1                              # Number of tasks (1 = single process)
#SBATCH --cpus-per-task=8                       # CPU cores allocated per task
#SBATCH --gres=gpu:1                            # Number of GPUs requested
#SBATCH --mem=48G                               # Total memory per node
#SBATCH --time=24:00:00                         # Maximum wall-clock time (HH:MM:SS)
#SBATCH --output=hpc/slurm_outputs/%x-%j.out   # Stdout log (%x=job name, %j=job ID)
#SBATCH --error=hpc/slurm_outputs/%x-%j.err    # Stderr log (%x=job name, %j=job ID)

# ---------------------  Shell Safety Options  --------------------------------
# -e: exit on error, -u: error on undefined vars, -o pipefail: catch pipe errors
set -euo pipefail

# ---------------------  Project & Python Setup  ------------------------------
# Resolve the project root directory. Prefers $PROJECT_ROOT env var, then the
# SLURM submission directory, and finally the current working directory.
project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

# Python interpreter to use for running the training script.
python_bin="python3"

# ---------------------  Experiment Hyperparameters  --------------------------
# Hardcoded experiment parameters (edit this file to change).
epochs="50"             # Number of training epochs
batch_size="64"        # Mini-batch size for training
timesteps="10"         # Number of SNN simulation timesteps per input
lr="1e-3"              # Learning rate for the optimizer
beta="0.95"            # Membrane potential decay factor (leaky neuron beta)
threshold="1.0"        # Spiking threshold for LIF neurons
seed="42"              # Random seed for reproducibility
device="cuda"          # Compute device: "cuda" for GPU, "cpu" for CPU

# ---------------------  Optuna Hyperparameter Tuning  ------------------------
# Set optuna_trials > 0 to enable hyperparameter search before final training.
optuna_trials="100"                    # Number of Optuna trials (0 = disabled)
optuna_epochs="100"                    # Epochs per Optuna trial
study_name="bptt_cifar10_vgg11"      # Optuna study name for tracking
optuna_storage=""                    # Optuna storage URL (e.g., sqlite:///...; empty = in-memory)

# ---------------------  Directory Setup  -------------------------------------
# Change to the project root and ensure output/data directories exist.
cd "${project_root}"
mkdir -p hpc/slurm_outputs src/Data

# ---------------------  Threading & Environment Variables  -------------------
# Configure thread counts for numerical libraries to match allocated CPUs,
# preventing over-subscription on shared HPC nodes.
cpus="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${cpus}}"        # OpenMP threads
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${cpus}}"        # Intel MKL threads
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${cpus}}" # OpenBLAS threads
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${cpus}}" # NumExpr threads
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${cpus}}" # NumExpr max threads
export OMP_PLACES="${OMP_PLACES:-cores}"                     # Bind threads to cores
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"              # Spread threads across cores
export PYTHONUNBUFFERED=1                                    # Disable Python output buffering

# ---------------------  Build the Command  -----------------------------------
# Construct the full command as an array for safe quoting and execution.
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

# Append the Optuna storage flag only if a storage URL has been specified.
if [[ -n "${optuna_storage}" ]]; then
    cmd+=("--optuna-storage" "${optuna_storage}")
fi

# ---------------------  Informational Logging  -------------------------------
# Print metadata about the job for debugging and provenance tracking.
echo "[INFO] start=$(date -Iseconds) job_id=${SLURM_JOB_ID:-na} node=${SLURMD_NODENAME:-na}"
echo "[INFO] project_root=${project_root} cpus_per_task=${cpus} python=${python_bin}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] command:"
printf ' %q' "${cmd[@]}"
echo

# Print GPU information if nvidia-smi is available (for logging purposes).
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
fi

# ---------------------  Job Execution  ---------------------------------------
# Supports two execution modes:
#   1. Apptainer (Singularity) container mode — if APPTAINER_IMAGE is set,
#      the job runs inside the specified container with GPU passthrough (--nv).
#   2. Bare-metal mode — runs the command directly via srun on the host.
if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    # Verify that the apptainer binary is available on this node.
    if ! command -v apptainer >/dev/null 2>&1; then
        echo "[ERROR] APPTAINER_IMAGE is set but apptainer is not available." >&2
        exit 1
    fi

    # Escape the command for safe embedding in a bash -c string.
    cmd_escaped="$(printf '%q ' "${cmd[@]}")"

    # Launch inside the Apptainer container with GPU support and project bind-mount.
    srun --cpu-bind=cores --hint=nomultithread \
        apptainer exec --nv --bind "${project_root}:${project_root}" "${APPTAINER_IMAGE}" \
        bash -lc "cd '${project_root}' && ${cmd_escaped}"
else
    # Launch directly on the host node via srun.
    srun --cpu-bind=cores --hint=nomultithread "${cmd[@]}"
fi

# ---------------------  Completion Log  --------------------------------------
echo "[INFO] done=$(date -Iseconds)"
