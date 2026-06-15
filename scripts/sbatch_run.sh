#!/bin/bash
# Universal Slurm submission script for mechinterp-qwen3.
#
# This script forwards any command (miq CLI, experiment scripts, etc.)
# and ensures the environment is set up. It uses the root config.yaml
# for project-wide defaults.
#
# Usage:
#   sbatch scripts/sbatch_run.sh [COMMAND...]
#
# Examples:
#   sbatch scripts/sbatch_run.sh miq generate-dataset --output_path results.jsonl
#   sbatch scripts/sbatch_run.sh miq attribute -p "calc: 1+1= "
#   sbatch scripts/sbatch_run.sh miq attribute -p "You are solving a simple comparison task. Two numbers are given: A and B. Answer with a single character: 'A' if A is larger, otherwise 'B'. A = 864, B = 394, Answer: "
#   sbatch scripts/sbatch_run.sh python experiments/addition/run.py --all
#   sbatch --mail-user=[ACCOUNT] --mail-type=BEGIN scripts/sbatch_run.sh command args...

# To use a custom config file:
#   sbatch scripts/sbatch_run.sh miq --config my_config.yaml attribute ...

#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=miq_run
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
##SBATCH --mail-type=BEGIN
##SBATCH --mail-user=eid23@cam.ac.uk

# ---- Repo root ----
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

# ---- Environment ----
module purge || true
module load rhel8/default-amp || true

# Disable nounset temporarily for system scripts
set +u
source ~/.bashrc
set -u
CONDA_ENV="${MIQ_CONDA_ENV:-p28_py311_env}"
CONDA_ROOT="${MIQ_CONDA_ROOT:-${HOME}/miniforge3}"
if [ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  echo "conda not found at ${CONDA_ROOT}" >&2
  echo "Install miniforge3 to ${CONDA_ROOT} first." >&2
  exit 2
fi
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
hash -r 2>/dev/null || true

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src:${REPO_ROOT}"
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

# Source project env: sets HF_HOME → shared RDS cache, disables Xet, etc.
# This overrides the node-local /local/$USER/hf set in ~/.bashrc.
set +u
source "${REPO_ROOT}/scripts/miq-env.sh"
set -u
# Unset legacy cache vars that .bashrc may set to node-local paths;
# miq-env.sh already sets HF_HOME to the shared RDS cache.
unset TRANSFORMERS_CACHE HF_HUB_CACHE || true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs

# ---- Logging ----
logfile="logs/${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID:-manual}_$(date +%Y-%m-%d_%H-%M-%S).log"

{
  echo "=========================================="
  echo "Job:       ${SLURM_JOB_ID:-manual}"
  echo "Node:      $(hostname)"
  echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "Repo root: ${REPO_ROOT}"
  echo "Command:   $@"
  echo "Logging to: ${logfile}"
  echo "Start Time: $(date)"
  echo "=========================================="

  # ---- Run command ----
  # All arguments are forwarded verbatim.
  # IMPORTANT: wrap your prompt in quotes when calling sbatch!
  "$@"

  echo "------------------------------------------"
  echo "End Time:   $(date)"
} 2>&1 | tee "$logfile"
