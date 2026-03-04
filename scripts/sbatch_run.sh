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
#   sbatch scripts/sbatch_run.sh miq attribute -p "calc: 1+1="
#   sbatch scripts/sbatch_run.sh python experiments/addition/run.py --all
#
# To use a custom config file:
#   sbatch scripts/sbatch_run.sh miq --config my_config.yaml attribute ...

#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=job_1gpu_local
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- Repo root ----
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

echo "=========================================="
echo "Job:       ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repo root: ${REPO_ROOT}"
echo "Command:   ${*:-<none — will print help>}"
echo "=========================================="

mkdir -p logs/slurm

# ---- Environment ----
module purge || true
module load rhel8/default-amp || true
source ~/.bashrc
conda activate p28_py311_env || true

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src"
export OMP_NUM_THREADS=16

# ---- Logging ----
logfile="logs/slurm/${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID:-manual}_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "Logging to: ${logfile}"
echo "=========================================="

{
  echo "Start Time: $(date)"
  echo "Command:    $@"
  echo "------------------------------------------"

  # ---- Run command ----
  # All arguments are forwarded verbatim
  "$@"

  echo "------------------------------------------"
  echo "End Time:   $(date)"
} 2>&1 | tee "$logfile"
