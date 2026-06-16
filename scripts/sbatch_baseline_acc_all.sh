#!/bin/bash
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=baseline_all
#SBATCH --output=logs/baseline_all_%j.log
#SBATCH --error=logs/baseline_all_%j.log

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

module purge || true
module load rhel8/default-amp || true

set +u
source ~/.bashrc || true
VENV_PATH="${MIQ_VENV:-${REPO_ROOT}/.venv}"
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "Python virtual environment not found at ${VENV_PATH}" >&2
  echo "Create it with: /usr/bin/python3.11 -m venv .venv && source .venv/bin/activate && python -m pip install -e .[test,dev]" >&2
  exit 2
fi
source "${VENV_PATH}/bin/activate"
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src:${REPO_ROOT}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${VENV_PATH}/lib:${VENV_PATH}/lib64"
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME="/rds/user/${USER}/hpc-work/p28/cache/hf"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="/rds/user/${USER}/hpc-work/p28/cache/torch"

mkdir -p logs

echo "=========================================="
echo "Job:       ${SLURM_JOB_ID:-manual}"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repo root: ${REPO_ROOT}"
echo "Start:     $(date)"
echo "=========================================="

CONCEPTS=(
    carry gcd residue_class transitive_ordering conservation
    causal_direction negation_scope balanced_parentheses
    decimal_termination doppler_shift dot_product_sign
    geometric_series momentum_conservation perfect_square
    syllogism triangle_inequality wave_interference
)
TEMPLATES=(T0 T1 T2)

echo ""
echo "=== Baseline accuracy: all concepts x all templates ==="

for concept in "${CONCEPTS[@]}"; do
    for template in "${TEMPLATES[@]}"; do
        echo ""
        echo "--- ${concept} / ${template} ---"
        python -m experiments.concept_localization.run_feature_modulation \
            --concept "${concept}" \
            --template "${template}" \
            --n 100 || { echo "FAILED: ${concept}/${template}"; }
    done
done

echo ""
echo "=== Done: $(date) ==="
