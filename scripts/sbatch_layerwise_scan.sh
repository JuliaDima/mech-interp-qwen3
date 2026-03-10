#!/bin/bash
# SLURM job array script for layer-wise carry probe analysis
#
# This script runs one probe per layer (36 total) in parallel using a job array.
# Each job trains a probe on a single layer with single token position.
#
# Usage:
#   sbatch scripts/sbatch_layerwise_scan.sh
#
# Monitor progress:
#   squeue -u $USER
#   ls -lh logs/carry_probe_layer_*
#
# After completion, analyze results:
#   python scripts/analyze_layerwise_results.py --scan_dir runs/carry_probe/layerwise_scan_YYYYMMDD_HHMMSS

#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=carry_probe
#SBATCH --array=0-35%4                  # Run 36 jobs (max 4 concurrent - limited by GPU availability)
#SBATCH --output=logs/carry_probe_layer_%a_%j.log
#SBATCH --error=logs/carry_probe_layer_%a_%j.log

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
conda activate p28_py311_env || true

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src:${REPO_ROOT}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/eid23/miniforge3/envs/p28_py311_env/lib"
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

# ---- Configuration ----
LAYER=${SLURM_ARRAY_TASK_ID}
N_TRAIN=1000
N_EPOCHS=30
MAX_VALUE=99
TOKEN_POSITION="answer"
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="runs/carry_probe/layerwise_scan_${RUN_TIMESTAMP}"

# Create output directory (only first job creates it)
mkdir -p "${OUTPUT_BASE}" logs

# Save configuration (only first job creates it)
if [ ${SLURM_ARRAY_TASK_ID} -eq 0 ]; then
  cat > "${OUTPUT_BASE}/config.txt" <<EOF
Layer-wise Scan Configuration
=============================
Date: $(date)
N_train: ${N_TRAIN}
N_epochs: ${N_EPOCHS}
Max_value: ${MAX_VALUE}
Token_position: ${TOKEN_POSITION}
Total layers: 36
SLURM Job ID: ${SLURM_ARRAY_JOB_ID}
EOF
fi

# ---- Logging header ----
echo "=========================================="
echo "Layer-wise Carry Probe Training"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array Job ID:  ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node:          $(hostname)"
echo "GPU:           ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Layer:         ${LAYER}/35"
echo "Repo root:     ${REPO_ROOT}"
echo "Output dir:    ${OUTPUT_BASE}"
echo "Start time:    $(date)"
echo "=========================================="
echo ""

# ---- Run training ----
python scripts/train_carry_probe.py \
    --layers ${LAYER} \
    --token_position ${TOKEN_POSITION} \
    --n_train ${N_TRAIN} \
    --n_epochs ${N_EPOCHS} \
    --learning_rate 5e-3 \
    --max_value ${MAX_VALUE} \
    --run_id "layer_${LAYER}" \
    --output_dir "${OUTPUT_BASE}"

# ---- Logging footer ----
echo ""
echo "=========================================="
echo "Layer ${LAYER} training complete!"
echo "End time: $(date)"
echo "=========================================="

# ---- Post-processing (only last job) ----
if [ ${SLURM_ARRAY_TASK_ID} -eq 35 ]; then
  echo ""
  echo "All layers complete. To analyze results:"
  echo "  python scripts/analyze_layerwise_results.py --scan_dir ${OUTPUT_BASE}"
fi
