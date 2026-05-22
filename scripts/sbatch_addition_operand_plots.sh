#!/bin/bash
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=addition_operand_plots
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module purge || true
module load rhel8/default-amp || true

source ~/.bashrc || true
conda activate p28_py311_env || true

set -euo pipefail

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/eid23/mechinterp-qwen-3B-Instruct/mechinterp-qwen3

cd /home/eid23/mechinterp-qwen-3B-Instruct/mechinterp-qwen3

echo "=== Phase 1: make-prompts ==="
python experiments/addition/run.py \
    --make-prompts \
    --out_root runs \
    --run_id addition

echo "=== Phase 2: operand-plots ==="
python experiments/addition/run.py \
    --operand-plots \
    --model Qwen/Qwen3-4B \
    --transcoder_set mwhanna/qwen3-4b-transcoders \
    --dtype bfloat16 \
    --top_k_features 50 \
    --out_root runs \
    --run_id addition \
    --seed 42

echo "=== Done. Outputs in runs/addition/operand_plots/ ==="
