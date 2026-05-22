#!/bin/bash
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=grid_and_plot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs data

module purge || true
module load rhel8/default-amp || true

source ~/.bashrc || true
conda activate p28_py311_env || true

set -euo pipefail

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/eid23/mechinterp-qwen-3B-Instruct/mechinterp-qwen3

# Generate full grid (a, b ∈ {0,...,99}, T0 only) with per-position probabilities
miq generate-dataset \
    --model Qwen/Qwen3-4B \
    --output_path data/addition_grid.jsonl \
    --sampling_strategy grid \
    --max_value 99 \
    --templates T0 \
    --seed 42 \
    --dtype bfloat16

# Plot all visualizations including positional cascade (pos 0 vs pos 1)
python -m experiments.addition.dataset_generation.visualize_dataset \
    data/addition_grid.jsonl \
    --output_dir runs/addition/accuracy_sweep/plots \
    --template T0

echo "Done. Plots in runs/addition/accuracy_sweep/plots/"
