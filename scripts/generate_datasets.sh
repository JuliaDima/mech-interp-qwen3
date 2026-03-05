#!/bin/bash
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=generate_datasets
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
mkdir -p data

module purge || true
module load rhel8/default-amp || true

source ~/.bashrc || true
conda activate p28_py311_env || true

set -euo pipefail

export OMP_NUM_THREADS=16
export PYTHONPATH=${PYTHONPATH-}:/home/eid23/mechinterp-qwen-3B-Instruct/mechinterp-qwen3

logfile="logs/generate_datasets_$(date +%Y-%m-%d_%H-%M-%S).log"

run_cmd() {
  local cmd=("$@")
  echo "------------------------------------------------------------" | tee -a "$logfile"
  echo "Running command: ${cmd[*]}" | tee -a "$logfile"
  echo "------------------------------------------------------------" | tee -a "$logfile"
  "${cmd[@]}" 2>&1 | tee -a "$logfile"
}

# Run 1: Grid Sampling (All Templates, 0-20)
run_cmd miq generate-dataset \
    --model Qwen/Qwen3-4B \
    --output_path data/addition_grid.jsonl \
    --sampling_strategy grid \
    --max_value 20 \
    --templates T0 T1 T2 \
    --seed 42 \
    --dtype bfloat16

# Run 2: Stratified Sampling (Balanced Carry Patterns, 0-100)
run_cmd miq generate-dataset \
    --model Qwen/Qwen3-4B \
    --output_path data/addition_stratified.jsonl \
    --sampling_strategy stratified \
    --max_value 100 \
    --templates T0 \
    --stratified_n_per_category 50 \
    --stratified_uniform_remainder 100 \
    --seed 42 \
    --dtype bfloat16

# Run 3: Random Sampling (0-1000, 500 samples)
run_cmd miq generate-dataset \
    --model Qwen/Qwen3-4B \
    --output_path data/addition_random.jsonl \
    --sampling_strategy random \
    --max_value 1000 \
    --n_samples 500 \
    --templates T1 \
    --seed 42 \
    --dtype bfloat16

echo "All dataset generation runs complete!" | tee -a "$logfile"
