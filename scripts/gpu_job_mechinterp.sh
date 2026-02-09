#!/bin/bash
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=job_1gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# Recommended Ampere environment (per docs)
module purge
module load rhel8/default-amp

source ~/.bashrc
conda activate p28_py311_env

# Helpful defaults (tune if needed)
export OMP_NUM_THREADS=16

# Run your program
# Build attribution graph for a specific example
miq-build-graph \
  --prompt "You are solving a simple comparison task.
Two numbers are given: A and B.
Answer with a single character: 'A' if A is larger, otherwise 'B'.

A = 864
B = 394
Answer: " \
  --slug gt_864_394 \
  --layers 4,12,20 \
  --graph_dir graphs
