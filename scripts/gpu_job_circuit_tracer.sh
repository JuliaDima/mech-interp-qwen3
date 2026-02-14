#!/bin/bash
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=job_1gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

# Recommended Ampere environment (per docs)
module purge
module load rhel8/default-amp

source ~/.bashrc
conda activate p28_py311_env

set -euo pipefail

# Helpful defaults (tune if needed)
export OMP_NUM_THREADS=16
export PYTHONPATH=${PYTHONPATH-}:/home/eid23/mechinterp-qwen-3B-Instruct/circuit_tracer_github
export LD_LIBRARY_PATH=/home/eid23/miniforge3/envs/p28_py311_env/lib

# Run your program
# Build attribution graph for a specific example
python -m circuit_tracer attribute \
  --prompt "You are solving a simple comparison task. Two numbers are given: A and B. Answer with a single character: 'A' if A is larger, otherwise 'B'. A = 864, B = 394, Answer: " \
  --transcoder_set mwhanna/qwen3-4b-transcoders \
  --model Qwen/Qwen3-4B \
  --slug qwen3-4b \
  --graph_file_dir ./graphs \
  --verbose \
  --node_threshold 0.7 \
  --edge_threshold 0.8 \
  --backend transformerlens \
  --lazy-encoder \
  --dtype bfloat16 \
  --offload disk \
  2>&1 | tee logs/job_1gpu_circuit_tracer$(date +%s).log
