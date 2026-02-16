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

source /home/eid23/miniforge3/etc/profile.d/conda.sh
conda activate p28_py311_env

# Debugging info
which python
python --version
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device Count: {torch.cuda.device_count()}')"
nvidia-smi

set -euo pipefail

# Helpful defaults (tune if needed)
export OMP_NUM_THREADS=16
export PYTHONPATH=${PYTHONPATH-}:/home/eid23/mechinterp-qwen-3B-Instruct/mechinterp-qwen3
export LD_LIBRARY_PATH=/home/eid23/miniforge3/envs/p28_py311_env/lib

# Build attribution graph for a specific example
miq-build-graph \
  --prompt "You are solving a simple comparison task. Two numbers are given: A and B. Answer with a single character: 'A' if A is larger, otherwise 'B'. A = 864, B = 394, Answer: " \
  --slug gt_864_394 \
  --layers 4,12,20 \
  --graph_file_dir graphs \
  --max_n_logits 10 \
  --use_patching True \
  --node_threshold 0.8 \
  --edge_threshold 0.85 \
  2>&1 | tee logs/job_1gpu_$(date +%s).log
