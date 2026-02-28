#!/bin/bash
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=job_1gpu_local
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module purge
module load rhel8/default-amp

source ~/.bashrc
conda activate p28_py311_env

set -euo pipefail

export OMP_NUM_THREADS=16
export PYTHONPATH=${PYTHONPATH-}:/home/eid23/mechinterp-qwen-3B-Instruct/mechinterp_qwen3
export LD_LIBRARY_PATH=/home/eid23/miniforge3/envs/p28_py311_env/lib

logfile="logs/job_1gpu_local_$(date +%Y-%m-%d_%H-%M-%S).log"

cmd=(
  python3 -m mechinterp_qwen3 attribute
  --prompt "You are solving a simple comparison task. Two numbers are given: A and B. Answer with a single character: 'A' if A is larger, otherwise 'B'. A = 864, B = 394, Answer: "
  --transcoder_set mwhanna/qwen3-4b-transcoders
  --model Qwen/Qwen3-4B
  --slug qwen3-4b
  --graph_file_dir ./graphs
  --verbose
  --node_threshold 0.8
  --edge_threshold 0.95
  --lazy-encoder
  --dtype bfloat16
  --offload cpu
  --stats_file stats.json
)


{
  echo "Command:"
  i=0
  while [ $i -lt ${#cmd[@]} ]; do
    if [[ "${cmd[$i]}" == --* ]] && [ $((i+1)) -lt ${#cmd[@]} ] && [[ "${cmd[$((i+1))]}" != --* ]]; then
      printf '  %s %q\n' "${cmd[$i]}" "${cmd[$((i+1))]}"
      i=$((i+2))
    else
      printf '  %s\n' "${cmd[$i]}"
      i=$((i+1))
    fi
  done
  echo

  "${cmd[@]}"
} 2>&1 | tee "$logfile"
