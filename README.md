# Mechanistic Interpretability of Qwen3-4B-Instruct

This repository contains the code for the paper "Mechanistic Interpretability of Qwen3-4B-Instruct". The code is written in Python and is based on the Hugging Face transformers library.


## Installation

To install the code, run the following command:

```bash
conda create -n p28_py311_env python=3.11 -y
conda activate p28_py311_env

pip install -U pip
pip install "transformers>=4.45" "huggingface_hub>=0.23" accelerate safetensors torch
```

## Build prompts
miq-build-prompts --out src/mechinterp_qwen3/prompts/greater_than.jsonl --n 80 --seed 0
# Writes N prompts -> output path

## Run baseline (greedy)
miq-run-baseline --prompts src/mechinterp_qwen3/prompts/greater_than.jsonl --seed 0

## Capture activations (not needed anymore - step 2 crossed)
<!-- miq-capture-acts --run_path runs/RUN_ID --layers 4,12,20,28 --seed 0 -->