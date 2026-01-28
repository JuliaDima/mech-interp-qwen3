# Mechanistic Interpretability of Qwen3-4B-Instruct

This repository contains the code for the paper "Mechanistic Interpretability of Qwen3-4B-Instruct". The code is written in Python and is based on the Hugging Face transformers library.


## Installation

To install the code, run the following command:

```bash
pip install -e .
```

## Build prompts
miq-build-prompts --out src/mechinterp_qwen3/prompts/greater_than.jsonl --n 80 --seed 0

## Run baseline (greedy)
miq-run-baseline --prompts src/mechinterp_qwen3/prompts/greater_than.jsonl --seed 0

## Capture activations
# Replace RUN_ID with the folder created in runs/
miq-capture-acts --run_path runs/RUN_ID --layers 4,12,20,28 --seed 0