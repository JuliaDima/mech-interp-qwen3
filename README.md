# Mechanistic Interpretability of Qwen3-4B-Instruct

[![pipeline status](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/badges/main/pipeline.svg)](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/-/pipelines) [![coverage report](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/badges/main/coverage.svg)](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/-/jobs)

This repository contains the code for the paper "Mechanistic Interpretability of Qwen3-4B-Instruct". The code is written in Python and is based on the Hugging Face transformers library. The attribution graph can be visualized at [https://mechinterp-viz-94c364.uniofcam.dev/](https://mechinterp-viz-94c364.uniofcam.dev/).

## Project goals

The scope of this project is to investigate the mechanistic interpretability of Qwen3-4B-Instruct. We aim to identify and understand the circuits responsible for specific behaviours in the model. We will use sparse autoencoders (SAEs) to extract interpretable features from the model's activations, and then build a pruned dependency graph to visualise the flow of information through the model. Finally, we will validate our findings through inhibition or swap-in interventions.

The focus is on independent implementation and documentation, with a standalone repository, in line with Research Computing guidance that projects investigate reproducibility rather than merely match numbers.

The workflow: run Qwen3-4B-Instruct on a small prompt set for each chosen behaviour, capture activations for a small subset of layers, use sparse autoencoders (SAEs) to obtain interpretable features, build a pruned dependency graph from inputs through SAE features to decisive logits, and validate with inhibition or swap-in style interventions. Negative or partial reproductions are acceptable if analysed rigorously. Project goals Main goals of the project:

1. Setup and baselines. Run Qwen3-4B-Instruct, prepare prompts for one to two chosen be- haviours, and record baseline outputs with fixed seeds. haviours, and record baseline outputs with fixed seeds.

2. (**Not needed anymore, branch to 2.1 instead**) Train sparse autoencoders. Collect MLP activations on a small prompt set for a few selected layers, then train lightweight SAEs per layer (clear train/validation split, bottleneck size reported). Map discovered features to tokens or behaviours.

2.1. Start with one of the transcoder models linked in this repository: https://github.com/safety-research/circuit-tracer?tab=readme-ov-file. There are Qwen transcoders linked in the "Available Transcoders" section. Furthermore, you can explore different scales of the Qwen3 models if you so wish (such as 0.6B through 14B) - there are transcoders linked for each of these. The repository has the transcoder class which you will need to actually load the modules (https://github.com/safety-research/circuit-tracer/tree/main/circuit_tracer/transcoder).

3. Attribution-style graph. Construct a pruned dependency graph from input features through SAE features to decisive logits, mirroring the key publication’s diagrams at small scale.

4. Validation by intervention. Perform inhibition or swap-in interventions on upstream feature groups and quantify effects on downstream groups and model outputs. Report successes and failures carefully.

5. Reproducibility pack. Release a public repository with scripts, environment, seeds, graphs, and a short comparison to the key publication’s narrative and figures.

These should be the key steps that students reproduce in their project. Extension directions:

1. Science-related circuits. Search for and analyse behaviours tied to scientific reasoning or numeracy, where practical.

Reading List:

1. On the Biology of a Large Language Model (follow citations within the post for context). Data Access Instructions on how to access the data with necessary links, if any.
2. Model: Qwen/Qwen3-4B-Instruct-2507 (open weights; feasible on a modern laptop or de- partmental GPU).
3. Prompts: Small synthetic sets adapted from the key publication’s examples, plus minor variants for robustness.
4. Code: Student maintains an independent repository implementing activation capture, SAE training, attribution-graph construction, and interventions.
cation's diagrams at small scale.

## Installation

To install the code, run the following command:

```bash
conda create -n p28_py311_env python=3.11 -y
conda activate p28_py311_env

pip install -U pip
pip install -e ".[docs,test]"
```
## Build attribution graph

We implement attribution graph construction ourselves rather than using circuit-tracer's high-level API:
- **Forward pass**: Capture MLP activations and extract SAE features using transcoders
- **Attribution**: Compute gradients from output logits to SAE features (∂logit/∂feature)
- **Graph building**: Create nodes (tokens, features, logits) and edges (attributions)
- **Pruning**: Remove low-attribution nodes and edges

```bash
# Build attribution graph for a specific example
miq-build-graph \
  --prompt "You are solving a simple comparison task. Two numbers are given: A and B. Answer with a single character: 'A' if A is larger, otherwise 'B'. A = 864, B = 394, Answer: " \
  --slug gt_864_394 \
  --layers 4,12,20 \
  --graph_dir graphs \
  --max_n_logits 2 \
  --use_patching 0 \
  --node_threshold 0.8 \
  --edge_threshold 0.85
```

## Dataset Generation

Before building graphs, you often need a controlled dataset of prompts (e.g., addition problems) with ground-truth model statistics.

```bash
# Generate a grid of addition problems (0-20)
miq generate-dataset \
  --max_value 20 \
  --output_path data/addition_20.jsonl \
  --sampling_strategy grid \
  --templates T0
```

For more details, see [DATASET_GENERATION_README.md](DATASET_GENERATION_README.md) and [QUICKSTART_DATASET.md](QUICKSTART_DATASET.md).
