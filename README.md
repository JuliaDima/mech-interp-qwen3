# Mechanistic Interpretability of Qwen3-4B-Instruct

**Documentation**: [https://eid23-ab47b6.uniofcam.dev/](https://eid23-ab47b6.uniofcam.dev/)

**Visualizing the attribution graph**: [https://mechinterp-viz-94c364.uniofcam.dev/](https://mechinterp-viz-94c364.uniofcam.dev/)

[![pipeline status](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/badges/main/pipeline.svg)](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/-/pipelines) [![coverage report](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/badges/main/coverage.svg)](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/-/jobs)

This repository contains the pipeline for investigating the internal circuits and features of the **Qwen3-4B-Instruct** model using Sparse Autoencoders (SAEs) and Attribution Graphs.

## 🚀 Quick Start

### Installation

```bash
conda create -n mechinterp python=3.11 -y
conda activate mechinterp
pip install -e ".[docs,test]"
```

### 1. Dataset Generation
Generate controlled datasets (e.g., addition) with model statistics.
```bash
# Example generating 441 addition prompts (loads defaults from config.yaml)
miq generate-dataset --max_value 20 --output_path data/addition_20.jsonl
```

### 2. Attribution Analysis
Construct pruned dependency graphs from inputs through features to logits.
```bash
miq attribute -t mwhanna/qwen3-4b-transcoders -p "calc: 36+59=" --slug addition_36_59 --graph_file_dir graphs/
```

### 3. Case Study: Addition Reproduction
Run the end-to-end Anthropic addition case study reproduction.
```bash
python experiments/addition/run.py --all
```

### 4. Stitching Experiment (SAE-mediated)
Transfer arithmetic circuits from a small specialized model into Qwen3-4B.
```bash
python experiments/stitching/run.py --all
```

## 📖 Key Documentation

For in-depth guides, visit the [Documentation Site](https://eid23-ab47b6.uniofcam.dev/) or explore the `docs/` folder:

- **[Carry Discovery](docs/carry_discovery.rst)**: Scientific overview of the addition circuit reproduction.
- **[Stitching Experiment](docs/stitching_experiment.rst)**: Guide to transferring circuits between models using SAEs.
- **[Dataset Generation](docs/dataset_generation.rst)**: Guide to sampling strategies and teacher-forcing.
- **[Configuration System](docs/configuration.rst)**: Details on the hierarchical `config.yaml` architecture.
- **[Visualization Guide](VISUALIZATION_GUIDE.md)**: 6 publication-quality figure types for behavior analysis.

## 🛠 Project Architecture

- `src/mechinterp_qwen3/`: Core package containing attribution logic and dataset generation.
- `experiments/addition/`: Specialized module for the Anthropic reproduction case study.
- `docs/`: Sphinx site source files.
- `scripts/`: Production sbatch and utility scripts.

## ⚙️ Configuration System

The project uses a centralized configuration system centered around the root **[config.yaml](config.yaml)** file:

- **Single Source of Truth**: This file contains all project-wide defaults (architecture, batch sizes, seeds, etc.).
- **Automatic Usage**: All `miq` CLI commands and scripts (including `sbatch_run.sh`) load this file by default.
- **Overrides**: You can override any value by passing an explicit argument (e.g., `--dtype bfloat16`) or by providing a custom YAML file via the `--config` flag.

---
