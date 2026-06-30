# The Geometry of Concept Representation in Open-Source Language Models

[![pipeline status](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/badges/main/pipeline.svg)](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/-/pipelines)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This project is associated with the submission of the MPhil Data Intensive Science research project at the University of Cambridge. The associated project report can be found under [Report](Report/thesis.pdf). The associated executive summary can be found under [Executive Summary](Report/ExecutiveSummary.pdf).

**Visualisation**: [https://mechinterp-viz-94c364.uniofcam.dev/](https://mechinterp-viz-94c364.uniofcam.dev/)

The primary objective of this project is to reproduce the circuit-tracing methodology from [*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Lindsey et al., 2025) on an instruction-tuned open-source model, **Qwen3-4B**, and to extend this work by introducing a new method for localising abstract concepts within the residual stream. The concept localisation method operates on contrastive prompt pairs, computes layerwise delta trajectories, and validates geometric anchors via activation patching and sparse transcoder feature projection. It is applied to three arithmetic tasks of increasing representational complexity — carry detection, GCD divisibility, and residue class membership — and is designed to generalise to any domain where concepts can be expressed through matched contrastive pairs.

## Table of Contents

- [Data Availability](#data-availability)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Cluster Usage](#cluster-usage)
- [Model and Transcoders](#model-and-transcoders)
- [Support](#support)
- [License](#license)
- [Project Status](#project-status)
- [Note on the Use of AI Tools](#note-on-the-use-of-ai-tools)
- [Authors and Acknowledgment](#authors-and-acknowledgment)

---

## Data Availability

Model weights and transcoders are loaded from Hugging Face Hub and cached on the HPC at:

```
/rds/user/eid23/hpc-work/p28/cache/hf/hub/
```

Pre-computed experiment outputs (residual-stream delta arrays, transcoder projections, attribution graphs) are stored on RDS and are **not committed** to this repository. Pre-exported JSON for the interactive visualiser is committed under `viz/data/` for the three main concepts (`carry`, `gcd`, `residue_class`).

---

## Installation

### Requirements

- Python 3.11 or higher
- Conda (for environment management)
- CUDA-capable GPU (required for inference; plotting/export runs on CPU)

### Setup

1. **Clone the repository:**

    ```bash
    git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23
    cd eid23
    ```

2. **Create and activate the environment:**

    ```bash
    conda create -n mechinterp python=3.11 -y
    conda activate mechinterp
    ```

3. **Install the package:**

    ```bash
    pip install -e ".[docs,test]"
    ```

4. **Set HuggingFace Hub to offline mode** (if using cached weights on HPC):

    ```bash
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    ```

---

## Usage

### Concept Localisation (main experiment)

Finds where and how a binary concept is encoded across layers and token positions, using contrastive residual-stream deltas projected onto transcoder features.

**17 concepts** across arithmetic, logic, physics, and language:
`carry`, `gcd`, `residue_class`, `decimal_termination`, `perfect_square`, `geometric_series`, `balanced_parentheses`, `negation_scope`, `transitive_ordering`, `triangle_inequality`, `causal_direction`, `conservation`, `momentum_conservation`, `doppler_shift`, `wave_interference`, `dot_product_sign`, `syllogism`

#### Run a single concept

```bash
# Full pipeline (GPU required — submit via Slurm)
sbatch scripts/sbatch_run.sh python -m experiments.concept_localization.pipeline.run_concept \
    --concept carry
```

Output lands in `runs/concept_localization/{concept}/`.

#### Run all main concepts

```bash
for concept in carry gcd residue_class; do
    sbatch scripts/sbatch_run.sh python -m experiments.concept_localization.pipeline.run_concept \
        --concept $concept
done
```

#### Plot per-anchor summary grids (CPU, no GPU needed)

```bash
python -m experiments.concept_localization.plots.plot_emergence_per_anchor \
    --concept carry --template T0 --top_k 6 --thesis
```

#### Export for the interactive visualiser

```bash
cd viz
python scripts/export_concept_run.py \
    ../runs/concept_localization/carry/carry_T0 > data/carry_T0.json
```

---

### Attribution Graph Reproduction (Anthropic)

Reproduces the circuit-tracing methodology from *On the Biology of a Large Language Model*. Builds node-ablation attribution graphs and recovers interpretable addition circuits in Qwen3-4B.

```bash
miq attribute -t mwhanna/qwen3-4b-transcoders -p "calc: 36+59=" \
    --slug addition_36_59 --graph_file_dir graphs/
```

#### Submit all attribution graph jobs

```bash
python scripts/submit_concept_attribution_graphs.py
```

---

### Soft Prompt / Knowledge Editing

Trains a learned $k=10$ token prefix on frozen base weights and evaluates whether input-space perturbations align with concept directions identified in the localisation step.

```bash
python experiments/soft_prompt/run.py
```

Results are saved as a summary table and per-task checkpoint under `runs/soft_prompt/`.

---

### Compile the Report

The thesis is written in LaTeX and compiled from `Report/`:

```bash
cd Report
make all
```

The executive summary compiles as a standalone document from `Report/`:

```bash
cd Report
pdflatex ExecutiveSummary && bibtex ExecutiveSummary && \
pdflatex ExecutiveSummary && pdflatex ExecutiveSummary
```

---

## Project Structure

```
src/mechinterp_qwen3/            # core package (pip install -e .)
  attribution_model.py           # AttributionModel: Qwen3-4B + transcoders
  interventions.py               # feature ablation/injection hooks
  transcoder/                    # SingleLayerTranscoder, CrossLayerTranscoder
  probe/                         # linear probes on residual stream
  utils/                         # hf_utils, model_utils, token_utils, config_utils

experiments/
  concept_localization/          # main experiment pipeline
    concept_datasets/            # one dataset file per concept (17 concepts)
    pipeline/                    # run_concept.py, delta_feature_projections.py
    plots/                       # plot_emergence_per_anchor.py
  addition/                      # Anthropic reproduction (Fourier, operand plots)
  soft_prompt/                   # knowledge editing via soft prompt optimisation

scripts/
  sbatch_run.sh                  # universal Slurm wrapper
  submit_concept_attribution_graphs.py

Report/               # LaTeX report source
  thesis.tex                     # main document
  Pages/                         # chapter .tex files and figures
  cam-thesis.cls                 # Cambridge thesis class

ExecutiveSummary.tex             # standalone executive summary (LaTeX)
viz/                             # interactive visualiser source + pre-exported JSON
runs/                            # all outputs (on RDS, not committed)
config.yaml                      # project-wide defaults
```

---

## Cluster Usage

All inference jobs require GPU and run via Slurm:

```bash
sbatch scripts/sbatch_run.sh python <command>
```

CPU-only jobs (plotting, export, report compilation) can run directly on the login node.

---

## Model and Transcoders

- **Model**: `Qwen/Qwen3-4B` — 36 transformer layers, d\_model = 2560, 32 attention heads (8 KV heads), RoPE positional encoding, instruction-tuned
- **Transcoders**: `mwhanna/qwen3-4b-transcoders` — one sparse autoencoder per MLP layer, trained to reconstruct MLP output in terms of interpretable features; accessed via `AttributionModel.transcoders[layer]`

---

## Support

For questions or feedback, please contact [eid23@cam.ac.uk](mailto:eid23@cam.ac.uk).

---

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/) — see the [LICENSE](LICENSE) file for details.

---

## Project Status

The project is complete and ready for submission. All experiment pipelines, the LaTeX report, and the executive summary have been finalised.

---

## Note on the Use of AI Tools

[Claude Code](https://claude.ai/code) (Anthropic, claude-sonnet-4-6) was used as an AI coding assistant throughout this project. Its use is described below.

#### Report writing and editing

Claude Code was used to assist with drafting, restructuring, and proofreading sections of the LaTeX thesis report and the executive summary. This included suggesting alternative phrasings, trimming overlong sections, and checking mathematical notation for consistency. All scientific claims, results, and interpretations are the author's own.

#### Code assistance

Claude Code was used to assist with:
- Debugging LaTeX compilation errors (Unicode character declarations, `\middle` scoping, figure path resolution across compilation directories)
- Reviewing and suggesting minor edits to Python scripts

All core experiment design, model analysis code, and result interpretation were written and validated by the author independently.

#### Example interaction — LaTeX path fix

**Prompt:** The `cam-thesis.cls` file hardcodes `Pages/Figures/CollegeShields/CUni.pdf` but compilation from a parent directory fails to resolve this path. How can I make the path auto-detect the working directory?

**Claude Code suggestion:**
```latex
\IfFileExists{Pages/Figures/CollegeShields/CUni.pdf}%
  {\def\cam@CUniPath{Pages/Figures/CollegeShields/CUni.pdf}}%
  {\def\cam@CUniPath{Report/Pages/Figures/CollegeShields/CUni.pdf}}
```

**Modification used:** The suggestion was adopted directly and integrated into `cam-thesis.cls` alongside an equivalent block for `dis_logo.pdf`.

---

## Authors and Acknowledgment

This project is maintained by [Elisabeta-Iulia (Julia) Dima](mailto:eid23@cam.ac.uk) at the University of Cambridge, supervised by Dr Miles Cranmer and Dr Alessandro Favero.

June 2026
