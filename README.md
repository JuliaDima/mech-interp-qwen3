# Mechanistic Circuits and Concept Representation in Qwen3-4B

[![pipeline status](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/badges/main/pipeline.svg)](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/-/pipelines)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This project is the submission of the MPhil Data Intensive Science research project at the University of Cambridge. The project report can be found under [Report](Report/thesis.pdf). The executive summary can be found under [Executive Summary](Report/ExecutiveSummary.pdf).

The thesis first reproduces two mechanistic interpretability experiments from [*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Lindsey et al., 2025) in the instruction-tuned open-source model **Qwen3-4B**. The first reproduction studies two-digit addition through attribution graphs, operand-grid feature scans, and teacher-forced accuracy analysis. The second tests multilingual antonym circuits by intervening on operation, operand, and output-language features across English, Chinese, and French.

The project then builds on these reproductions with a contrastive residual-stream method for studying concept representation, using matched prompt pairs defining a target computational predicate, such as carry detection, GCD divisibility, or residue-class membership. The pipeline computes layerwise delta trajectories, compares them with a permutation null baseline, validates anchors by activation patching, and projects the resulting directions through sparse transcoder features. 

The soft-prompting study tests whether learned continuous prefixes can modify model behaviour and whether those learned directions are interpretable relative to the residual-stream geometry.

**Visualising concept representations and attributions graphs**: [https://mechinterp-viz-94c364.uniofcam.dev/](https://mechinterp-viz-94c364.uniofcam.dev/)

<p align="center">
  <img src="docs/_static/images/gcd_concept_emergence.gif" alt="GCD concept emergence animation">
  <br>
  <sub><em>The animation shows concept localisation for GCD divisibility. Each frame steps to the next anchor position (ranked by contrastive signal strength as the model reads the prompt) and shows the transcoder features most aligned with the divisibility direction at that position, their individual activation profiles (x-axis shows `a mod 7` feature activations), and how the residual-stream direction stabilises across layers. </em></sub>
</p>

The visualiser is designed for inspecting how a contrastive predicate emerges as the model consumes a prompt, and for connecting the residual-stream geometry to sparse transcoder features and attribution-supported feature constellations.

- **Prompt timeline**: highlights the token position used as the current anchor.
- **Transcoder feature alignment**: shows top transcoder features aligned with the contrastive direction (opposing vs. supporting) and repeated feature-to-feature connections.
- **Delta trajectory and permutation null**: shows the raw and double-normalised residual-stream delta across model layers, and the excess of the observed signal above a permutation-null baseline.
- **Inter-layer direction similarity**: a layer-by-layer cosine-similarity heatmap of the delta directions, showing how quickly the anchor direction stabilises across depth.
- **Feature detail**: clicking a feature in the constellation opens its own activation-profile plot (mean positive/negative activation per modular input, or a 2D activation map).

The visualised also provides an interactive view of the attribution graph for an input prompt. [Here](https://mechinterp-viz-94c364.uniofcam.dev/?conceptRun=%2Fdata%2Fcarry_T0.json) is an example for the addition dataset.


## Table of Contents

- [Data Availability](#data-availability)
- [Visualisation](#visualisation)
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

Pre-computed experiment outputs (residual-stream delta arrays, transcoder projections, attribution graphs) are stored on RDS and are **not committed** to this repository. Pre-exported JSON for the interactive visualiser is committed under `data/` for the three main concepts (`carry`, `gcd`, `residue_class`).

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

### Reproducibility Experiments

The thesis begins with two reproductions of the circuit-tracing methodology from Lindsey et al. in Qwen3-4B.

#### Addition attribution graphs

Builds node-ablation attribution graphs and recovers lookup-like addition circuits.

```bash
miq attribute -t mwhanna/qwen3-4b-transcoders -p "calc: 36+59=" \
    --slug addition_36_59 --graph_file_dir graphs/
```

#### Multilingual antonym interventions

Tests whether operation, operand, and output-language components can be independently suppressed and injected across English, Chinese, and French. The experiment code is under `experiments/multilingual_circuits/`.

---

### Concept Localisation

Finds where and how a contrastively specified concept is encoded across layers and token positions, using residual-stream deltas projected onto transcoder features.

**Main thesis concepts**:
`carry`, `gcd`, `residue_class`

Additional dataset definitions are included for arithmetic, logic, physics, and language concepts, and can be 
found [here](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/-/tree/main/experiments/concept_localization/concept_datasets?ref_type=heads).

#### Run a single concept

```bash
# Full pipeline (GPU required — submit via Slurm)
sbatch scripts/sbatch_run.sh python -m experiments.concept_localization.pipeline.run_concept \
    --concept carry
```

Output is found in `runs/concept_localization/{concept}/`.

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

#### Visualiser exports

Committed visualiser exports are available as `data/carry_T0.concept.json`, `data/gcd_T0.concept.json`, and `data/residue_class_T0.concept.json`. These are lightweight summaries of the larger run directories on RDS and contain prompt tokens, anchor trajectories, null baselines, top transcoder features, and feature-constellation edges.

---

### Batch Attribution Graph Jobs

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
    concept_emergence_gif/       # model-based token-consumption GIF renderer
  addition/                      # Anthropic reproduction (Fourier, operand plots)
  multilingual_circuits/         # operation, operand, language interventions
  soft_prompt/                   # knowledge editing via soft prompt optimisation

scripts/
  sbatch_run.sh                  # universal Slurm wrapper
  submit_concept_attribution_graphs.py
  render_gcd_readme_gif.py       # README GIF, captured from the live visualiser via Playwright

Report/               # LaTeX report source
  thesis.tex                     # main document
  Pages/                         # chapter .tex files and figures
  cam-thesis.cls                 # Cambridge thesis class

data/                            # lightweight exports for README and visualiser
docs/_static/images/             # README and documentation images
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

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/) — see the [LICENSE](LICENSE) file for details.

---

## Note on the Use of AI Tools

[Claude Code](https://claude.ai/code) (Anthropic, claude-sonnet-4-6) was used as an AI coding assistant throughout this project. Its use is described below.

#### Report writing and editing

Claude Code was used to assist with drafting, restructuring, and proofreading sections of the LaTeX thesis report and the executive summary. This included suggesting alternative phrasings, trimming overlong sections, and checking mathematical notation for consistency. All scientific claims, results, and interpretations are the author's own.

#### Code assistance

Claude Code (Anthropic) and GitHub Copilot (OpenAI Codex) were used as coding assistants throughout the project, primarily for code style, boilerplate, and software engineering practice: docstrings, type annotations, repetitive utility functions, and minor debugging. The overall experiment design, analysis pipeline structure, result verification, and substantive implementation choices are the author's own.

Claude Code was additionally used for LaTeX debugging (Unicode character declarations, `\middle` scoping, figure path resolution across compilation directories).

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

This project is implemented by [Elisabeta-Iulia (Julia) Dima](mailto:eid23@cam.ac.uk) at the University of Cambridge, supervised by Dr Miles Cranmer and Dr Alessandro Favero.

June 2026
