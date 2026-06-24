# Mechanistic Interpretability of Qwen3-4B

**Visualisation**: [https://mechinterp-viz-94c364.uniofcam.dev/](https://mechinterp-viz-94c364.uniofcam.dev/)

[![pipeline status](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/badges/main/pipeline.svg)](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/eid23/-/pipelines)

MPhil thesis project. Investigates how abstract concepts are encoded in the residual stream of **Qwen3-4B** using contrastive pair analysis, transcoder feature projection, and causal validation. Also includes a reproduction of Anthropic's *On the Biology of a Large Language Model* circuit-tracing methodology and a soft-prompt knowledge editing experiment.

---

## Installation

```bash
conda create -n mechinterp python=3.11 -y
conda activate mechinterp
pip install -e ".[docs,test]"
```

All jobs run with HF Hub offline (models cached at `/rds/user/eid23/hpc-work/p28/cache/hf/hub/`):

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

---

## Concept Localisation (main experiment)

Finds where and how a binary concept is encoded across layers and token positions, using contrastive residual-stream deltas projected onto transcoder features.

**17 concepts** across arithmetic, logic, physics, and language:
`carry`, `gcd`, `residue_class`, `decimal_termination`, `perfect_square`, `geometric_series`, `balanced_parentheses`, `negation_scope`, `transitive_ordering`, `triangle_inequality`, `causal_direction`, `conservation`, `momentum_conservation`, `doppler_shift`, `wave_interference`, `dot_product_sign`, `syllogism`

### Run a concept

```bash
# Full pipeline (GPU required — submit via Slurm)
sbatch scripts/sbatch_run.sh python -m experiments.concept_localization.pipeline.run_concept \
    --concept carry

# Run all concepts
for concept in carry gcd residue_class; do
    sbatch scripts/sbatch_run.sh python -m experiments.concept_localization.pipeline.run_concept \
        --concept $concept
done
```

Output lands in `runs/concept_localization/{concept}/`.

### Plot per-anchor summary grids (no GPU needed)

```bash
python -m experiments.concept_localization.plots.plot_emergence_per_anchor \
    --concept carry --template T0 --top_k 6 --thesis
```

### Export for the visualiser

```bash
cd mechinterp-qwen3-viz
python scripts/export_concept_run.py \
    ../runs/concept_localization/carry/carry_T0 > data/carry_T0.json
```

Pre-exported JSON for `carry`, `gcd`, and `residue_class` is committed to `mechinterp-qwen3-viz/data/`.

---

## Attribution Graph Reproduction (Anthropic)

Reproduces the circuit-tracing methodology from *On the Biology of a Large Language Model*.

```bash
miq attribute -t mwhanna/qwen3-4b-transcoders -p "calc: 36+59=" \
    --slug addition_36_59 --graph_file_dir graphs/
```

---

## Soft Prompt / Knowledge Editing

```bash
python experiments/soft_prompt/run.py
```

---

## Project Structure

```
src/mechinterp_qwen3/          # core package (pip install -e .)
  attribution_model.py         # AttributionModel: Qwen3-4B + transcoders
  interventions.py             # feature ablation/injection hooks
  transcoder/                  # SingleLayerTranscoder, CrossLayerTranscoder
  probe/                       # linear probes on residual stream
  utils/                       # hf_utils, model_utils, token_utils, config_utils

experiments/
  concept_localization/        # main experiment pipeline
    concept_datasets/          # one dataset file per concept (17 concepts)
    pipeline/                  # run_concept.py, delta_feature_projections.py
    plots/                     # plot_emergence_per_anchor.py
  addition/                    # Anthropic reproduction (Fourier, operand plots)
  soft_prompt/                 # knowledge editing via soft prompt optimisation

scripts/
  sbatch_run.sh                # universal Slurm wrapper
  submit_concept_attribution_graphs.py

runs/                          # all outputs (on RDS, not committed)
config.yaml                    # project-wide defaults
```

---

## Cluster Usage

All inference jobs require GPU and run via Slurm:

```bash
sbatch scripts/sbatch_run.sh python <command>
```

CPU-only jobs (plotting, export) can run directly on the login node.

---

## Model and Transcoders

- **Model**: `Qwen/Qwen3-4B` — 36 layers, d_model=2560, 32 attention heads (8 KV), RoPE
- **Transcoders**: `mwhanna/qwen3-4b-transcoders` — one sparse autoencoder per MLP layer, trained to reconstruct MLP output; accessed via `AttributionModel.transcoders[layer]`
