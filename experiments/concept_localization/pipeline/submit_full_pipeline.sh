#!/bin/bash
# Submit the full concept localization pipeline for all (or specified) concepts.
#
# Per concept, two dependent SLURM jobs are submitted:
#   1. make_gif     (GPU, 20 min)  — runs T0, 100 pairs; writes emergence.npy
#   2. coordinator  (GPU,  5 min)  — reads emergence.npy, selects top-6 anchors,
#                                    submits one anchor_pipeline job per anchor
#
# Each anchor_pipeline job (GPU, 60 min) then runs:
#   run_concept + null permutation + transcoder sweep + cluster analysis
#   sequentially.
#
# After all of a concept's anchor jobs complete, the coordinator submits one
# final peak-feature plot job (plot_sweep_peak_features.py) that aggregates
# every anchor into per-anchor top-feature and Fourier-mode heatmaps.
#
# All concept pipelines run in parallel across the cluster.
#
# Usage
# -----
#   bash experiments/concept_localization/pipeline/submit_full_pipeline.sh                    # all concepts
#   bash experiments/concept_localization/pipeline/submit_full_pipeline.sh carry              # carry only
#   bash experiments/concept_localization/pipeline/submit_full_pipeline.sh carry gcd          # specific concepts
#   bash experiments/concept_localization/pipeline/submit_full_pipeline.sh --dry-run carry    # preview for carry
#
# Optional env overrides
# ----------------------
#   GIF_N=100              pairs per template for make_gif    (default: 100)
#   ANCHOR_K=6             number of anchors to select        (default: 6)
#   ANCHOR_TIME=01:30:00   time limit per anchor pipeline     (default: 01:30:00)
#   N_PAIRS=100            pairs per template for run_concept (default: 100)
#   NULL_K=20              null permutation count             (default: 20)
#   TEMPLATE=T0            template for per-anchor analyses   (default: T0)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ── All registered concepts ─────────────────────────────────────────────────
ALL_CONCEPTS=(
    carry
    gcd
    residue_class
    transitive_ordering
    conservation
    causal_direction
    negation_scope
    balanced_parentheses
    decimal_termination
    doppler_shift
    dot_product_sign
    geometric_series
    momentum_conservation
    perfect_square
    syllogism
    triangle_inequality
    wave_interference
    prime
)

# ── Defaults (override via env) ─────────────────────────────────────────────
GIF_N="${GIF_N:-100}"
ANCHOR_CANDIDATES="${ANCHOR_CANDIDATES:-6}"   # pipeline jobs submitted (ranked by mean_cos)
ANCHOR_K="${ANCHOR_K:-3}"                     # displayed anchors (top-k by combined score)
ANCHOR_TIME="${ANCHOR_TIME:-01:30:00}"
N_PAIRS="${N_PAIRS:-100}"
NULL_K="${NULL_K:-20}"
TEMPLATE="${TEMPLATE:-T0}"
MODEL_CONFIG="${MODEL_CONFIG:-}"              # path to model config YAML; empty = default
MODEL_PROFILE="${MODEL_PROFILE:-}"            # profile name to activate (e.g. qwen3_0_6b_lowl0)
CONCEPT_SUFFIX="${CONCEPT_SUFFIX:-}"          # appended to concept name in output dirs only (e.g. _qwen3_0_6b)
# ── Arg parsing ─────────────────────────────────────────────────────────────
DRY_RUN=false
CONCEPTS=()

for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    else
        CONCEPTS+=("$arg")
    fi
done

# Default to all concepts if none specified
if [[ ${#CONCEPTS[@]} -eq 0 ]]; then
    CONCEPTS=("${ALL_CONCEPTS[@]}")
fi

mkdir -p logs

# ---------------------------------------------------------------------------
# submit <sbatch-args...>
#   Submits via sbatch --parsable and returns the job ID.
#   In dry-run mode, prints the command to stderr and returns "0".
# ---------------------------------------------------------------------------
submit() {
    if $DRY_RUN; then
        echo "  [DRY] sbatch $*" >&2
        echo "0"
    else
        sbatch --parsable "$@"
    fi
}

echo "========================================================"
echo "Submitting full pipeline for ${#CONCEPTS[@]} concept(s)"
echo "  GIF_N=${GIF_N}  ANCHOR_K=${ANCHOR_K}  ANCHOR_TIME=${ANCHOR_TIME}"
echo "  N_PAIRS=${N_PAIRS}  NULL_K=${NULL_K}"
echo "  TEMPLATE=${TEMPLATE}"
[[ -n "$MODEL_CONFIG" ]]   && echo "  MODEL_CONFIG=${MODEL_CONFIG}"
[[ -n "$MODEL_PROFILE" ]]  && echo "  MODEL_PROFILE=${MODEL_PROFILE}"
[[ -n "$CONCEPT_SUFFIX" ]] && echo "  CONCEPT_SUFFIX=${CONCEPT_SUFFIX}"
$DRY_RUN && echo "  *** DRY RUN — no jobs submitted ***"
echo "========================================================"

for CONCEPT in "${CONCEPTS[@]}"; do
    OUT_CONCEPT="${CONCEPT}${CONCEPT_SUFFIX}"
    echo ""
    echo "--- ${CONCEPT} → ${OUT_CONCEPT} ---"

    # ── 1. make_gif (T0, GIF_N pairs) ──────────────────────────────────────
    GIF_ARGS=(--concept "$CONCEPT" --n "$GIF_N" --template T0 --out_concept "$OUT_CONCEPT")
    [[ -n "$MODEL_CONFIG" ]]  && GIF_ARGS+=(--model_config "$MODEL_CONFIG")
    [[ -n "$MODEL_PROFILE" ]] && GIF_ARGS+=(--profile "$MODEL_PROFILE")
    GIF_JID=$(submit \
        --job-name="gif_${OUT_CONCEPT}" \
        --time=00:20:00 \
        scripts/sbatch_run.sh \
            python -m experiments.concept_localization.concept_emergence_gif.make_gif \
                "${GIF_ARGS[@]}")
    echo "  make_gif            → job ${GIF_JID}"

    # ── 2. coordinator: select anchors + submit per-anchor jobs ────────────
    #    Runs after make_gif, reads emergence.npy, submits anchor_pipeline jobs.
    COORD_ARGS=(
        --concept "$CONCEPT"
        --out_concept "$OUT_CONCEPT"
        --candidates "$ANCHOR_CANDIDATES"
        --k "$ANCHOR_K"
        --template "$TEMPLATE"
        --anchor_time "$ANCHOR_TIME"
        --n "$N_PAIRS"
        --null_k "$NULL_K"
    )
    [[ -n "$MODEL_CONFIG" ]]  && COORD_ARGS+=(--model_config "$MODEL_CONFIG")
    [[ -n "$MODEL_PROFILE" ]] && COORD_ARGS+=(--profile "$MODEL_PROFILE")
    COORD_JID=$(submit \
        --job-name="coord_${OUT_CONCEPT}" \
        --time=00:10:00 \
        --dependency=afterok:"${GIF_JID}" \
        scripts/sbatch_run.sh \
            python experiments/concept_localization/pipeline/select_and_submit_anchors.py \
                "${COORD_ARGS[@]}")
    echo "  coordinator         → job ${COORD_JID}  (after ${GIF_JID})"
    echo "  anchor_pipeline x${ANCHOR_CANDIDATES} (display top ${ANCHOR_K})  → submitted by coordinator after it runs"
done

echo ""
echo "========================================================"
echo "Done. Each coordinator will submit ${ANCHOR_CANDIDATES} anchor_pipeline jobs (display top ${ANCHOR_K})"
echo "once its concept's make_gif job completes."
echo "========================================================"
