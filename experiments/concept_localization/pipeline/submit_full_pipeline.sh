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
#   bash scripts/submit_full_pipeline.sh                    # all concepts
#   bash scripts/submit_full_pipeline.sh carry              # carry only
#   bash scripts/submit_full_pipeline.sh carry gcd          # specific concepts
#   bash scripts/submit_full_pipeline.sh --dry-run carry    # preview for carry
#
# Optional env overrides
# ----------------------
#   GIF_N=100              pairs per template for make_gif    (default: 100)
#   ANCHOR_K=6             number of anchors to select        (default: 6)
#   ANCHOR_TIME=01:30:00   time limit per anchor pipeline     (default: 01:30:00)
#   N_PAIRS=100            pairs per template for run_concept (default: 100)
#   NULL_K=50              null permutation count             (default: 50)
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
)

# ── Defaults (override via env) ─────────────────────────────────────────────
GIF_N="${GIF_N:-100}"
ANCHOR_CANDIDATES="${ANCHOR_CANDIDATES:-6}"   # pipeline jobs submitted (ranked by mean_cos)
ANCHOR_K="${ANCHOR_K:-3}"                     # displayed anchors (top-k by combined score)
ANCHOR_TIME="${ANCHOR_TIME:-01:30:00}"
N_PAIRS="${N_PAIRS:-100}"
NULL_K="${NULL_K:-50}"
TEMPLATE="${TEMPLATE:-T0}"
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
$DRY_RUN && echo "  *** DRY RUN — no jobs submitted ***"
echo "========================================================"

for CONCEPT in "${CONCEPTS[@]}"; do
    echo ""
    echo "--- ${CONCEPT} ---"

    # ── 1. make_gif (T0, GIF_N pairs) ──────────────────────────────────────
    GIF_JID=$(submit \
        --job-name="gif_${CONCEPT}" \
        --time=00:20:00 \
        scripts/sbatch_run.sh \
            python -m experiments.concept_localization.concept_emergence_gif.make_gif \
                --concept "$CONCEPT" \
                --n "$GIF_N" \
                --template T0)
    echo "  make_gif            → job ${GIF_JID}"

    # ── 2. coordinator: select anchors + submit per-anchor jobs ────────────
    #    Runs after make_gif, reads emergence.npy, submits anchor_pipeline jobs.
    COORD_JID=$(submit \
        --job-name="coord_${CONCEPT}" \
        --time=00:10:00 \
        --dependency=afterok:"${GIF_JID}" \
        scripts/sbatch_run.sh \
            python experiments/concept_localization/pipeline/select_and_submit_anchors.py \
                --concept "$CONCEPT" \
                --candidates "$ANCHOR_CANDIDATES" \
                --k "$ANCHOR_K" \
                --template "$TEMPLATE" \
                --anchor_time "$ANCHOR_TIME" \
                --n "$N_PAIRS" \
                --null_k "$NULL_K")
    echo "  coordinator         → job ${COORD_JID}  (after ${GIF_JID})"
    echo "  anchor_pipeline x${ANCHOR_CANDIDATES} (display top ${ANCHOR_K})  → submitted by coordinator after it runs"
done

echo ""
echo "========================================================"
echo "Done. Each coordinator will submit ${ANCHOR_CANDIDATES} anchor_pipeline jobs (display top ${ANCHOR_K})"
echo "once its concept's make_gif job completes."
echo "========================================================"
