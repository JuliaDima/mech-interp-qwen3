#!/bin/bash
# Submit the full concept localization pipeline for all concepts in phases.json.
#
# Per concept, four dependent SLURM jobs are submitted in parallel chains:
#   1. run_concept  (GPU, 4h)  — delta extraction, causal analysis, multi-anchor
#   2. make_gif     (GPU, 2h)  — emergence trajectory GIF       [depends on 1]
#   3. sweep        (GPU, 2h)  — feature sweep at phase layers  [depends on 1]
#
# After ALL make_gif jobs finish:
#   4. plot_anchor_analysis  (no GPU needed, 30m)              [depends on all 2s]
#
# Usage
# -----
#   bash scripts/sbatch_concept_pipeline.sh              # submit all concepts
#   bash scripts/sbatch_concept_pipeline.sh --dry-run    # preview all sbatch submissions

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=false

[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

PHASES_JSON="runs/concept_localization/phases.json"
mkdir -p logs logs/gif_runs logs/sweep_runs

# ---------------------------------------------------------------------------
# submit <sbatch-args...>
#   Submits via sbatch --parsable and prints the job ID.
#   In dry-run mode, prints the command and echoes "0".
# ---------------------------------------------------------------------------
submit() {
    if $DRY_RUN; then
        echo "  [DRY] sbatch $*" >&2
        echo "0"
    else
        sbatch --parsable "$@"
    fi
}

CONCEPTS=$(python -c "
import json
print(' '.join(json.load(open('$PHASES_JSON')).keys()))
")

gif_job_ids=()

for CONCEPT in $CONCEPTS; do
    LAYERS=$(python -c "
import json
layers = json.load(open('$PHASES_JSON')).get('$CONCEPT', {}).get('T0', [])
print(','.join(map(str, layers)))
")
    if [ -z "$LAYERS" ]; then
        echo "[$CONCEPT] no T0 layers in phases.json — skipping"
        continue
    fi

    echo "=== $CONCEPT  (layers: $LAYERS) ==="

    # ── 1. run_concept ─────────────────────────────────────────────────────
    RUN_JID=$(submit \
        --job-name="run_${CONCEPT}" \
        scripts/sbatch_run.sh \
            python -m experiments.concept_localization.run_concept \
                --concept "$CONCEPT" \
                --causal \
                --n_feature_anchors 3 \
                --n 200)
    echo "  run_concept         → job $RUN_JID"

    # ── 2. make_gif (after run_concept) ────────────────────────────────────
    GIF_JID=$(submit \
        --job-name="gif_${CONCEPT}" \
        --dependency=afterok:"$RUN_JID" \
        scripts/sbatch_run.sh \
            python -m experiments.concept_localization.concept_emergence_gif.make_gif \
                --concept "$CONCEPT")
    echo "  make_gif            → job $GIF_JID  (after $RUN_JID)"
    gif_job_ids+=("$GIF_JID")

    # ── 3. sweep (after run_concept, layers from phases.json) ──────────────
    SWEEP_JID=$(submit \
        --job-name="sweep_${CONCEPT}" \
        --dependency=afterok:"$RUN_JID" \
        scripts/sbatch_run.sh \
            python scripts/sweeps/run_concept_sweep.py \
                --concept "$CONCEPT" \
                --layers "$LAYERS")
    echo "  sweep               → job $SWEEP_JID  (after $RUN_JID)"
done

# ── 4. plot_anchor_analysis after ALL make_gif jobs ────────────────────────
if [ ${#gif_job_ids[@]} -gt 0 ]; then
    GIF_DEP="afterok:$(IFS=:; echo "${gif_job_ids[*]}")"
    ANCHOR_JID=$(submit \
        --job-name="anchor_analysis" \
        --dependency="$GIF_DEP" \
        scripts/sbatch_run.sh \
            python -m experiments.concept_localization.plot_anchor_analysis)
    echo "=== plot_anchor_analysis → $ANCHOR_JID  (after all gif jobs)"
fi
