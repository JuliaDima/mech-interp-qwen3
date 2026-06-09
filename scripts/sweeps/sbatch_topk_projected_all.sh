#!/bin/bash
# Submit concept-specific PySR scripts for all concepts and all their anchor dirs.
#
# For each concept, looks for:
#   experiments/concept_localization/concept_fits/pysr_{concept}.py
# Skips the concept if no such script exists — no generic fallback.
#
# Usage
# -----
#   bash scripts/sweeps/sbatch_topk_projected_all.sh              # submit all
#   bash scripts/sweeps/sbatch_topk_projected_all.sh --dry-run    # preview

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

RUNS_DIR="runs/concept_localization"
TOP_K=15
N_PAIRS=200
R2_THRESHOLD=0.0
NITERATIONS=40

mkdir -p logs/topk_projected

submit() {
    if $DRY_RUN; then
        echo "  [DRY] sbatch $*" >&2
        echo "0"
    else
        sbatch --parsable "$@"
    fi
}

# Find all anchor dirs that have both results.json and deltas.pt
while IFS= read -r results_json; do
    anchor_dir="$(dirname "$results_json")"
    [[ -f "$anchor_dir/deltas.pt" ]] || continue

    # Derive concept from path: runs/concept_localization/<concept>/...
    rel="${anchor_dir#$RUNS_DIR/}"
    concept="${rel%%/*}"

    anchor_name="$(basename "$anchor_dir")"

    # Skip if no concept-specific PySR script exists
    CONCEPT_SCRIPT="experiments/concept_localization/concept_fits/pysr_${concept}.py"
    if [[ ! -f "$REPO_ROOT/$CONCEPT_SCRIPT" ]]; then
        echo "[$concept / $anchor_name] no concept script — skipping"
        continue
    fi

    # Skip if concept-specific pysr already done
    if [[ -f "$anchor_dir/sweep/edec_pysr/pysr_${concept}_summary.json" ]]; then
        echo "[$concept / $anchor_name] already done — skipping"
        continue
    fi

    echo "[$concept / $anchor_name] submitting ($CONCEPT_SCRIPT)..."
    JID=$(submit \
        --job-name="topk_${concept}_${anchor_name}" \
        --output="logs/topk_projected/${concept}_${anchor_name}_%j.out" \
        --error="logs/topk_projected/${concept}_${anchor_name}_%j.err" \
        --time=01:00:00 \
        scripts/sbatch_run.sh \
            python "$CONCEPT_SCRIPT" \
                --anchor_dir "$anchor_dir" \
                --top_k $TOP_K \
                --n_pairs $N_PAIRS \
                --niterations $NITERATIONS \
                --r2_threshold $R2_THRESHOLD)
    echo "  → job $JID"
done < <(find "$RUNS_DIR" -name "results.json" | sort)
