#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

TOP_N="${TOP_N:-30}"
LAYERS="${LAYERS:-all}"
SECTION="${SECTION:-centered}"
CSV_DIR="${CSV_DIR:-${PROJECT_ROOT}}"

if [ "$#" -gt 0 ]; then
  csvs=("$@")
else
  mapfile -t csvs < <(find "${CSV_DIR}" -maxdepth 1 -type f -name 'L*_F*.csv' | sort)
fi

if [ "${#csvs[@]}" -eq 0 ]; then
  echo "No CSV files found in ${CSV_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
for csv in "${csvs[@]}"; do
  stem="$(basename "${csv}" .csv)"
  echo "Submitting ${stem}"
  sbatch --job-name="closest_${stem}" scripts/sbatch_run.sh \
    python experiments/concept_localization/plots/find_and_plot_closest_csvs_all_anchors.py \
      --csv "${csv}" \
      --layers "${LAYERS}" \
      --top_n "${TOP_N}" \
      --section "${SECTION}" \
      --device cuda \
      --skip_existing
done
