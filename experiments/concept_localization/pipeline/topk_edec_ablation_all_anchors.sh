#!/bin/bash
# Thin wrapper around topk_edec_ablation_compare.py: joint-ablate, on every
# anchor of carry/gcd/residue_class/prime, two top-K feature-selection
# strategies (default dec+enc_dec score vs. standardised Cohen's d), and dump
# a combined baseline/ablated-A/ablated-B token comparison dataset. The
# ranking, anchor bookkeeping, and multi-config token dump all live in
# topk_edec_ablation_compare.py now -- this script just forwards args to it.
#
# Usage:
#   bash experiments/concept_localization/pipeline/topk_edec_ablation_all_anchors.sh
#   bash experiments/concept_localization/pipeline/topk_edec_ablation_all_anchors.sh --concepts gcd --top_k 10
set -euo pipefail

python -m experiments.concept_localization.pipeline.topk_edec_ablation_compare "$@"
