"""Compute concept localization metric across all concepts.

For each concept, computes what fraction of top-k features are "well-localized":
- Concept-localized: high activation on pos, low on neg
- Anti-localized: high activation on neg, low on pos

A feature is well-localized if ≥threshold% of examples match the expected pattern.

Usage:
    python scripts/sweeps/compute_localization_metric.py --top_k 200 --consistency_threshold 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_BASE = _REPO_ROOT / "runs" / "concept_localization"

import matplotlib.pyplot as plt

import experiments.plot_style as ps
from scripts.sweeps.run_concept_sweep import CONCEPTS

def _localization_for_sweep(
    sweep_dir: Path, top_k: int, consistency_threshold: float
) -> dict | None:
    """Compute localization for a single anchor sweep dir. Returns None on failure."""
    ranked_path = sweep_dir / "sweep_ranked.json"
    activations_path = sweep_dir / "sweep_activations.npz"
    if not ranked_path.exists() or not activations_path.exists():
        return None
    try:
        with open(ranked_path) as f:
            ranked = json.load(f)
        acts_file = np.load(activations_path)
        pos_mask = acts_file["pos_mask"]

        concept_localized_count = 0
        anti_localized_count = 0
        for feat_info in ranked[:top_k]:
            key = f"L{feat_info['layer']}_F{feat_info['feat_id']}"
            if key not in acts_file:
                continue
            acts = acts_file[key]
            acts_pos = acts[pos_mask]
            acts_neg = acts[~pos_mask]
            threshold = (acts_pos.mean() + acts_neg.mean()) / 2.0

            pct_pos_high = 100.0 * (acts_pos > threshold).mean()
            pct_neg_low = 100.0 * (acts_neg <= threshold).mean()
            if pct_pos_high >= consistency_threshold and pct_neg_low >= consistency_threshold:
                concept_localized_count += 1

            pct_neg_high = 100.0 * (acts_neg > threshold).mean()
            pct_pos_low = 100.0 * (acts_pos <= threshold).mean()
            if pct_neg_high >= consistency_threshold and pct_pos_low >= consistency_threshold:
                anti_localized_count += 1

        n_features = min(top_k, len(ranked))
        return {
            "concept_localized": 100.0 * concept_localized_count / max(n_features, 1),
            "anti_localized": 100.0 * anti_localized_count / max(n_features, 1),
            "n_features": n_features,
        }
    except Exception:
        return None


def compute_concept_localization(
    concept: str, top_k: int = 200, consistency_threshold: float = 50.0
) -> dict:
    """Compute localization metrics for a concept, averaged across all anchor sweeps.

    Returns:
        {
            "concept": str,
            "n_features": int,
            "concept_localized": float (% of top-k that are well-localized),
            "anti_localized": float (% of top-k that are anti-localized),
            "error": str or None
        }
    """
    concept_dir = _BASE / concept
    sweep_dirs = sorted(concept_dir.glob("anchor_rank*/sweep"))

    if not sweep_dirs:
        return {
            "concept": concept,
            "n_features": 0,
            "concept_localized": None,
            "anti_localized": None,
            "error": f"Sweep results not found in {concept_dir}",
        }

    anchor_results = [
        r for sd in sweep_dirs if (r := _localization_for_sweep(sd, top_k, consistency_threshold)) is not None
    ]

    if not anchor_results:
        return {
            "concept": concept,
            "n_features": 0,
            "concept_localized": None,
            "anti_localized": None,
            "error": f"No valid sweep data found in {concept_dir}",
        }

    return {
        "concept": concept,
        "n_features": int(np.mean([r["n_features"] for r in anchor_results])),
        "concept_localized": float(np.mean([r["concept_localized"] for r in anchor_results])),
        "anti_localized": float(np.mean([r["anti_localized"] for r in anchor_results])),
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument(
        "--consistency_threshold",
        type=float,
        default=50.0,
        help="% of examples that must show consistent behavior",
    )
    parser.add_argument(
        "--output_path", default="runs/concept_localization/localization_metrics.json"
    )
    args = parser.parse_args()

    print(f"Computing localization metrics for {len(CONCEPTS)} concepts...")
    print(f"  Top-k: {args.top_k}")
    print(f"  Consistency threshold: {args.consistency_threshold}%\n")

    results = []
    for concept in sorted(CONCEPTS):
        result = compute_concept_localization(concept, args.top_k, args.consistency_threshold)
        results.append(result)

        status = "✓" if result["error"] is None else "⊘"
        if result["concept_localized"] is not None:
            print(
                f"{status} {concept:35s}  concept: {result['concept_localized']:5.1f}%  "
                f"anti: {result['anti_localized']:5.1f}%"
            )
        else:
            # Skip concepts without sweep results
            if "not found" in result["error"].lower():
                print(f"{status} {concept:35s}  (skipped — no sweep results)")
            else:
                print(f"{status} {concept:35s}  ERROR: {result['error']}")

    # Save results
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics → {args.output_path}")

    # Plot (only concepts with valid results)
    valid_results = [r for r in results if r["error"] is None]
    concepts = [r["concept"] for r in valid_results]
    concept_scores = [r["concept_localized"] for r in valid_results]
    anti_scores = [r["anti_localized"] for r in valid_results]

    if not concepts:
        print("\nNo valid results to plot (all concepts missing sweep results)")
        return

    print(f"\nPlotting {len(concepts)}/{len(CONCEPTS)} concepts with sweep results...")

    ps.apply()
    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(concepts))
    width = 0.35

    ax.bar(x - width / 2, concept_scores, width, label="Concept-localized", color="#2196F3")
    ax.bar(x + width / 2, anti_scores, width, label="Anti-localized", color="#E53935")

    ax.set_xlabel("Concept")
    ax.set_ylabel("% of top-k features well-localized")
    ax.set_title(
        f"Concept Localization Metrics (top-{args.top_k} features, ≥{args.consistency_threshold}% consistency)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha="right", fontsize=8)
    ax.set_ylim([0, 105])
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()

    fig.tight_layout()
    plot_path = Path(args.output_path).parent / "localization_metrics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
