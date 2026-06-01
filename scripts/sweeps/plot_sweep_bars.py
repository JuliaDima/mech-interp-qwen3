"""Plot sweep results from saved activations.

Loads sweep_ranked.json and sweep_activations.npz and generates bar plot.
No model loading required.

Usage:
    python scripts/sweeps/plot_sweep_bars.py --concept carry
    python scripts/sweeps/plot_sweep_bars.py --concept gcd --sweep_dir runs/concept_localization/gcd
    python scripts/sweeps/plot_sweep_bars.py --concept decimal_termination --top_per_layer 10
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

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import experiments.plot_style as ps


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", required=True, help="Concept name")
    parser.add_argument(
        "--sweep_dir",
        default=None,
        help="Sweep results directory (default: runs/concept_localization/<concept>)",
    )
    parser.add_argument(
        "--top_per_layer", type=int, default=5, help="Top features to display per layer"
    )
    parser.add_argument(
        "--output_path", default=None, help="Output path (default: <sweep_dir>/sweep_bars.png)"
    )
    args = parser.parse_args()

    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else Path(f"runs/concept_localization/{args.concept}")
    )
    ranked_path = sweep_dir / "sweep_ranked.json"
    activations_path = sweep_dir / "sweep_activations.npz"

    if not ranked_path.exists():
        print(f"ERROR: {ranked_path} not found")
        sys.exit(1)
    if not activations_path.exists():
        print(f"ERROR: {activations_path} not found")
        sys.exit(1)

    # Load ranked features and activations
    with open(ranked_path) as f:
        ranked = json.load(f)

    acts_file = np.load(activations_path)
    pos_mask = acts_file["pos_mask"]

    # Determine display items: top-per-layer per layer
    display_items = []
    layer_order = list(dict.fromkeys(r["layer"] for r in ranked))

    for layer in layer_order:
        layer_feats = [r for r in ranked if r["layer"] == layer][: args.top_per_layer]
        display_items.extend(layer_feats)

    if not display_items:
        print("No features to plot")
        sys.exit(1)

    layer_order = list(dict.fromkeys(r["layer"] for r in display_items))
    n_layers = len(layer_order)
    ncols = args.top_per_layer
    nrows = n_layers

    print(
        f"Plotting {len(display_items)} features ({n_layers} layers, {args.top_per_layer} per layer)..."
    )

    ps.apply()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.0 + 0.5, nrows * 2.5 + 1.2), squeeze=False
    )

    x = np.arange(len(pos_mask))
    bar_colors = ["#2196F3" if m else "#E53935" for m in pos_mask]

    col_idx = {l: 0 for l in layer_order}
    for feat_info in display_items:
        layer = feat_info["layer"]
        feat_id = feat_info["feat_id"]
        score = feat_info["score"]
        jaccard = feat_info["jaccard"]

        row = layer_order.index(layer)
        col = col_idx[layer]
        col_idx[layer] += 1
        if col >= ncols:
            continue

        ax = axes[row][col]
        key = f"L{layer}_F{feat_id}"
        if key not in acts_file:
            ax.set_visible(False)
            continue

        acts = acts_file[key]
        ax.bar(x, acts, color=bar_colors, width=0.8)
        ax.tick_params(labelsize=4)
        ax.set_xticks([])
        sign = "+" if score >= 0 else ""
        ax.set_title(
            f"L{layer:02d} F{feat_id:06d}\n{sign}{score:.2f}  J={jaccard:.2f}", fontsize=6, pad=2
        )

    for row in range(nrows):
        for col in range(col_idx.get(layer_order[row], 0), ncols):
            axes[row][col].set_visible(False)

    fig.legend(
        handles=[Patch(facecolor="#2196F3", label="pos"), Patch(facecolor="#E53935", label="neg")],
        loc="upper right",
        fontsize=7,
        framealpha=0.8,
    )
    fig.suptitle(
        f"{args.concept} — feature sweep ranked by Jaccard × |score|\n"
        "rows = layers · columns = top features per layer",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()

    output_path = Path(args.output_path) if args.output_path else sweep_dir / "sweep_bars.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
