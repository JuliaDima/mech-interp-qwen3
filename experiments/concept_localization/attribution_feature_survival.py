"""Compute feature survival statistics across attribution graphs.

For each CLT feature (layer, feat_idx) that appears in any attribution graph,
computes:
  - n_graphs:          total graphs it appears in
  - n_carry:           appearances in carry (pos) graphs
  - n_nocarry:         appearances in no-carry (neg) graphs
  - mean_influence:    average influence score when present
  - mean_activation:   average activation when present
  - carry_enrichment:  (n_carry/n_carry_total) / (n_nocarry/n_nocarry_total)
                       > 1 means more common in carry graphs

Outputs:
  - survival_stats.json:     full table sorted by n_graphs descending
  - top_features.json:       top-k features by survival (for use in modulation)
  - survival_by_layer.png:   bar chart of surviving features per layer

Usage:
    python -m experiments.concept_localization.attribution_feature_survival \\
        --graphs_dir graphs --min_survival 0.1 --topk 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_node_id(node_id: str) -> tuple[int, int, int] | None:
    """Parse CLT node_id '{layer}_{feat_idx}_{ctx_idx}' → (layer, feat_idx, ctx_idx)."""
    parts = node_id.split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def load_graph(path: Path) -> dict:
    for f in path.iterdir():
        if f.suffix == ".json" and f.name != "graph-metadata.json":
            return json.loads(f.read_text())
    raise FileNotFoundError(f"No graph JSON in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", default="carry",
                        help="Concept name — used to set default pattern, out_dir, and pos/neg labels")
    parser.add_argument("--graphs_dir", default="graphs",
                        help="Directory containing per-prompt graph subdirs")
    parser.add_argument("--pattern", default=None,
                        help="Glob pattern to filter graph directories (default: {concept}_T0)")
    parser.add_argument("--neg_tag", default="nocarry",
                        help="String in dir name that marks a negative/control graph "
                             "(default: 'nocarry'; for other concepts may be e.g. 'neg')")
    parser.add_argument("--min_survival", type=float, default=0.05,
                        help="Minimum fraction of graphs a feature must appear in to be kept")
    parser.add_argument("--topk", type=int, default=100,
                        help="Number of top features to save in top_features.json")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    pattern = args.pattern or f"{args.concept}_T0"
    graphs_dir = _REPO_ROOT / args.graphs_dir
    graph_dirs = sorted(d for d in graphs_dir.iterdir()
                        if d.is_dir() and pattern in d.name)

    pos_dirs = [d for d in graph_dirs if args.neg_tag not in d.name]
    neg_dirs = [d for d in graph_dirs if args.neg_tag in d.name]
    n_pos, n_neg = len(pos_dirs), len(neg_dirs)
    n_total = len(graph_dirs)

    print(f"Found {n_total} graphs: {n_pos} pos, {n_neg} neg  (pattern={pattern!r}, neg_tag={args.neg_tag!r})")

    # Accumulate per-feature stats
    # key: (layer, feat_idx)
    # value: {n_total, n_pos, n_neg, influences, activations}
    stats: dict[tuple[int,int], dict] = defaultdict(lambda: {
        "n_total": 0, "n_pos": 0, "n_neg": 0,
        "influences": [], "activations": [],
    })

    for i, gdir in enumerate(graph_dirs):
        is_pos = args.neg_tag not in gdir.name
        try:
            g = load_graph(gdir)
        except Exception as e:
            print(f"  skipping {gdir.name}: {e}")
            continue

        # Deduplicate per graph: aggregate over context positions first,
        # then record one entry per (layer, feat_idx) per graph.
        graph_feats: dict[tuple[int,int], dict] = {}
        for node in g.get("nodes", []):
            if node.get("feature_type") != "CLT":
                continue
            parsed = parse_node_id(node["node_id"])
            if parsed is None:
                continue
            layer, feat_idx, _ = parsed
            key = (layer, feat_idx)
            inf = float(node.get("influence", 0.0))
            act = float(node.get("activation", 0.0))
            if key not in graph_feats:
                graph_feats[key] = {"max_influence": inf, "max_activation": act}
            else:
                graph_feats[key]["max_influence"] = max(graph_feats[key]["max_influence"], inf)
                graph_feats[key]["max_activation"] = max(graph_feats[key]["max_activation"], act)

        for key, gf in graph_feats.items():
            s = stats[key]
            s["n_total"] += 1
            if is_pos:
                s["n_pos"] += 1
            else:
                s["n_neg"] += 1
            s["influences"].append(gf["max_influence"])
            s["activations"].append(gf["max_activation"])

        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{n_total} graphs, {len(stats)} unique features so far")

    print(f"\nTotal unique (layer, feat_idx) pairs: {len(stats)}")

    # Build rows
    rows = []
    for (layer, feat_idx), s in stats.items():
        n = s["n_total"]
        np_ = s["n_pos"]
        nn  = s["n_neg"]
        pos_rate = np_ / n_pos if n_pos > 0 else 0.0
        neg_rate = nn  / n_neg if n_neg  > 0 else 0.0
        enrichment = pos_rate / neg_rate if neg_rate > 1e-9 else float("inf")
        rows.append({
            "layer": layer,
            "feat_idx": feat_idx,
            "feature_key": f"L{layer}_F{feat_idx}",
            "n_graphs": n,
            "n_pos": np_,
            "n_neg": nn,
            "survival_rate": n / n_total,
            "pos_rate": pos_rate,
            "neg_rate": neg_rate,
            "pos_enrichment": enrichment,
            "mean_influence": float(np.mean(s["influences"])),
            "max_influence": float(np.max(s["influences"])),
            "mean_activation": float(np.mean(s["activations"])),
        })

    rows.sort(key=lambda r: r["n_graphs"], reverse=True)

    # Filter by min_survival
    min_n = int(args.min_survival * n_total)
    surviving = [r for r in rows if r["n_graphs"] >= min_n]
    print(f"Features surviving ≥{args.min_survival:.0%} of graphs ({min_n}/{n_total}): {len(surviving)}")

    out_dir = Path(args.out_dir or
                   _REPO_ROOT / "runs" / "concept_localization" / args.concept / "feature_survival")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "survival_stats.json").write_text(json.dumps({
        "config": {
            "concept": args.concept,
            "pattern": pattern,
            "neg_tag": args.neg_tag,
            "n_total_graphs": n_total,
            "n_pos_graphs": n_pos,
            "n_neg_graphs": n_neg,
            "min_survival": args.min_survival,
            "min_n_graphs": min_n,
        },
        "features": rows,
        "surviving": surviving,
    }, indent=2))

    top = surviving[:args.topk]
    (out_dir / "top_features.json").write_text(
        json.dumps([r["feature_key"] for r in top], indent=2)
    )
    print(f"Saved top {len(top)} features → {out_dir}/top_features.json")

    # Pos-discriminative: pos_enrichment > 1.3 and enough pos appearances
    min_pos = max(5, int(0.04 * n_pos))
    pos_disc = [
        r for r in rows
        if r["n_pos"] >= min_pos and r["pos_enrichment"] > 1.3
    ]
    pos_disc.sort(key=lambda r: (r["pos_enrichment"] * r["n_pos"]), reverse=True)
    (out_dir / "pos_discriminative.json").write_text(json.dumps({
        "config": {
            "concept": args.concept,
            "n_pos_graphs": n_pos,
            "n_neg_graphs": n_neg,
            "min_pos": min_pos,
            "min_enrichment": 1.3,
        },
        "features": pos_disc,
    }, indent=2))
    print(f"Saved {len(pos_disc)} pos-discriminative features → {out_dir}/pos_discriminative.json")

    # --- Layer distribution plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        sys.path.insert(0, str(_REPO_ROOT))
        import experiments.plot_style as ps
        ps.apply()
        navy, violet, teal, gray = ps.NAVY, ps.VIOLET, ps.TEAL, ps.GRAY
    except Exception:
        navy, violet, teal, gray = "#1f3a5f", "#7b3fa0", "#2a7f6f", "#888888"

    n_layers = 36
    layers = np.arange(n_layers)

    # Count surviving features per layer
    surv_per_layer = np.zeros(n_layers, dtype=int)
    pos_enr_layer  = np.zeros(n_layers, dtype=int)  # pos-enriched (enrichment > 1.5)
    for r in surviving:
        l = r["layer"]
        if l < n_layers:
            surv_per_layer[l] += 1
            if r["pos_enrichment"] > 1.5:
                pos_enr_layer[l] += 1

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1, 1]})

    # Top: surviving features per layer
    ax1.bar(layers, surv_per_layer, color=navy, alpha=0.75, label="all surviving")
    ax1.bar(layers, pos_enr_layer,  color=violet, alpha=0.85, label="pos-enriched (>1.5×)")
    ax1.set_ylabel(f"Surviving features\n(≥{args.min_survival:.0%} of graphs)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", color="#E0E0E0", lw=0.5)
    ax1.set_title(
        f"Attribution graph feature survival — {args.concept} {pattern}, "
        f"{n_total} graphs, min_survival={args.min_survival:.0%}",
        fontsize=11,
    )

    # Middle: max mean influence of surviving features per layer
    mean_inf_per_layer = np.zeros(n_layers)
    for r in surviving:
        l = r["layer"]
        if l < n_layers:
            mean_inf_per_layer[l] = max(mean_inf_per_layer[l], r["mean_influence"])

    ax2.bar(layers, mean_inf_per_layer, color=teal, alpha=0.75)
    ax2.set_ylabel("Max mean influence\n(surviving)", fontsize=10)
    ax2.grid(axis="y", color="#E0E0E0", lw=0.5)

    # Bottom: pos-discriminative feature count per layer
    cd_per_layer = np.zeros(n_layers, dtype=int)
    for r in pos_disc:
        l = r["layer"]
        if l < n_layers:
            cd_per_layer[l] += 1

    ax3.bar(layers, cd_per_layer, color=violet, alpha=0.80)
    ax3.set_ylabel("Pos-discriminative\nfeatures (>1.3×)", fontsize=10)
    ax3.set_xlabel("Layer", fontsize=10)
    ax3.grid(axis="y", color="#E0E0E0", lw=0.5)
    ax3.set_xticks(layers[::2])

    fig.tight_layout()
    plot_path = out_dir / "survival_by_layer.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {plot_path}")

    # Print summary table
    print(f"\nTop 30 features by survival:")
    print(f"{'Feature':<18} {'n_graphs':>9} {'survival':>9} {'pos_rate':>10} "
          f"{'neg_rate':>10} {'enrichment':>11} {'mean_inf':>10}")
    print("-" * 82)
    for r in rows[:30]:
        enr = f"{r['pos_enrichment']:.2f}" if r['pos_enrichment'] != float("inf") else "  inf"
        print(
            f"{r['feature_key']:<18} {r['n_graphs']:>9} {r['survival_rate']:>9.1%} "
            f"{r['pos_rate']:>10.1%} {r['neg_rate']:>10.1%} "
            f"{enr:>11} {r['mean_influence']:>10.4f}"
        )

    print(f"\nLayer summary (surviving features):")
    print(f"{'Layer':>6} {'n_surviving':>12} {'pos_enriched':>13}")
    print("-" * 34)
    for l in range(n_layers):
        if surv_per_layer[l] > 0:
            print(f"{l:>6} {surv_per_layer[l]:>12} {pos_enr_layer[l]:>13}")


if __name__ == "__main__":
    main()
