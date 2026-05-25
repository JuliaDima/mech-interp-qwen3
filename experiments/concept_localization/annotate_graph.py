"""Annotate an attribution graph JSON with concept-aligned features.

Reads one or more concept localization results.json files and injects a
``concept_features`` key into the graph JSON.  The viz can then highlight
nodes whose layer_featId appears in that set.

Schema added to the graph file
--------------------------------
"concept_features": {
    "<concept>": [
        {"layer_feat": "14_73141", "layer": 14, "feature_id": 73141,
         "projection": 0.45, "rank": 0},
        ...
    ]
}

A node with node_id "14_73141_9" matches entry with layer_feat "14_73141".

Usage
-----
    python -m experiments.concept_localization.annotate_graph \\
        --graph  graphs/addition_36_59.json \\
        --concepts carry residue_class \\
        --top_k  10 \\
        --out    graphs/addition_36_59_annotated.json

    # annotate in-place
    python -m experiments.concept_localization.annotate_graph \\
        --graph graphs/addition_36_59.json --inplace

Correlation check
-----------------
After annotation, run with --check to print, for each concept feature,
whether a matching node appears in the graph and what its influence is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_RUNS = _REPO_ROOT / "runs" / "concept_localization"


def load_concept_features(
    concept: str,
    runs_dir: Path,
    top_k: int = 10,
) -> list[dict]:
    """Return top-k features for a concept, sorted by |projection|."""
    results_path = runs_dir / concept / "results.json"
    if not results_path.exists():
        print(f"  [warn] no results.json for concept '{concept}', skipping")
        return []

    results = json.loads(results_path.read_text())
    top_by_layer = results.get("top_features_by_layer", {})

    entries = []
    for layer_str, feats in top_by_layer.items():
        layer = int(layer_str)
        for rank, feat in enumerate(feats[:top_k]):
            fid = feat["feature_id"]
            entries.append({
                "layer_feat": f"{layer}_{fid}",
                "layer": layer,
                "feature_id": fid,
                "projection": feat.get("projection", 0.0),
                "rank": rank,
                "input_tokens": feat.get("input_tokens", []),
                "output_tokens": feat.get("output_tokens", []),
            })

    # Sort globally by |projection| so the viz can highlight the most salient first
    entries.sort(key=lambda e: abs(e["projection"]), reverse=True)
    return entries


def correlation_check(entries: list[dict], node_id_set: set[str]) -> None:
    """Print how many concept features appear as nodes in the graph."""
    present = [e for e in entries if e["layer_feat"] in node_id_set]
    print(f"  {len(present)}/{len(entries)} concept features present in graph")
    for e in present[:5]:
        print(f"    layer={e['layer']} feat={e['feature_id']} "
              f"proj={e['projection']:.4f} "
              f"out={e['output_tokens'][:2]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, type=Path,
                    help="Path to the attribution graph JSON")
    ap.add_argument("--concepts", nargs="+", default=["carry"],
                    help="Concept names (must have results.json in runs_dir/<concept>/)")
    ap.add_argument("--runs_dir", type=Path, default=_DEFAULT_RUNS)
    ap.add_argument("--top_k", type=int, default=10,
                    help="Top-k features per layer to include per concept")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: <graph>_annotated.json)")
    ap.add_argument("--inplace", action="store_true",
                    help="Overwrite the input graph file")
    ap.add_argument("--check", action="store_true",
                    help="Print correlation statistics after annotating")
    args = ap.parse_args()

    graph = json.loads(args.graph.read_text())

    # Build set of layer_feat strings present in the graph for correlation check
    node_lf_set: set[str] = set()
    for node in graph.get("nodes", []):
        nid = node.get("node_id", "")
        parts = nid.split("_")
        if len(parts) >= 2 and node.get("feature_type") == "CLT":
            node_lf_set.add(f"{parts[0]}_{parts[1]}")

    concept_features: dict[str, list[dict]] = {}
    for concept in args.concepts:
        print(f"Loading concept '{concept}'...")
        entries = load_concept_features(concept, args.runs_dir, top_k=args.top_k)
        concept_features[concept] = entries
        if args.check:
            correlation_check(entries, node_lf_set)

    graph["concept_features"] = concept_features

    if args.inplace:
        out = args.graph
    elif args.out:
        out = args.out
    else:
        out = args.graph.with_stem(args.graph.stem + "_annotated")

    out.write_text(json.dumps(graph, indent=2))
    total = sum(len(v) for v in concept_features.values())
    print(f"Annotated {total} feature entries across {len(concept_features)} concepts → {out}")


if __name__ == "__main__":
    main()
