"""Build attribution graphs for greater-than behavior (from scratch implementation).

This script constructs attribution graphs by:
1. Running forward pass with SAE feature extraction
2. Computing gradients from output logits to SAE features
3. Building graph structure with nodes and edges
4. Pruning low-attribution nodes and edges
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_attribution import compute_attribution_graph


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build attribution graph from scratch using gradient-based attribution"
    )
    ap.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to analyze (full prompt text)",
    )
    ap.add_argument(
        "--slug",
        type=str,
        required=True,
        help="Name/identifier for this analysis run",
    )
    ap.add_argument(
        "--graph_dir",
        type=str,
        default="graphs",
        help="Directory to save graph files",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model name (must match transcoder dimensions)",
    )
    ap.add_argument(
        "--transcoder_repo",
        type=str,
        default="mwhanna/qwen3-4b-transcoders",
        help="HuggingFace repo containing transcoders",
    )
    ap.add_argument(
        "--layers",
        type=str,
        default="4,12,20",
        help="Comma-separated layer IDs to analyze",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype (bfloat16, float16, float32)",
    )
    ap.add_argument(
        "--max_n_logits",
        type=int,
        default=10,
        help="Maximum number of top logits to attribute from",
    )
    ap.add_argument(
        "--desired_logit_prob",
        type=float,
        default=0.95,
        help="Cumulative probability threshold for salient logit selection",
    )
    ap.add_argument(
        "--feature_threshold",
        type=float,
        default=0.01,
        help="Minimum feature activation to include",
    )
    ap.add_argument(
        "--min_attribution",
        type=float,
        default=1e-3,
        help="Minimum |attribution score| to include an edge in the raw graph",
    )
    ap.add_argument(
        "--node_threshold",
        type=float,
        default=0.8,
        help="Pruning: keep fewest nodes covering this fraction of total attribution",
    )
    ap.add_argument(
        "--edge_threshold",
        type=float,
        default=0.98,
        help="Pruning: keep fewest edges covering this fraction of total attribution",
    )
    args = ap.parse_args()

    # Parse layers
    layers = [int(x.strip()) for x in args.layers.split(",")]

    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 80)
    print(f"Building Attribution Graph: {args.slug}")
    print("=" * 80)
    print(f"Prompt: {args.prompt[:100]}...")
    print(f"Model: {args.model}")
    print(f"Transcoders: {args.transcoder_repo}")
    print(f"Layers: {layers}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Create output directory
    graph_dir = Path(args.graph_dir) / args.slug
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Model loaded on {args.device}")

    # Compute attribution graph
    print("\n" + "=" * 80)
    print("Computing Attribution Graph")
    print("=" * 80)

    graph = compute_attribution_graph(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        layers_to_analyze=layers,
        transcoder_repo=args.transcoder_repo,
        max_n_logits=args.max_n_logits,
        desired_logit_prob=args.desired_logit_prob,
        feature_threshold=args.feature_threshold,
        min_attribution=args.min_attribution,
    )

    # Save raw graph
    raw_graph_path = graph_dir / "raw_graph.json"
    with open(raw_graph_path, "w") as f:
        json.dump(graph.to_dict(), f, indent=2)
    print(f"\n✓ Saved raw graph to: {raw_graph_path}")

    # Prune graph
    print("\n" + "=" * 80)
    print("Pruning Graph")
    print("=" * 80)
    print(f"Node threshold: {args.node_threshold}")
    print(f"Edge threshold: {args.edge_threshold}")

    pruned_graph = graph.prune(
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )

    print(f"\nPruned graph: {pruned_graph}")

    # Save pruned graph
    pruned_graph_path = graph_dir / "pruned_graph.json"
    with open(pruned_graph_path, "w") as f:
        json.dump(pruned_graph.to_dict(), f, indent=2)
    print(f"✓ Saved pruned graph to: {pruned_graph_path}")

    # Save metadata
    metadata = {
        "slug": args.slug,
        "prompt": args.prompt,
        "model": args.model,
        "transcoder_repo": args.transcoder_repo,
        "layers": layers,
        "max_n_logits": args.max_n_logits,
        "desired_logit_prob": args.desired_logit_prob,
        "feature_threshold": args.feature_threshold,
        "min_attribution": args.min_attribution,
        "node_threshold": args.node_threshold,
        "edge_threshold": args.edge_threshold,
        "raw_graph": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
        },
        "pruned_graph": {
            "nodes": len(pruned_graph.nodes),
            "edges": len(pruned_graph.edges),
        },
    }

    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata to: {metadata_path}")

    print("\n" + "=" * 80)
    print("✓ Attribution Graph Construction Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {graph_dir}")
    print("  - raw_graph.json: Full attribution graph")
    print(
        f"  - pruned_graph.json: Pruned graph (top {args.node_threshold * 100}% nodes, {args.edge_threshold * 100}% edges)"
    )
    print("  - metadata.json: Run configuration and statistics")


if __name__ == "__main__":
    main()
