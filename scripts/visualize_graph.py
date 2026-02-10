#!/usr/bin/env python3
"""Visualize attribution graph using Graphviz."""

import argparse
import json
import sys
from pathlib import Path

try:
    import graphviz
except ImportError:
    print(
        "Error: graphviz python package not found. Please install it with `pip install graphviz`."
    )
    sys.exit(1)


def load_graph(json_path: str):
    with open(json_path) as f:
        return json.load(f)


def visualize(graph_data: dict, output_path: str, format: str = "pdf", limit_features: int = 0):
    """Render graph using graphviz."""
    dot = graphviz.Digraph(comment="Attribution Graph", format=format)
    dot.attr(rankdir="LR")  # Left to right layout
    dot.attr(newrank="true")  # Allow cross-subgraph ranking
    # dot.attr(ratio="compress") # Try to compress?
    dot.attr("node", shape="box", style="filled", fontname="Helvetica")

    # Filter features if limit is set
    features = [n for n in graph_data["nodes"] if n["node_type"] == "feature"]
    tokens = [n for n in graph_data["nodes"] if n["node_type"] == "token"]
    logits = [n for n in graph_data["nodes"] if n["node_type"] == "logit"]
    errors = [n for n in graph_data["nodes"] if n["node_type"] == "error"]

    if limit_features > 0 and len(features) > limit_features:
        print(f"  Limiting features to top {limit_features} (by attribution sum)...")
        # Calculate approximate attribution weight for each feature
        # This is strictly local connectivity here, might need 'total_attribution' if available
        # or sum of absolute edge weights connected to it.
        # Let's rely on 'total_attribution' if present in node, else edge sum.

        # Build score map
        node_scores = {n["node_id"]: n.get("total_attribution", 0.0) for n in features}

        # If total_attribution is missing/zero, sum edges
        if all(s == 0 for s in node_scores.values()):
            for edge in graph_data["edges"]:
                if edge["source"] in node_scores:
                    node_scores[edge["source"]] += abs(edge["attribution_score"])
                if edge["target"] in node_scores:
                    node_scores[edge["target"]] += abs(edge["attribution_score"])

        # Sort and pick top K
        sorted_feats = sorted(
            features, key=lambda n: node_scores.get(n["node_id"], 0), reverse=True
        )
        kept_features = set(n["node_id"] for n in sorted_feats[:limit_features])

        # Filter nodes list
        graph_data["nodes"] = (
            tokens + logits + errors + [n for n in features if n["node_id"] in kept_features]
        )

        # Filter edges
        valid_ids = set(n["node_id"] for n in graph_data["nodes"])
        graph_data["edges"] = [
            e for e in graph_data["edges"] if e["source"] in valid_ids and e["target"] in valid_ids
        ]

    # Create Subgraphs to enforce layout
    with dot.subgraph(name="cluster_0_input") as inputs:
        inputs.attr(label="Input Tokens")
        inputs.attr(rank="source")
        for node in tokens:
            _add_node(inputs, node)

    with dot.subgraph(name="cluster_1_features") as middle:
        middle.attr(label="Features / Error")
        # middle.attr(rank="same") # Too restrictive if we have errors?
        for node in graph_data["nodes"]:
            if node["node_type"] in ["feature", "error"]:
                _add_node(middle, node)

    with dot.subgraph(name="cluster_2_output") as outputs:
        outputs.attr(label="Output Logits")
        outputs.attr(rank="sink")
        for node in logits:
            _add_node(outputs, node)

    # Add edges
    # Normalize edge widths?
    max_attr = 0.0
    for edge in graph_data["edges"]:
        max_attr = max(max_attr, abs(edge["attribution_score"]))

    for edge in graph_data["edges"]:
        src = edge["source"]
        dst = edge["target"]
        score = edge["attribution_score"]

        # Color: Red for negative, Blue for positive
        color = "#d32f2f" if score < 0 else "#1976d2"

        # Width based on magnitude
        penwidth = 1.0 + (abs(score) / max_attr) * 4.0 if max_attr > 0 else 1.0

        dot.edge(src, dst, color=color, penwidth=str(penwidth), tooltip=f"{score:.4f}")

    # Render
    output_file = dot.render(output_path, view=False, cleanup=True)
    print(f"Graph rendered to: {output_file}")


def _add_node(graph, node):
    node_id = node["node_id"]
    node_type = node["node_type"]

    label = node_id
    fillcolor = "white"

    if node_type == "token":
        label = f"{node['token_str']}\n({node['token_pos']})"
        fillcolor = "#e1f5fe"  # Light blue
        # graph.attr("node", group=f"token_{node['token_pos']}")

    elif node_type == "logit":
        label = f"{node['logit_token_str']}\n{node['activation']:.2f}"
        fillcolor = "#fff3e0"  # Light orange

    elif node_type == "feature":
        # Feature node
        layer = node["layer"]
        feat_id = node["feature_id"]
        act = node["activation"]
        label = f"L{layer}.{feat_id}\nAct: {act:.2f}"
        fillcolor = "#f3e5f5"  # Light purple

    elif node_type == "error":
        # Error node
        layer = node["layer"]
        act = node["activation"]
        label = f"Error L{layer}\n|r|: {act:.2f}"
        fillcolor = "#ffcdd2"  # Light Red

    graph.node(node_id, label=label, fillcolor=fillcolor)


def main():
    parser = argparse.ArgumentParser(description="Visualize attribution graph JSON.")
    parser.add_argument(
        "json_file", type=str, help="Path to graph JSON file (e.g. pruned_graph.json)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output filename (without extension)"
    )
    parser.add_argument("--format", type=str, default="pdf", help="Output format (pdf, png, svg)")
    parser.add_argument(
        "--split", action="store_true", help="Split graph by layer into separate files"
    )
    parser.add_argument(
        "--limit_features",
        type=int,
        default=0,
        help="Limit number of features per graph (0 = no limit)",
    )
    args = parser.parse_args()

    input_path = Path(args.json_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    if args.output is None:
        args.output = input_path.with_suffix("").name + "_viz"
        # If input is in a subdir, output there too?
        # dot.render handles path.
        output_path = str(input_path.parent / args.output)
    else:
        output_path = args.output

    print(f"Loading graph from {input_path}...")
    graph_data = load_graph(input_path)

    print(f"Visualizing {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges...")

    if args.split:
        layers = set()
        for node in graph_data["nodes"]:
            if "layer" in node and node["layer"] is not None:
                layers.add(node["layer"])

        layers = sorted(list(layers))
        print(f"Found layers: {layers}. Splitting graph...")

        for layer in layers:
            viz_data = {"nodes": [], "edges": []}

            # Filter nodes: keep Tokens, Logits, and nodes for THIS layer
            node_ids = set()
            for node in graph_data["nodes"]:
                should_keep = False
                if node["node_type"] in ["token", "logit"] or node.get("layer") == layer:
                    should_keep = True

                if should_keep:
                    viz_data["nodes"].append(node)
                    node_ids.add(node["node_id"])

            # Filter edges: only if both source/target are kept
            for edge in graph_data["edges"]:
                if edge["source"] in node_ids and edge["target"] in node_ids:
                    viz_data["edges"].append(edge)

            layer_output = f"{output_path}_L{layer}"
            print(f"  Rendering Layer {layer} graph ({len(viz_data['nodes'])} nodes)...")
            visualize(viz_data, layer_output, args.format, limit_features=args.limit_features)

    else:
        visualize(graph_data, output_path, args.format, limit_features=args.limit_features)


if __name__ == "__main__":
    main()
