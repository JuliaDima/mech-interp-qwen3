"""Simple test script for attribution graph construction.

Tests the from-scratch implementation on a simple greater-than example.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mechinterp_qwen3.compute_attribution import compute_attribution_graph


def test_simple_example() -> None:
    """Test attribution graph on a simple greater-than example."""
    # Simple prompt: 864 > 394, answer should be "A"
    prompt = """You are solving a simple comparison task.
Two numbers are given: A and B.
Answer with a single character: 'A' if A is larger, otherwise 'B'.

A = 864
B = 394
Answer: """

    print("=" * 80)
    print("Testing Attribution Graph Construction")
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print("=" * 80)

    # Load model
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    # Compute attribution graph
    print("\nComputing attribution graph...")
    graph = compute_attribution_graph(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        layers_to_analyze=[4, 12, 20],
        transcoder_repo="mwhanna/qwen3-4b-transcoders",
        max_n_logits=10,
        desired_logit_prob=0.95,
        feature_threshold=0.01,
        min_attribution=1e-3,
    )

    print(f"\n{'=' * 80}")
    print(f"Graph constructed: {graph}")
    print(f"{'=' * 80}")

    # Prune graph
    print("\nPruning graph...")
    pruned_graph = graph.prune(node_threshold=0.8, edge_threshold=0.98)
    print(f"Pruned graph: {pruned_graph}")

    # Print statistics
    print(f"\n{'=' * 80}")
    print("Graph Statistics")
    print(f"{'=' * 80}")
    print(f"Raw graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"Pruned graph: {len(pruned_graph.nodes)} nodes, {len(pruned_graph.edges)} edges")

    # Diagnostics
    n_edges = len(graph.edges)
    if n_edges == 0:
        print("\nWARNING: No edges in graph! Check gradient flow.")
    else:
        attr_scores = [abs(e.attribution_score) for e in graph.edges]
        print(f"\nAttribution magnitude stats ({n_edges} edges):")
        print(f"  max:  {max(attr_scores):.6f}")
        print(f"  mean: {sum(attr_scores)/len(attr_scores):.6f}")
        print(f"  min:  {min(attr_scores):.6f}")

    # Show top features by attribution
    print("\nTop 10 features by total attribution:")
    feature_nodes = [n for n in pruned_graph.nodes.values() if n.node_type == "feature"]
    feature_nodes.sort(key=lambda n: n.total_attribution, reverse=True)

    for i, node in enumerate(feature_nodes[:10]):
        print(
            f"  {i+1}. Layer {node.layer}, Feature {node.feature_id} "
            f"(pos={node.token_pos}, attr={node.total_attribution:.4f}, "
            f"act={node.activation:.4f})"
        )

    # Show logit nodes
    print("\nLogit nodes:")
    logit_nodes = [n for n in graph.nodes.values() if n.node_type == "logit"]
    for node in logit_nodes:
        print(f"  '{node.logit_token_str}' (prob={node.activation:.4f})")

    print("\nTest complete!")


if __name__ == "__main__":
    test_simple_example()
