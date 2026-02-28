import json
import os
import time

import torch
from transformers import AutoTokenizer


def add_graph_metadata(graph_metadata, path):
    assert os.path.exists(os.path.dirname(path)), f"Could not find {os.path.dirname(path)}"
    if os.path.isdir(path):
        path = os.path.join(path, "graph-metadata.json")

    if os.path.exists(path):
        with open(path) as f:
            metadata = json.load(f)
    else:
        metadata = {"graphs": []}

    metadata["graphs"] = [g for g in metadata["graphs"] if g["slug"] != graph_metadata["slug"]]
    metadata["graphs"].append(graph_metadata)

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def process_token(token: str) -> str:
    return token.replace("\n", "⏎").replace("\t", "→").replace("\r", "↵")


def load_graph_data(file_path):
    """Load graph data from a PyTorch file."""
    from ..graph import Graph

    start_time = time.time()
    graph = Graph.from_pt(file_path)
    time_ms = (time.time() - start_time) * 1000
    print(f"INFO: Loading graph data: {time_ms=:.2f} ms")
    return graph


def create_nodes(graph, node_mask, tokenizer, cumulative_scores):
    """Create all nodes for the graph."""
    from ..graph import Node

    start_time = time.time()

    nodes = {}

    n_features = len(graph.selected_features)
    layers = graph.cfg.n_layers
    error_end_idx = n_features + graph.n_pos * layers  # type: ignore
    token_end_idx = error_end_idx + len(graph.input_tokens)

    for node_idx in node_mask.nonzero().squeeze().tolist():
        if node_idx in range(n_features):
            layer, pos, feat_idx = graph.active_features[graph.selected_features[node_idx]].tolist()
            nodes[node_idx] = Node.feature_node(
                layer,
                pos,
                feat_idx,
                influence=cumulative_scores[node_idx],
                activation=graph.activation_values[graph.selected_features[node_idx]].item(),
            )
        elif node_idx in range(n_features, error_end_idx):
            layer, pos = divmod(node_idx - n_features, graph.n_pos)
            nodes[node_idx] = Node.error_node(layer, pos, influence=cumulative_scores[node_idx])
        elif node_idx in range(error_end_idx, token_end_idx):
            pos = node_idx - error_end_idx
            nodes[node_idx] = Node.token_node(
                pos, graph.input_tokens[pos], influence=cumulative_scores[node_idx]
            )
        elif node_idx in range(token_end_idx, len(cumulative_scores)):
            pos = node_idx - token_end_idx
            nodes[node_idx] = Node.logit_node(
                pos=graph.n_pos - 1,
                vocab_idx=graph.logit_tokens[pos],
                token=tokenizer.decode(graph.logit_tokens[pos]),
                target_logit=pos == 0,
                token_prob=graph.logit_probabilities[pos].item(),
                num_layers=layers,
            )

    total_time = (time.time() - start_time) * 1000
    print(f"INFO: Total node creation: {total_time=:.2f} ms")

    return nodes


def create_used_nodes_and_edges(graph, nodes, edge_mask):
    """Filter to only used nodes and create edges."""
    start_time = time.time()
    edges = edge_mask.numpy()
    dsts, srcs = edges.nonzero()
    weights = graph.adjacency_matrix.numpy()[dsts, srcs].tolist()

    used_edges = [
        {"source": nodes[src].node_id, "target": nodes[dst].node_id, "weight": weight}
        for src, dst, weight in zip(srcs, dsts, weights, strict=False)
        if src in nodes and dst in nodes
    ]

    connected_ids = set()
    for edge in used_edges:
        connected_ids.add(edge["source"])
        connected_ids.add(edge["target"])

    nodes_before = len(nodes)
    used_nodes = [
        node
        for node in nodes.values()
        if node.node_id in connected_ids or node.feature_type in ["embedding", "logit"]
    ]
    nodes_after = len(used_nodes)
    print(f"INFO: Filtered {nodes_before - nodes_after} nodes")

    time_ms = (time.time() - start_time) * 1000
    print(f"INFO: Creating used nodes and edges: {time_ms=:.2f} ms")
    print(f"INFO: Used nodes: {len(used_nodes)}, Used edges: {len(used_edges)}")

    return used_nodes, used_edges


def build_model(graph, used_nodes, used_edges, slug, scan, node_threshold, tokenizer):
    """Build the full model object."""
    from ..graph import Metadata, Model, QParams

    start_time = time.time()

    if isinstance(scan, list):
        transcoder_list = scan
        transcoder_list_str = "-".join(transcoder_list)
        transcoder_list_hash = hash(transcoder_list_str)
        scan = "custom-" + str(transcoder_list_hash)
    else:
        transcoder_list = []

    meta = Metadata(
        slug=slug,
        scan=scan,
        transcoder_list=transcoder_list,
        prompt_tokens=[tokenizer.decode(t) for t in graph.input_tokens],
        prompt=graph.input_string,
        node_threshold=node_threshold,
    )

    qparams = QParams(
        pinnedIds=[],
        supernodes=[],
        linkType="both",
        clickedId="",
        sg_pos="",
    )

    full_model = Model(
        metadata=meta,
        qParams=qparams,
        nodes=used_nodes,
        links=used_edges,
    )

    time_ms = (time.time() - start_time) * 1000
    print(f"INFO: Building model: {time_ms=:.2f} ms")

    return full_model


def create_graph_files(
    graph_or_path,
    slug: str,
    output_path,
    scan=None,
    node_threshold=0.8,
    edge_threshold=0.98,
):
    from ..graph import Graph, prune_graph

    total_start_time = time.time()

    graph = graph_or_path if isinstance(graph_or_path, Graph) else load_graph_data(graph_or_path)

    if os.path.exists(output_path):
        assert os.path.isdir(output_path)
    else:
        os.makedirs(output_path, exist_ok=True)

    if scan is None:
        if graph.scan is None:
            raise ValueError(
                "Neither scan nor graph.scan was set. One must be set to identify "
                "which transcoders were used when creating the graph."
            )
        scan = graph.scan

    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph.to(device)
    node_mask, edge_mask, cumulative_scores = (
        el.cpu() for el in prune_graph(graph, node_threshold, edge_threshold)
    )
    graph.to("cpu")

    tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)
    nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores)
    used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
    model = build_model(graph, used_nodes, used_edges, slug, scan, node_threshold, tokenizer)

    # Write the output locally
    with open(os.path.join(output_path, f"{slug}.json"), "w") as f:
        f.write(model.model_dump_json(indent=2))
    add_graph_metadata(model.metadata.model_dump(), output_path)
    print(f"INFO: Graph data written to {output_path}")

    total_time_ms = (time.time() - total_start_time) * 1000
    print(f"INFO: Total execution time: {total_time_ms=:.2f} ms")


def save_graph_stats(graph, path: str):
    """Save graph statistics to a file (JSON or text).

    Stats include:
    - Number of layers, tokens, logits, features
    - Total nodes and edges
    - Per-layer statistics for activations and edges (mean, median, min, max, sum)
    """
    import numpy as np

    n_layers = graph.cfg.n_layers
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)

    adj = graph.adjacency_matrix
    total_nodes = adj.shape[0]
    n_edges = (adj != 0).sum().item()

    # Feature node indices: 0 to n_features - 1
    # Layer for each feature node
    feat_layers = graph.active_features[graph.selected_features][:, 0].cpu().numpy()
    feat_activations = (
        graph.activation_values[graph.selected_features].to(torch.float32).cpu().numpy()
    )

    def get_summary_stats(data):
        if len(data) == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "sum": 0.0,
                "count": 0,
            }
        return {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "std": float(np.std(data)),
            "sum": float(np.sum(data)),
            "count": int(len(data)),
        }

    per_layer = []
    for layer_idx in range(n_layers):
        layer_mask = feat_layers == layer_idx
        layer_feat_indices = np.where(layer_mask)[0]
        layer_acts = feat_activations[layer_mask]

        # Edges involving these feature nodes
        # Outgoing edges: from these features to anything
        if n_edges > 0 and len(layer_feat_indices) > 0:
            out_adj = adj[:, layer_feat_indices]
            out_weights = out_adj[out_adj != 0].to(torch.float32).cpu().numpy()

            in_adj = adj[layer_feat_indices, :]
            in_weights = in_adj[in_adj != 0].to(torch.float32).cpu().numpy()
        else:
            out_weights = np.array([])
            in_weights = np.array([])

        per_layer.append(
            {
                "layer": int(layer_idx),
                "n_features": int(len(layer_feat_indices)),
                "activations": get_summary_stats(layer_acts),
                "edge_weights_out": get_summary_stats(out_weights),
                "edge_weights_in": get_summary_stats(in_weights),
            }
        )

    stats = {
        "summary": {
            "n_layers": n_layers,
            "n_tokens": n_tokens,
            "n_logits": n_logits,
            "n_features": n_features,
            "total_nodes": total_nodes,
            "n_edges": n_edges,
        },
        "per_layer": per_layer,
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
    else:
        with open(path, "w") as f:
            f.write("Graph Statistics Summary:\n")
            f.write("-------------------------\n")
            f.write(f"Layers:        {n_layers}\n")
            f.write(f"Input Tokens:  {n_tokens}\n")
            f.write(f"Output Nodes:  {n_logits}\n")
            f.write(f"Feature Nodes: {n_features}\n")
            f.write(f"Total Nodes:   {total_nodes}\n")
            f.write(f"Total Edges:   {n_edges}\n\n")

            f.write("Per-Layer Breakdown:\n")
            f.write("--------------------\n")
            for layer in per_layer:
                f.write(f"Layer {layer['layer']} ({layer['n_features']} features):\n")
                acts = layer["activations"]
                f.write(
                    f"  Activations: mean={acts['mean']:.4f}, med={acts['median']:.4f}, "
                    f"min={acts['min']:.4f}, max={acts['max']:.4f}\n"
                )
                e_out = layer["edge_weights_out"]
                f.write(
                    f"  Edges Out:   count={e_out['count']}, mean={e_out['mean']:.4f}, "
                    f"max={e_out['max']:.4f}\n"
                )
                e_in = layer["edge_weights_in"]
                f.write(
                    f"  Edges In:    count={e_in['count']}, mean={e_in['mean']:.4f}, "
                    f"max={e_in['max']:.4f}\n\n"
                )

    print(f"INFO: Graph statistics saved to {path}")
