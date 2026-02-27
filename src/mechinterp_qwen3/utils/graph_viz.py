import json
import os
import time
import urllib.parse
from collections import namedtuple

import torch
from transformers import AutoTokenizer

from ..graph import Metadata, Model, Node, QParams


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


Feature = namedtuple("Feature", ["layer", "pos", "feature_idx"])


def decode_url_features(url: str) -> tuple[dict[str, list[Feature]], list[Feature]]:
    """
    Extract both supernode features and individual singleton features from URL.

    Returns:
        Tuple of (supernode_features, singleton_features)
        - supernode_features: Dict mapping supernode names to lists of Features
        - singleton_features: List of individual Feature objects
    """
    decoded = urllib.parse.unquote(url)

    parsed_url = urllib.parse.urlparse(decoded)
    query_params = urllib.parse.parse_qs(parsed_url.query)

    # Extract supernodes
    supernodes_json = query_params.get("supernodes", ["[]"])[0]
    supernodes_data = json.loads(supernodes_json)

    supernode_features = {}
    name_counts = {}

    for supernode in supernodes_data:
        name = supernode[0]
        node_ids = supernode[1:]

        # Handle duplicate names by adding counter
        if name in name_counts:
            name_counts[name] += 1
            unique_name = f"{name} ({name_counts[name]})"
        else:
            name_counts[name] = 1
            unique_name = name

        nodes = []
        for node_id in node_ids:
            layer, feature_idx, pos = map(int, node_id.split("_"))
            nodes.append(Feature(layer, pos, feature_idx))

        supernode_features[unique_name] = nodes

    # Extract individual/singleton features from pinnedIds
    pinned_ids_str = query_params.get("pinnedIds", [""])[0]
    singleton_features = []

    if pinned_ids_str:
        pinned_ids = pinned_ids_str.split(",")
        for pinned_id in pinned_ids:
            # Handle both regular format (layer_feature_pos) and E_ format
            if pinned_id.startswith("E_"):
                # E_26865_9 format - embedding layer
                parts = pinned_id[2:].split("_")  # Remove 'E_' prefix
                if len(parts) == 2:
                    feature_idx, pos = map(int, parts)
                    # Use -1 to indicate embedding layer
                    singleton_features.append(Feature(-1, pos, feature_idx))
            else:
                # Regular layer_feature_pos format
                parts = pinned_id.split("_")
                if len(parts) == 3:
                    layer, feature_idx, pos = map(int, parts)
                    singleton_features.append(Feature(layer, pos, feature_idx))

    return supernode_features, singleton_features


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
    - Number of layers
    - Number of input tokens
    - Number of output (logit) nodes
    - Number of feature nodes
    - Total number of nodes
    - Total number of edges (non-zero entries in adjacency matrix)
    """
    n_layers = graph.cfg.n_layers
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)

    total_nodes = graph.adjacency_matrix.shape[0]
    # Count non-zero edges. Adjacency matrix is [target, source]
    n_edges = (graph.adjacency_matrix != 0).sum().item()

    stats = {
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "n_logits": n_logits,
        "n_features": n_features,
        "total_nodes": total_nodes,
        "n_edges": n_edges,
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
    else:
        with open(path, "w") as f:
            f.write("Graph Statistics:\n")
            f.write("-----------------\n")
            f.write(f"Layers:        {n_layers}\n")
            f.write(f"Input Tokens:  {n_tokens}\n")
            f.write(f"Output Nodes:  {n_logits}\n")
            f.write(f"Feature Nodes: {n_features}\n")
            f.write(f"Total Nodes:   {total_nodes}\n")
            f.write(f"Total Edges:   {n_edges}\n")

    print(f"INFO: Graph statistics saved to {path}")
