import os
import tempfile

import torch

from mechinterp_qwen3.graph import Graph, UnifiedConfig
from mechinterp_qwen3.utils.graph_viz import save_graph_stats


def test_save_graph_stats_text():
    # Setup dummy graph
    cfg = UnifiedConfig(
        n_layers=2,
        d_model=8,
        d_head=4,
        n_heads=2,
        d_mlp=16,
        d_vocab=100,
        tokenizer_name="gpt2",
        model_name="test",
        original_architecture="test",
    )

    input_tokens = torch.tensor([1, 2, 3])
    active_features = torch.tensor([[0, 0, 1], [1, 1, 5]])
    adjacency_matrix = torch.zeros(10, 10)
    adjacency_matrix[0, 1] = 1.2
    adjacency_matrix[2, 3] = 0.5

    graph = Graph(
        input_string="test",
        input_tokens=input_tokens,
        active_features=active_features,
        adjacency_matrix=adjacency_matrix,
        cfg=cfg,
        logit_tokens=torch.tensor([4, 5]),
        logit_probabilities=torch.tensor([0.6, 0.4]),
        selected_features=torch.tensor([0, 1]),
        activation_values=torch.tensor([1.0, 2.0]),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = os.path.join(tmpdir, "stats.txt")
        save_graph_stats(graph, stats_path)

        assert os.path.exists(stats_path)
        with open(stats_path) as f:
            content = f.read()
            assert "Layers:        2" in content
            assert "Input Tokens:  3" in content
            assert "Output Nodes:  2" in content
            assert "Feature Nodes: 2" in content
            assert "Total Edges:   2" in content


def test_save_graph_stats_json():
    # Setup dummy graph (same as above)
    cfg = UnifiedConfig(
        n_layers=2,
        d_model=8,
        d_head=4,
        n_heads=2,
        d_mlp=16,
        d_vocab=100,
        tokenizer_name="gpt2",
        model_name="test",
        original_architecture="test",
    )
    graph = Graph(
        input_string="test",
        input_tokens=torch.tensor([1]),
        active_features=torch.tensor([[0, 0, 0]]),
        adjacency_matrix=torch.ones(5, 5),
        cfg=cfg,
        logit_tokens=torch.tensor([1]),
        logit_probabilities=torch.tensor([1.0]),
        selected_features=torch.tensor([0]),
        activation_values=torch.tensor([1.0]),
    )

    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = os.path.join(tmpdir, "stats.json")
        save_graph_stats(graph, stats_path)

        assert os.path.exists(stats_path)
        with open(stats_path) as f:
            data = json.load(f)
            assert data["summary"]["n_layers"] == 2
            assert data["summary"]["n_edges"] == 25
            assert "per_layer" in data
            assert len(data["per_layer"]) == 2
            assert data["per_layer"][0]["layer"] == 0
            assert data["per_layer"][0]["n_features"] == 1
            assert "activations" in data["per_layer"][0]
            assert "edge_weights_out" in data["per_layer"][0]
