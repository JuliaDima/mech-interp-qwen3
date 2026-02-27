import pytest
import torch

from mechinterp_qwen3.graph import Graph, Node, UnifiedConfig, compute_graph_scores, prune_graph


@pytest.fixture
def sample_graph():
    cfg = UnifiedConfig(
        n_layers=1,
        d_model=4,
        d_head=2,
        n_heads=2,
        d_mlp=8,
        d_vocab=10,
        tokenizer_name="gpt2",
        model_name="test",
        original_architecture="test",
    )
    # Correct order: [features (2), errors (2), tokens (2), logits (1)]
    # Total = 7 nodes
    n_features = 2
    n_errors = 2
    n_tokens = 2
    n_logits = 1
    total_nodes = n_features + n_errors + n_tokens + n_logits

    adj = torch.zeros(total_nodes, total_nodes)
    # Logit 0 (index 6) from feature 0 (index 0)
    adj[6, 0] = 0.8
    # Feature 0 (index 0) from token 0 (index 4)
    adj[0, 4] = 0.5
    # Feature 1 (index 1) from token 1 (index 5)
    adj[1, 5] = 0.5

    return Graph(
        input_string="test",
        input_tokens=torch.tensor([1, 2]),
        active_features=torch.tensor([[0, 0, 0], [0, 0, 1]]),
        adjacency_matrix=adj,
        cfg=cfg,
        logit_tokens=torch.tensor([8]),
        logit_probabilities=torch.tensor([1.0]),
        selected_features=torch.tensor([0, 1]),
        activation_values=torch.tensor([1.0, 1.0]),
    )


def test_prune_graph_basic(sample_graph):
    node_mask, edge_mask, final_scores = prune_graph(
        sample_graph, node_threshold=0.9, edge_threshold=0.9
    )

    assert node_mask.any()
    assert edge_mask.any()
    # Logit node (index 6) should be kept
    assert node_mask[6]
    # Feature 0 (index 0) should be kept as it's the main influencer
    assert node_mask[0]


def test_node_factory():
    fn = Node.feature_node(layer=0, pos=1, feat_idx=42, influence=0.5, activation=1.2)
    assert fn.feature_type == "cross layer transcoder"
    assert fn.layer == "0"
    assert fn.ctx_idx == 1
    assert "0_42_1" in fn.node_id

    tn = Node.token_node(pos=0, vocab_idx=123, influence=0.1)
    assert tn.feature_type == "embedding"
    assert tn.ctx_idx == 0


def test_compute_graph_scores(sample_graph):
    replacement, completeness = compute_graph_scores(sample_graph)
    assert 0 <= replacement <= 1.0
    assert 0 <= completeness <= 1.0
    # In my graph, error nodes have 0 influence because I didn't add edges to them
    assert replacement == 1.0
