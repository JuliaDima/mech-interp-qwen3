import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from mechinterp_qwen3.graph import Graph, UnifiedConfig
from mechinterp_qwen3.utils.graph_viz import create_graph_files


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda x: f"token_{x}"
    return tokenizer


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
    n_features = 1
    n_errors = 1
    n_tokens = 1
    n_logits = 1
    total = n_features + n_errors + n_tokens + n_logits

    adj = torch.zeros(total, total)
    # Logit (3) from feat (0)
    adj[3, 0] = 1.0
    # Feat (0) from token (2)
    adj[0, 2] = 1.0

    return Graph(
        input_string="test",
        input_tokens=torch.tensor([1]),
        active_features=torch.tensor([[0, 0, 0]]),
        adjacency_matrix=adj,
        cfg=cfg,
        logit_tokens=torch.tensor([8]),
        logit_probabilities=torch.tensor([1.0]),
        selected_features=torch.tensor([0]),
        activation_values=torch.tensor([1.0]),
        scan="test-scan",
    )


def test_create_graph_files_basic(sample_graph, mock_tokenizer):
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("mechinterp_qwen3.utils.graph_viz.AutoTokenizer") as mock_auto,
    ):
        mock_auto.from_pretrained.return_value = mock_tokenizer

        create_graph_files(
            sample_graph,
            slug="test-slug",
            output_path=tmpdir,
            node_threshold=0.0,
            edge_threshold=0.0,
        )

        output_file = os.path.join(tmpdir, "test-slug.json")
        assert os.path.exists(output_file)

        with open(output_file) as f:
            data = json.load(f)
            assert data["metadata"]["slug"] == "test-slug"
            assert len(data["nodes"]) > 0
            assert len(data["links"]) > 0

        # Check graph-metadata.json
        metadata_file = os.path.join(tmpdir, "graph-metadata.json")
        assert os.path.exists(metadata_file)
