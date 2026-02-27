import pytest
import torch

from mechinterp_qwen3.graph import Graph, UnifiedConfig, standardize_config


@pytest.fixture
def sample_config_dict():
    return {
        "n_layers": 2,
        "d_model": 16,
        "d_head": 4,
        "n_heads": 4,
        "d_mlp": 32,
        "d_vocab": 100,
        "tokenizer_name": "test_tok",
        "model_name": "test_model",
        "original_architecture": "TestArch",
    }


def test_unified_config(sample_config_dict):
    cfg = UnifiedConfig.from_dict(sample_config_dict)
    assert cfg.n_layers == 2
    assert cfg.d_model == 16
    assert cfg.to_dict()["n_layers"] == 2


def test_standardize_config(sample_config_dict):
    # From dict
    cfg = standardize_config(sample_config_dict)
    assert isinstance(cfg, UnifiedConfig)
    assert cfg.n_layers == 2

    # From UnifiedConfig
    cfg2 = standardize_config(cfg)
    assert cfg2 is cfg


def test_graph_initialization(sample_config_dict):
    cfg = UnifiedConfig.from_dict(sample_config_dict)
    input_string = "Hello"
    input_tokens = torch.tensor([1, 2, 3])
    active_features = torch.tensor([[0, 0, 10], [1, 2, 20]])
    adjacency_matrix = torch.zeros((10, 10))
    logit_tokens = torch.tensor([5, 6])
    logit_probabilities = torch.tensor([0.8, 0.1])
    selected_features = torch.tensor([10, 20])
    activation_values = torch.tensor([1.0, 0.5])

    graph = Graph(
        input_string=input_string,
        input_tokens=input_tokens,
        active_features=active_features,
        adjacency_matrix=adjacency_matrix,
        cfg=cfg,
        logit_tokens=logit_tokens,
        logit_probabilities=logit_probabilities,
        selected_features=selected_features,
        activation_values=activation_values,
        scan="test_scan",
    )

    assert graph.input_string == input_string
    assert torch.equal(graph.active_features, active_features)
    assert isinstance(graph.cfg, UnifiedConfig)
    assert graph.scan == "test_scan"


def test_graph_to_device(sample_config_dict):
    cfg = UnifiedConfig.from_dict(sample_config_dict)
    active_features = torch.tensor([[0, 0, 10]])
    adjacency_matrix = torch.zeros((1, 1))

    graph = Graph(
        "test",
        torch.tensor([1]),
        active_features,
        adjacency_matrix,
        cfg,
        torch.tensor([1]),
        torch.tensor([1.0]),
        torch.tensor([1]),
        torch.tensor([1.0]),
    )

    # Moving to CPU (should already be there, but let's test the method)
    graph.to("cpu")
    assert graph.active_features.device == torch.device("cpu")
