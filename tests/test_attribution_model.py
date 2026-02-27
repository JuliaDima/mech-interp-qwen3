import pytest
import torch
import torch.nn as nn
from transformer_lens import HookedTransformerConfig

from mechinterp_qwen3.attribution_model import AttributionMLP, AttributionModel, AttributionUnembed
from mechinterp_qwen3.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet


@pytest.fixture
def tiny_cfg():
    # transformer-lens 2.17.0 signature: (n_layers, d_model, n_ctx, d_head, ...)
    # These 4 seem required by the signature effectively.
    return HookedTransformerConfig(
        n_layers=2,
        d_model=16,
        n_ctx=10,
        d_head=4,
        n_heads=4,
        d_mlp=32,
        d_vocab=100,
        act_fn="relu",
        tokenizer_name="gpt2",
    )


@pytest.fixture
def tiny_transcoder_set(tiny_cfg):
    transcoders = {}
    for i in range(tiny_cfg.n_layers):
        transcoders[i] = SingleLayerTranscoder(
            d_model=tiny_cfg.d_model,
            d_transcoder=64,
            activation_function=torch.nn.functional.relu,
            layer_idx=i,
            dtype=torch.float32,
        )
    return TranscoderSet(
        transcoders=transcoders,
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
        scan="test_scan",
    )


def test_attribution_mlp():
    old_mlp = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
    rmlp = AttributionMLP(old_mlp)

    x = torch.randn(2, 5, 16)
    output = rmlp(x)
    assert output.shape == (2, 5, 16)
    assert hasattr(rmlp, "hook_in")
    assert hasattr(rmlp, "hook_out")


def test_attribution_unembed():
    old_unembed = nn.Linear(16, 100)
    runembed = AttributionUnembed(old_unembed)

    x = torch.randn(2, 5, 16)
    output = runembed(x)
    assert output.shape == (2, 5, 100)
    assert hasattr(runembed, "hook_pre")
    assert hasattr(runembed, "hook_post")


def test_attribution_model_creation(tiny_cfg, tiny_transcoder_set):
    model = AttributionModel.from_config(tiny_cfg, tiny_transcoder_set)

    assert len(model.blocks) == tiny_cfg.n_layers
    assert isinstance(model.blocks[0].mlp, AttributionMLP)
    assert isinstance(model.unembed, AttributionUnembed)

    # Test forward
    input_tokens = torch.randint(0, 100, (1, 10))
    logits = model(input_tokens)
    assert logits.shape == (1, 10, 100)


def test_attribution_model_gradient_flow(tiny_cfg, tiny_transcoder_set):
    model = AttributionModel.from_config(tiny_cfg, tiny_transcoder_set)

    # Check that parameters are frozen
    for param in model.parameters():
        assert not param.requires_grad

    input_tokens = torch.randint(0, 100, (1, 5))
    logits = model(input_tokens)
    assert logits.shape == (1, 5, 100)
