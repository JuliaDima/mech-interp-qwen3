import pytest
import torch
from transformer_lens import HookedTransformerConfig

from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.graph import Graph
from mechinterp_qwen3.run_attribution import attribute
from mechinterp_qwen3.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet


@pytest.fixture
def tiny_cfg():
    return HookedTransformerConfig(
        n_layers=1,
        d_model=8,
        n_ctx=10,
        d_head=4,
        n_heads=2,
        d_mlp=16,
        d_vocab=50257,
        act_fn="relu",
        tokenizer_name="gpt2",
    )


@pytest.fixture
def tiny_transcoder_set(tiny_cfg):
    transcoders = {
        0: SingleLayerTranscoder(8, 32, torch.nn.functional.relu, 0, dtype=torch.float32)
    }
    return TranscoderSet(
        transcoders=transcoders,
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
    )


def test_attribute_basic(tiny_cfg, tiny_transcoder_set):
    model = AttributionModel.from_config(tiny_cfg, tiny_transcoder_set)

    # Pass tokens directly
    input_tokens = torch.randint(0, 50, (1, 5))

    graph = attribute(input_tokens, model, max_n_logits=2, desired_logit_prob=0.9, verbose=True)

    assert isinstance(graph, Graph)
    # 5 random tokens + 1 prepended BOS = 6 tokens
    assert graph.input_tokens.shape == (6,)
    assert graph.logit_tokens.shape[0] <= 2
    assert graph.adjacency_matrix is not None


def test_attribute_with_empty_activations(tiny_cfg, tiny_transcoder_set):
    model = AttributionModel.from_config(tiny_cfg, tiny_transcoder_set)
    for t in tiny_transcoder_set:
        t.W_enc.data.zero_()
        t.b_enc.data.fill_(-100.0)

    input_tokens = torch.randint(0, 50, (1, 3))
    graph = attribute(input_tokens, model, max_n_logits=1)

    assert isinstance(graph, Graph)
    assert graph.active_features is not None
