import random

import numpy as np
import pytest
import torch
from transformer_lens import HookedTransformerConfig

from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet
from mechinterp_qwen3.utils_seed import SeedConfig, set_all_seeds


def test_basic_determinism():
    cfg = SeedConfig(seed=42, deterministic=True)

    # Test random
    set_all_seeds(cfg)
    r1 = random.random()
    set_all_seeds(cfg)
    r2 = random.random()
    assert r1 == r2

    # Test numpy
    set_all_seeds(cfg)
    n1 = np.random.rand(5)
    set_all_seeds(cfg)
    n2 = np.random.rand(5)
    assert np.array_equal(n1, n2)

    # Test torch
    set_all_seeds(cfg)
    t1 = torch.rand(5)
    set_all_seeds(cfg)
    t2 = torch.rand(5)
    assert torch.equal(t1, t2)


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


def test_model_determinism(tiny_cfg, tiny_transcoder_set):
    cfg = SeedConfig(seed=42, deterministic=True)

    set_all_seeds(cfg)
    model1 = AttributionModel.from_config(tiny_cfg, tiny_transcoder_set)
    input_tokens = torch.randint(0, 50, (1, 5))
    logits1 = model1(input_tokens)

    set_all_seeds(cfg)
    model2 = AttributionModel.from_config(tiny_cfg, tiny_transcoder_set)
    # Re-using input_tokens might be slightly cheating if the generation of tokens itself is the source of randomness,
    # but here we want to ensure the model's forward pass/initialization is deterministic if it has any randomness.
    logits2 = model2(input_tokens)

    assert torch.allclose(logits1, logits2)


def test_attribution_determinism(tiny_cfg, tiny_transcoder_set):
    from mechinterp_qwen3.run_attribution import attribute

    cfg = SeedConfig(seed=42, deterministic=True)

    model = AttributionModel.from_config(tiny_cfg, tiny_transcoder_set)
    input_tokens = torch.randint(0, 50, (1, 5))

    set_all_seeds(cfg)
    graph1 = attribute(input_tokens, model, max_n_logits=2, desired_logit_prob=0.9)

    set_all_seeds(cfg)
    graph2 = attribute(input_tokens, model, max_n_logits=2, desired_logit_prob=0.9)

    assert torch.allclose(graph1.adjacency_matrix, graph2.adjacency_matrix)
    assert torch.equal(graph1.input_tokens, graph2.input_tokens)
    assert torch.equal(graph1.logit_tokens, graph2.logit_tokens)
