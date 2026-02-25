"""Regression tests for attribution gradient flow."""

from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from mechinterp_qwen3.forward_with_sae import forward_linearized_with_sae_features


class MockTranscoder:
    """Simple mock transcoder for testing."""

    def __init__(self, d_model):
        self.d_model = d_model
        # W_dec needed for some logic? No, just encode/decode usually
        # But let's give it a dummy parameter just in case
        self.W_dec = torch.nn.Parameter(torch.randn(d_model, d_model))

    def encode(self, x):
        # x is [1, seq, d_model]
        # Identity-ish projection that preserves gradients
        return x  # Just pass through for simplicity, maybe scale

    def decode(self, f, x=None):
        return f


@pytest.fixture
def real_structure_model():
    """Create a real Qwen2 model structure with random weights (no download)."""
    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,  # Small for speed
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        _attn_implementation="eager",  # Default to eager for our logic
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


@pytest.fixture
def mock_tokenizer_qwen():
    """Mock tokenizer matching the vocab size."""
    tokenizer = MagicMock()
    tokenizer.vocab_size = 1000
    tokenizer.bos_token_id = 998
    tokenizer.eos_token_id = 999
    tokenizer.pad_token_id = 999
    tokenizer.all_special_ids = [998, 999]

    class BatchEncodingMock(dict):
        def to(self, device):
            # Move tensors in dict to device
            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    self[k] = v.to(device)
            return self

    def tokenize(text, return_tensors=None, **kwargs):
        # Return random tokens
        ids = [10, 11, 12, 13, 14, 15]
        if return_tensors == "pt":
            data = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, 6)}
            return BatchEncodingMock(data)
        return ids

    def decode(ids, **kwargs):
        return "token_" + str(ids)

    tokenizer.side_effect = tokenize
    tokenizer.__call__ = tokenize
    tokenizer.decode = decode
    return tokenizer


def test_gradient_flow_to_previous_tokens(real_structure_model, mock_tokenizer_qwen):
    """
    Verify that `forward_linearized_with_sae_features` allows gradients
    to flow to previous tokens (via Value path) despite Attention Pattern freezing.
    """
    model = real_structure_model
    tokenizer = mock_tokenizer_qwen

    # Setup transcoders for both layers
    transcoders = {
        0: MockTranscoder(model.config.hidden_size),
        1: MockTranscoder(model.config.hidden_size),
    }

    prompt = "Test prompt"
    layers = [0, 1]

    # Run the function
    try:
        results = forward_linearized_with_sae_features(
            model=model,
            tokenizer=tokenizer,
            transcoders=transcoders,
            prompt=prompt,
            layers_to_analyze=layers,
        )
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")

    pre_logit = results["pre_logit_hidden"]  # [1, seq, d_model]
    embeddings = results["embedding_activations"]  # [1, seq, d_model]

    assert (
        embeddings.grad_fn is not None or embeddings.requires_grad
    ), "Embeddings should track gradients"

    print(
        f"pre_logit shape: {pre_logit.shape}, requires_grad: {pre_logit.requires_grad}, grad_fn: {pre_logit.grad_fn}"
    )

    # Target: Sum of last token pre-logit hidden state
    target = pre_logit[0, -1, :].sum()

    print(
        f"target shape: {target.shape}, requires_grad: {target.requires_grad}, grad_fn: {target.grad_fn}"
    )

    # Backward
    target.backward()

    assert embeddings.grad is not None, "No gradients captured on embeddings"

    # Check gradient norm per token
    grad = embeddings.grad[0]  # [seq, d_model]
    token_norms = grad.norm(dim=-1)

    # We expect gradients on previous tokens!
    # With strict freezing (detached QK, but flowing V), earlier tokens contribute via V.
    # So their gradient should be > 0.

    # Note: Token 0 might be a sink or special, check if > 0.
    # At least some previous tokens must be > 0.

    non_zero_grads = (token_norms > 1e-9).sum().item()
    total_tokens = token_norms.shape[0]

    # We expect flow to more than just the last token
    assert (
        non_zero_grads > 1
    ), f"Gradients only flowed to {non_zero_grads}/{total_tokens} tokens! Expected flow to previous tokens."

    print(f"Gradient flow confirmed to {non_zero_grads}/{total_tokens} tokens.")
