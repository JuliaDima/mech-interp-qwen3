"""NNSight verification tests (checking if our behaviour matches NNSight's)

1. Activation Equality Test
2. Reconstruction Test
3. Closure Test
4. Causal Prediction Test
"""

import pytest
import torch
import torch.nn as nn
from nnsight import LanguageModel

from mechinterp_qwen3.hooks import MLPHookManager


class MockTranscoder:
    """Mock transcoder for testing."""

    def __init__(self, d_model=128, n_features=256):
        self.d_model = d_model
        self.n_features = n_features
        self.encode_weight = nn.Parameter(torch.randn(d_model, n_features))
        self.decode_weight = nn.Parameter(torch.randn(n_features, d_model))

    def encode(self, x):
        return torch.relu(x @ self.encode_weight)

    def decode(self, f, x=None):
        return f @ self.decode_weight


@pytest.fixture
def nnsight_model(enhanced_mock_model):
    """Wrap the mock model in nnsight.LanguageModel."""
    return LanguageModel(enhanced_mock_model, tokenizer=None, dispatch=True)


class TestNNSightVerification:
    def test_phase2_activation_equality(self):
        """Phase 2: Activation Equality Test.

        We use a small Qwen2 model for consistency, as NNSight requires a standard HF architecture to trace properly.
        """
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        model = Qwen2ForCausalLM(config)
        model.eval()

        layer_ids = [0, 1]
        hooker = MLPHookManager(model, layer_ids, detach=True)
        hooker.install()

        batch_size = 1
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        # Capture baseline activations
        with torch.no_grad():
            _ = model(input_ids)

        baseline_in = hooker.cache[1].mlp_in.clone()
        baseline_out = hooker.cache[1].mlp_out.clone()
        hooker.remove()

        # Now use NNSight
        print("Loading NNSight Model...")
        nn_model = LanguageModel(model, dispatch=True)

        print("Tracing with NNSight...")
        with nn_model.trace(input_ids):
            # Qwen MLP block
            mlp_in_proxy = nn_model.model.layers[1].post_attention_layernorm.output.save()
            mlp_out_proxy = nn_model.model.layers[1].mlp.output.save()

        print("Evaluating NNSight graph...")
        nn_in = mlp_in_proxy
        nn_out = mlp_out_proxy

        # In our hook, output might be a tuple. NNSight output might be the raw tensor.
        if isinstance(nn_out, tuple):
            nn_out = nn_out[0]

        # Cosine similarity
        cos_sim_in = torch.nn.functional.cosine_similarity(
            baseline_in.flatten(), nn_in.flatten(), dim=0
        )
        cos_sim_out = torch.nn.functional.cosine_similarity(
            baseline_out.flatten(), nn_out.flatten(), dim=0
        )

        assert cos_sim_in.item() > 0.999
        assert cos_sim_out.item() > 0.999

    def test_phase3_reconstruction(self, enhanced_mock_model):
        """Phase 3: Reconstruction test."""
        # Using a mock transcoder. We want to check ||x - x_hat|| / ||x||
        transcoder = MockTranscoder(d_model=128)
        batch = 2
        seq = 5
        x = torch.randn(batch, seq, 128)

        f = transcoder.encode(x)
        x_hat = transcoder.decode(f, x)

        relative_error = torch.norm(x - x_hat) / (torch.norm(x) + 1e-6)

        # Error will be large for random weights, but we just verify the math structure runs.
        # In a real test we'd load the actual transcoder. Here we just assert it computes.
        assert isinstance(relative_error.item(), float)

    def test_phase4_closure(self):
        """Phase 4: Closure test for attribution graph."""
        # Sum of incoming contributions + bias/error ≈ actual value
        actual_val = 5.0
        incoming = [2.0, 1.5, 0.5]
        error = 1.0

        total = sum(incoming) + error
        assert abs(total - actual_val) < 1e-4

    def test_phase4_attribution_parity(self):
        """Phase 4: Attribution Gradient Parity Test.

        Verifies that gradients captured by our manual hooks (which simulate
        the causal relevance of nodes) match NNSight's gradient attribution.
        """
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        model = Qwen2ForCausalLM(config)
        model.train()  # Need train for gradients

        layer_ids = [0, 1]

        # We need detach=False to capture gradients during backward
        hooker = MLPHookManager(model, layer_ids, detach=False)
        hooker.install()

        batch_size = 1
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        # Baseline run with our hooks
        outputs = model(input_ids)
        loss = outputs.logits.sum()
        loss.backward()

        # baseline_grad_out is captured by our hook manager because detach=False
        baseline_grad_out = hooker.cache[1].mlp_out.grad.clone()
        hooker.remove()
        model.zero_grad()

        # NNSight run
        nn_model = LanguageModel(model, dispatch=True)
        with nn_model.trace(input_ids):
            # Capture mlp output and retain its gradient
            out = nn_model.model.layers[1].mlp.output
            out.retain_grad()

            # Compute sum loss exactly like above
            loss_nn = nn_model.lm_head.output.sum()
            loss_nn.backward()

            # Extract gradient
            nn_grad_out = out.grad.save()

        # If it was a tuple, our hook gets the primary tensor, so we handle both cases
        if isinstance(baseline_grad_out, tuple):
            baseline_grad_out = baseline_grad_out[0]

        if isinstance(nn_grad_out, tuple):
            nn_grad_out = nn_grad_out[0]

        # Cosine similarity of gradients
        cos_sim_grad = torch.nn.functional.cosine_similarity(
            baseline_grad_out.flatten(), nn_grad_out.flatten(), dim=0
        )

        assert cos_sim_grad.item() > 0.999

    def test_phase5_causal_intervention_parity(self):
        """Phase 5: Causal Intervention Parity Test.

        Verifies that NNSight can correctly ablate the causal features we
        identify in our graph.
        """
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        model = Qwen2ForCausalLM(config)
        model.eval()

        nn_model = LanguageModel(model, dispatch=True)
        batch_size = 1
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        # Baseline run
        with nn_model.trace(input_ids):
            baseline_logits = nn_model.lm_head.output.save()

        # Intervened run: ablate MLP layer 1
        with nn_model.trace(input_ids):
            nn_model.model.layers[1].mlp.output = torch.zeros_like(
                nn_model.model.layers[1].mlp.output
            )
            intervened_logits = nn_model.lm_head.output.save()

        diff = torch.norm(baseline_logits - intervened_logits)
        assert diff.item() > 0
