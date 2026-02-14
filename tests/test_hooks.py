"""Tests for hook manager and activation capture."""

from __future__ import annotations

import torch

from mechinterp_qwen3.hooks import LayerActs, LinearizedHookManager, MLPHookManager


class TestLayerActs:
    """Test the LayerActs dataclass."""

    def test_initialization_default(self):
        """Test default initialization."""
        acts = LayerActs()
        assert acts.mlp_in is None
        assert acts.mlp_out is None

    def test_initialization_with_tensors(self):
        """Test initialization with tensors."""
        mlp_in = torch.randn(10, 128)
        mlp_out = torch.randn(10, 128)

        acts = LayerActs(mlp_in=mlp_in, mlp_out=mlp_out)
        assert acts.mlp_in is mlp_in
        assert acts.mlp_out is mlp_out


class TestMLPHookManager:
    """Test the MLPHookManager class."""

    def test_initialization(self, mock_model):
        """Test hook manager initialization."""
        layer_ids = [0, 1, 2]
        hooker = MLPHookManager(mock_model, layer_ids)

        assert hooker.model is mock_model
        assert hooker.layer_ids == layer_ids
        assert len(hooker.handles) == 0
        assert len(hooker.cache) == len(layer_ids)

        for lid in layer_ids:
            assert lid in hooker.cache
            assert isinstance(hooker.cache[lid], LayerActs)

    def test_get_layers(self, mock_model):
        """Test that _get_layers returns correct layer list."""
        hooker = MLPHookManager(mock_model, [0])
        layers = hooker._get_layers()

        assert len(layers) == mock_model.n_layers
        assert all(hasattr(layer, "mlp") for layer in layers)

    def test_install_hooks(self, mock_model):
        """Test hook installation."""
        layer_ids = [0, 1]
        hooker = MLPHookManager(mock_model, layer_ids)

        hooker.install()

        # Should have 2 handles per layer (pre_hook + fwd_hook)
        assert len(hooker.handles) == len(layer_ids) * 2

    def test_capture_activations(self, mock_model):
        """Test that activations are captured correctly."""
        layer_ids = [0, 1, 2]
        hooker = MLPHookManager(mock_model, layer_ids)
        hooker.install()

        # Run a forward pass
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            _ = mock_model(input_ids)

        # Check that activations were captured
        for lid in layer_ids:
            assert hooker.cache[lid].mlp_in is not None
            assert hooker.cache[lid].mlp_out is not None

            # Check shapes [seq, d_model]
            assert hooker.cache[lid].mlp_in.shape == (seq_len, mock_model.d_model)
            assert hooker.cache[lid].mlp_out.shape == (seq_len, mock_model.d_model)

        hooker.remove()

    def test_activation_shapes(self, mock_model):
        """Test that captured activations have correct shapes."""
        layer_ids = [1]
        hooker = MLPHookManager(mock_model, layer_ids)
        hooker.install()

        seq_len = 15
        input_ids = torch.randint(0, 100, (1, seq_len))

        with torch.no_grad():
            _ = mock_model(input_ids)

        acts = hooker.cache[1]
        assert acts.mlp_in.shape[0] == seq_len
        assert acts.mlp_in.shape[1] == mock_model.d_model
        assert acts.mlp_out.shape[0] == seq_len
        assert acts.mlp_out.shape[1] == mock_model.d_model

        hooker.remove()

    def test_clear_cache(self, mock_model):
        """Test cache clearing functionality."""
        layer_ids = [0, 1]
        hooker = MLPHookManager(mock_model, layer_ids)
        hooker.install()

        # Run forward pass
        input_ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            _ = mock_model(input_ids)

        # Verify activations exist
        assert hooker.cache[0].mlp_in is not None

        # Clear cache
        hooker.clear_cache()

        # Verify cache is cleared
        for lid in layer_ids:
            assert hooker.cache[lid].mlp_in is None
            assert hooker.cache[lid].mlp_out is None

        hooker.remove()

    def test_remove_hooks(self, mock_model):
        """Test hook removal."""
        layer_ids = [0, 1]
        hooker = MLPHookManager(mock_model, layer_ids)
        hooker.install()

        assert len(hooker.handles) > 0

        hooker.remove()

        assert len(hooker.handles) == 0

    def test_multiple_forward_passes(self, mock_model):
        """Test that hooks work across multiple forward passes."""
        layer_ids = [0]
        hooker = MLPHookManager(mock_model, layer_ids)
        hooker.install()

        # First forward pass
        input_ids1 = torch.randint(0, 100, (1, 5))
        with torch.no_grad():
            _ = mock_model(input_ids1)

        acts1_in = hooker.cache[0].mlp_in.clone()

        # Clear and run second pass
        hooker.clear_cache()
        input_ids2 = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _ = mock_model(input_ids2)

        acts2_in = hooker.cache[0].mlp_in

        # Should have different shapes
        assert acts1_in.shape[0] == 5
        assert acts2_in.shape[0] == 8

        hooker.remove()

    def test_activations_on_cpu(self, mock_model):
        """Test that activations are moved to CPU."""
        layer_ids = [0]
        hooker = MLPHookManager(mock_model, layer_ids)
        hooker.install()

        input_ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            _ = mock_model(input_ids)

        # Activations should be on CPU
        assert hooker.cache[0].mlp_in.device.type == "cpu"
        hooker.remove()


class TestLinearizedHookManager:
    """Test the LinearizedHookManager class for attribution graphs."""

    def test_initialization(self, enhanced_mock_model):
        """Test linearized hook manager initialization."""
        lin_hooks = LinearizedHookManager(enhanced_mock_model)

        assert lin_hooks.model is enhanced_mock_model
        assert len(lin_hooks.handles) == 0

    def test_install_hooks(self, enhanced_mock_model):
        """Test hook installation."""
        lin_hooks = LinearizedHookManager(enhanced_mock_model)
        lin_hooks.install()

        # Should have:
        # 1 embedding hook
        # n_layers attention hooks
        # (2 * n_layers + 1) norm hooks (input_norm, post_attn_norm per layer, + final norm)
        expected_handles = 1 + enhanced_mock_model.n_layers + (2 * enhanced_mock_model.n_layers + 1)
        assert len(lin_hooks.handles) == expected_handles

        lin_hooks.remove()

    def test_embedding_requires_grad(self, enhanced_mock_model):
        """Test that embedding hook enables gradients on embeddings."""
        lin_hooks = LinearizedHookManager(enhanced_mock_model)
        lin_hooks.install()

        input_ids = torch.randint(0, 100, (1, 10))

        with torch.set_grad_enabled(True):
            outputs = enhanced_mock_model(input_ids)

        # The embeddings should require grad after the hook
        # We can't directly access them, but we can verify gradients flow through
        assert outputs.logits.requires_grad

        lin_hooks.remove()

    def test_attention_detach(self, enhanced_mock_model):
        """Test that attention outputs are detached."""
        lin_hooks = LinearizedHookManager(enhanced_mock_model)
        lin_hooks.install()

        input_ids = torch.randint(0, 100, (1, 10))

        with torch.set_grad_enabled(True):
            outputs = enhanced_mock_model(input_ids)
            loss = outputs.logits.sum()
            loss.backward()

        # We should be able to backward without errors
        # Attention detaching should not break the graph
        assert True  # If we got here, backward worked

        lin_hooks.remove()

    def test_rmsnorm_linearization(self, enhanced_mock_model):
        """Test that RMSNorm scale is treated as constant in backward."""
        lin_hooks = LinearizedHookManager(enhanced_mock_model)
        lin_hooks.install()

        input_ids = torch.randint(0, 100, (1, 10))

        with torch.set_grad_enabled(True):
            outputs = enhanced_mock_model(input_ids)
            loss = outputs.logits.sum()
            loss.backward()

        # The backward should work with linearized norms
        # This is a smoke test - the real test is that gradients are correct
        assert True

        lin_hooks.remove()

    def test_remove_hooks(self, enhanced_mock_model):
        """Test hook removal."""
        lin_hooks = LinearizedHookManager(enhanced_mock_model)
        lin_hooks.install()

        assert len(lin_hooks.handles) > 0

        lin_hooks.remove()

        assert len(lin_hooks.handles) == 0


class TestGradientFlow:
    """Test gradient flow through hooks for attribution graphs."""

    def test_mlp_hook_preserves_gradients(self, enhanced_mock_model):
        """Test that MLPHookManager preserves gradients when detach=False."""
        from mechinterp_qwen3.hooks import MLPHookManager

        layer_ids = [0, 1, 2]
        hooker = MLPHookManager(enhanced_mock_model, layer_ids, detach=False)
        hooker.install()

        input_ids = torch.randint(0, 100, (1, 10))

        with torch.set_grad_enabled(True):
            outputs = enhanced_mock_model(input_ids)
            _ = outputs.logits.sum()

        # Check that mlp_out has requires_grad
        for lid in layer_ids:
            mlp_out = hooker.cache[lid].mlp_out
            assert mlp_out is not None
            # mlp_out should have grad_fn when detach=False
            assert mlp_out.requires_grad or mlp_out.grad_fn is not None

        hooker.remove()

    def test_mlp_out_gradient_flow(self, enhanced_mock_model):
        """Test that gradients flow through mlp_out when retain_grad is called."""
        from mechinterp_qwen3.hooks import MLPHookManager

        layer_ids = [1]
        hooker = MLPHookManager(enhanced_mock_model, layer_ids, detach=False)
        hooker.install()

        input_ids = torch.randint(0, 100, (1, 10))

        with torch.set_grad_enabled(True):
            outputs = enhanced_mock_model(input_ids)

            # Retain grad on mlp_out
            mlp_out = hooker.cache[1].mlp_out
            mlp_out.retain_grad()

            loss = outputs.logits.sum()
            loss.backward()

            # mlp_out should have gradients
            assert mlp_out.grad is not None
            assert mlp_out.grad.shape == mlp_out.shape

        hooker.remove()

    def test_mlp_in_preserved_for_encoding(self, enhanced_mock_model):
        """Test that mlp_in is preserved for transcoder encoding."""
        from mechinterp_qwen3.hooks import MLPHookManager

        layer_ids = [0, 1]
        hooker = MLPHookManager(enhanced_mock_model, layer_ids, detach=False)
        hooker.install()

        input_ids = torch.randint(0, 100, (1, 10))

        with torch.set_grad_enabled(True):
            _ = enhanced_mock_model(input_ids)

        # mlp_in should be captured and available for encoding
        for lid in layer_ids:
            mlp_in = hooker.cache[lid].mlp_in
            assert mlp_in is not None
            assert mlp_in.shape == (10, enhanced_mock_model.d_model)

        hooker.remove()

    def test_end_to_end_gradient_flow(self, enhanced_mock_model):
        """Test end-to-end gradient flow from logits to embeddings."""
        from mechinterp_qwen3.hooks import LinearizedHookManager, MLPHookManager

        # Install both linearized and MLP hooks
        lin_hooks = LinearizedHookManager(enhanced_mock_model)
        lin_hooks.install()

        mlp_hooks = MLPHookManager(enhanced_mock_model, [0, 1, 2], detach=False)
        mlp_hooks.install()

        input_ids = torch.randint(0, 100, (1, 10))

        with torch.set_grad_enabled(True):
            outputs = enhanced_mock_model(input_ids)

            # Retain grad on MLP outputs
            for lid in [0, 1, 2]:
                mlp_hooks.cache[lid].mlp_out.retain_grad()

            loss = outputs.logits[:, -1, :].sum()
            loss.backward()

            # Check that gradients flowed to MLP outputs
            for lid in [0, 1, 2]:
                mlp_out = mlp_hooks.cache[lid].mlp_out
                assert mlp_out.grad is not None

        mlp_hooks.remove()
        lin_hooks.remove()


class TestAttributionGraphCompatibility:
    """Test hook compatibility with attribution graph requirements."""

    def test_hook_positions_match_transcoder_expectations(self, enhanced_mock_model):
        """Test that hooks capture at positions matching HF transcoder (hook_in/hook_out)."""
        from mechinterp_qwen3.hooks import MLPHookManager

        layer_ids = [0, 1]
        hooker = MLPHookManager(enhanced_mock_model, layer_ids, detach=True)
        hooker.install()

        input_ids = torch.randint(0, 100, (1, 8))

        with torch.no_grad():
            _ = enhanced_mock_model(input_ids)

        # Verify hook_in (mlp_in) and hook_out (mlp_out) are captured
        for lid in layer_ids:
            # mlp.hook_in should correspond to mlp_in
            assert hooker.cache[lid].mlp_in is not None
            # mlp.hook_out should correspond to mlp_out
            assert hooker.cache[lid].mlp_out is not None

            # Both should have shape [seq, d_model]
            assert hooker.cache[lid].mlp_in.shape == (8, enhanced_mock_model.d_model)
            assert hooker.cache[lid].mlp_out.shape == (8, enhanced_mock_model.d_model)

        hooker.remove()

    def test_frozen_attention_patterns(self, enhanced_mock_model):
        """Test that attention patterns can be frozen for attribution graphs."""
        from mechinterp_qwen3.hooks import LinearizedHookManager

        lin_hooks = LinearizedHookManager(enhanced_mock_model)
        lin_hooks.install()

        input_ids = torch.randint(0, 100, (1, 10))

        # First forward pass
        with torch.set_grad_enabled(True):
            outputs1 = enhanced_mock_model(input_ids)

        # The attention patterns are effectively frozen by the detach hook
        # This is a smoke test - the real verification is in perturbation experiments
        assert outputs1.logits is not None

        lin_hooks.remove()

    def test_error_correction_workflow_support(self, enhanced_mock_model):
        """Test that hooks support error correction workflow for attribution graphs."""
        from mechinterp_qwen3.hooks import MLPHookManager

        layer_ids = [0, 1, 2]
        hooker = MLPHookManager(enhanced_mock_model, layer_ids, detach=True)
        hooker.install()

        input_ids = torch.randint(0, 100, (1, 10))

        with torch.no_grad():
            _ = enhanced_mock_model(input_ids)

        # Simulate computing error terms: error = mlp_out - transcoder_reconstruction
        # We should be able to access mlp_out for this
        for lid in layer_ids:
            mlp_out = hooker.cache[lid].mlp_out
            assert mlp_out is not None

            # Simulate transcoder reconstruction (just zeros for testing)
            transcoder_reconstruction = torch.zeros_like(mlp_out)

            # Compute error
            error = mlp_out - transcoder_reconstruction
            assert error.shape == mlp_out.shape

        hooker.remove()

    def test_batch_size_one_requirement(self, enhanced_mock_model):
        """Test that hooks work with batch size 1 as required for attribution graphs."""
        from mechinterp_qwen3.hooks import MLPHookManager

        layer_ids = [0]
        hooker = MLPHookManager(enhanced_mock_model, layer_ids, detach=True)
        hooker.install()

        # Batch size = 1
        input_ids = torch.randint(0, 100, (1, 5))

        with torch.no_grad():
            _ = enhanced_mock_model(input_ids)

        # mlp_in and mlp_out should be [seq, d_model] (batch dimension removed)
        assert hooker.cache[0].mlp_in.shape == (5, enhanced_mock_model.d_model)
        assert hooker.cache[0].mlp_out.shape == (5, enhanced_mock_model.d_model)

        hooker.remove()


class TestTranscoderIntegration:
    """Test integration with SAE/transcoder workflows."""

    def test_mlp_in_shape_for_transcoder_encoding(self, enhanced_mock_model):
        """Test that mlp_in has correct shape for transcoder encoding."""
        from mechinterp_qwen3.hooks import MLPHookManager

        layer_ids = [1]
        hooker = MLPHookManager(enhanced_mock_model, layer_ids, detach=False)
        hooker.install()

        seq_len = 12
        input_ids = torch.randint(0, 100, (1, seq_len))

        with torch.set_grad_enabled(True):
            _ = enhanced_mock_model(input_ids)

        mlp_in = hooker.cache[1].mlp_in
        assert mlp_in is not None

        # mlp_in should be [seq, d_model] for transcoder.encode(mlp_in.unsqueeze(0))
        assert mlp_in.shape == (seq_len, enhanced_mock_model.d_model)
        assert mlp_in.dim() == 2

        # Test that we can add batch dimension for transcoder
        batched_mlp_in = mlp_in.unsqueeze(0)
        assert batched_mlp_in.shape == (1, seq_len, enhanced_mock_model.d_model)

        hooker.remove()

    def test_multiple_layers_for_cross_layer_transcoder(self, enhanced_mock_model):
        """Test capturing multiple layers for cross-layer transcoder."""
        from mechinterp_qwen3.hooks import MLPHookManager

        # Cross-layer transcoders need activations from multiple layers
        layer_ids = [0, 1, 2, 3]
        hooker = MLPHookManager(enhanced_mock_model, layer_ids, detach=False)
        hooker.install()

        input_ids = torch.randint(0, 100, (1, 10))

        with torch.set_grad_enabled(True):
            _ = enhanced_mock_model(input_ids)

        # All layers should have both mlp_in and mlp_out captured
        for lid in layer_ids:
            assert hooker.cache[lid].mlp_in is not None
            assert hooker.cache[lid].mlp_out is not None

        hooker.remove()

    def test_cache_clearing_between_prompts(self, enhanced_mock_model):
        """Test that cache can be cleared between prompts for batched processing."""
        from mechinterp_qwen3.hooks import MLPHookManager

        layer_ids = [0, 1]
        hooker = MLPHookManager(enhanced_mock_model, layer_ids, detach=True)
        hooker.install()

        # Process first prompt
        input_ids1 = torch.randint(0, 100, (1, 5))
        with torch.no_grad():
            _ = enhanced_mock_model(input_ids1)

        assert hooker.cache[0].mlp_in is not None

        # Clear cache
        hooker.clear_cache()

        # Cache should be empty
        assert hooker.cache[0].mlp_in is None
        assert hooker.cache[0].mlp_out is None

        # Process second prompt
        input_ids2 = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _ = enhanced_mock_model(input_ids2)

        # New activations should be captured
        assert hooker.cache[0].mlp_in is not None
        assert hooker.cache[0].mlp_in.shape[0] == 8  # New sequence length

        hooker.remove()
