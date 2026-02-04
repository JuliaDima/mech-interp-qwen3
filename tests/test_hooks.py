"""Tests for hook manager and activation capture."""

from __future__ import annotations

import torch

from mechinterp_qwen3.hooks import LayerActs, MLPHookManager


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
        assert hooker.cache[0].mlp_out.device.type == "cpu"

        hooker.remove()
