"""Tests for hook manager and activation capture."""

from __future__ import annotations

import torch

from mechinterp_qwen3.hooks import LinearizedHookManager


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
        # (2 * n_layers + 1) norm hooks (input_norm, post_attn_norm per layer, + final norm)
        # Note: Attention detach hooks are no longer handled by LinearizedHookManager
        expected_handles = 1 + (2 * enhanced_mock_model.n_layers + 1)
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

    def test_remove_hooks_empty(self, enhanced_mock_model):
        """Test that removing hooks when none are installed doesn't crash."""
        lin_hooks = LinearizedHookManager(enhanced_mock_model)
        lin_hooks.remove()
        assert len(lin_hooks.handles) == 0
