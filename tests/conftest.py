"""Pytest configuration and shared fixtures for mechinterp-qwen3 tests."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test file I/O."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_prompt_data():
    """Sample prompt data for testing."""
    return [
        {
            "prompt_id": "gt_0000",
            "behaviour": "greater_than",
            "a": 42,
            "b": 17,
            "prompt": "You are solving a simple comparison task.\nTwo numbers are given: A and B.\nAnswer with a single character: 'A' if A is larger, otherwise 'B'.\n\nA = 42\nB = 17\nAnswer: ",
            "expected": "A",
        },
        {
            "prompt_id": "gt_0001",
            "behaviour": "greater_than",
            "a": 10,
            "b": 99,
            "prompt": "You are solving a simple comparison task.\nTwo numbers are given: A and B.\nAnswer with a single character: 'A' if A is larger, otherwise 'B'.\n\nA = 10\nB = 99\nAnswer: ",
            "expected": "B",
        },
    ]


class MockMLP(nn.Module):
    """Mock MLP module for testing hooks."""
    
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass."""
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


class MockTransformerBlock(nn.Module):
    """Mock transformer block with MLP."""
    
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.mlp = MockMLP(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass."""
        # Simplified: just MLP + residual
        return x + self.mlp(self.layer_norm(x))


class MockModel(nn.Module):
    """Mock model with proper layer structure for testing."""
    
    def __init__(self, n_layers: int = 4, d_model: int = 128):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockTransformerBlock(d_model) for _ in range(n_layers)
        ])
        self.d_model = d_model
        self.n_layers = n_layers
        self.device = torch.device("cpu")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Simple forward pass through all layers."""
        # input_ids: [batch, seq]
        # Create simple embeddings
        batch, seq = input_ids.shape
        x = torch.randn(batch, seq, self.d_model, device=self.device)
        
        for layer in self.model.layers:
            x = layer(x)
        
        return x


@pytest.fixture
def mock_model():
    """Provide a mock model with proper layer structure."""
    model = MockModel(n_layers=4, d_model=128)
    model.eval()
    return model


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer."""
    tokenizer = MagicMock()
    
    def tokenize_fn(text: str, return_tensors: str | None = None):
        # Simple mock: return token IDs based on text length
        token_ids = list(range(len(text) // 10 + 5))  # Arbitrary length
        result = {"input_ids": token_ids}
        
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor([token_ids])
            result["attention_mask"] = torch.ones_like(result["input_ids"])
        
        return result
    
    tokenizer.side_effect = tokenize_fn
    tokenizer.__call__ = tokenize_fn
    tokenizer.eos_token_id = 0
    
    def decode_fn(token_ids, skip_special_tokens=False):
        # Simple mock decode
        return "A"
    
    tokenizer.decode = decode_fn
    
    return tokenizer
