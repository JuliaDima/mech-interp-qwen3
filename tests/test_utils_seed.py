"""Tests for seed utilities and determinism."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from mechinterp_qwen3.utils_seed import SeedConfig, set_all_seeds


class TestSeedConfig:
    """Test the SeedConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        cfg = SeedConfig(seed=42)
        assert cfg.seed == 42
        assert cfg.deterministic is True
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        cfg = SeedConfig(seed=123, deterministic=False)
        assert cfg.seed == 123
        assert cfg.deterministic is False


class TestSetAllSeeds:
    """Test the set_all_seeds function."""
    
    def test_sets_random_seed(self):
        """Test that Python random seed is set."""
        set_all_seeds(SeedConfig(seed=42))
        val1 = random.random()
        
        set_all_seeds(SeedConfig(seed=42))
        val2 = random.random()
        
        assert val1 == val2
    
    def test_sets_numpy_seed(self):
        """Test that NumPy random seed is set."""
        set_all_seeds(SeedConfig(seed=42))
        val1 = np.random.rand()
        
        set_all_seeds(SeedConfig(seed=42))
        val2 = np.random.rand()
        
        assert val1 == val2
    
    def test_sets_torch_seed(self):
        """Test that PyTorch seed is set."""
        set_all_seeds(SeedConfig(seed=42))
        val1 = torch.rand(1).item()
        
        set_all_seeds(SeedConfig(seed=42))
        val2 = torch.rand(1).item()
        
        assert val1 == val2
    
    def test_different_seeds_produce_different_values(self):
        """Test that different seeds produce different random values."""
        set_all_seeds(SeedConfig(seed=0))
        val1 = torch.rand(1).item()
        
        set_all_seeds(SeedConfig(seed=1))
        val2 = torch.rand(1).item()
        
        assert val1 != val2
    
    def test_deterministic_mode(self):
        """Test that deterministic mode is set correctly."""
        cfg = SeedConfig(seed=42, deterministic=True)
        set_all_seeds(cfg)
        
        # Create two identical operations
        x1 = torch.randn(10, 10)
        y1 = torch.randn(10, 10)
        result1 = torch.matmul(x1, y1)
        
        # Reset seed and repeat
        set_all_seeds(cfg)
        x2 = torch.randn(10, 10)
        y2 = torch.randn(10, 10)
        result2 = torch.matmul(x2, y2)
        
        # Should be identical
        assert torch.allclose(x1, x2)
        assert torch.allclose(y1, y2)
        assert torch.allclose(result1, result2)
    
    def test_all_rngs_synchronized(self):
        """Test that all RNG states are synchronized."""
        seed = 123
        set_all_seeds(SeedConfig(seed=seed))
        
        # Generate values from all RNGs
        py_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()
        
        # Reset and generate again
        set_all_seeds(SeedConfig(seed=seed))
        
        assert py_val == random.random()
        assert np_val == np.random.rand()
        assert torch_val == torch.rand(1).item()
    
    def test_sequence_determinism(self):
        """Test that sequences of random values are deterministic."""
        set_all_seeds(SeedConfig(seed=42))
        sequence1 = [torch.rand(1).item() for _ in range(10)]
        
        set_all_seeds(SeedConfig(seed=42))
        sequence2 = [torch.rand(1).item() for _ in range(10)]
        
        assert sequence1 == sequence2
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_seed_set(self):
        """Test that CUDA seed is set when available."""
        set_all_seeds(SeedConfig(seed=42))
        val1 = torch.cuda.FloatTensor(1).normal_().item()
        
        set_all_seeds(SeedConfig(seed=42))
        val2 = torch.cuda.FloatTensor(1).normal_().item()
        
        assert val1 == val2
