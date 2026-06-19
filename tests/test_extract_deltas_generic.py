"""Tests for extract_layer_deltas_generic."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from experiments.concept_localization.concept_pair import ConceptPair
from experiments.concept_localization.extract_deltas_generic import (
    extract_layer_deltas_generic,
)


def _make_model(n_layers: int = 4, d_model: int = 8, seq_len: int = 6):
    """Minimal mock model accepted by extract_layer_deltas_generic."""

    class _Tokenizer:
        # Attributes required by tokenize_qwen_input
        pad_token_id = 0
        bos_token_id = None
        eos_token_id = None
        all_special_ids = [0]

        def __call__(self, text, add_special_tokens=False):
            # Start ids from 1 so token 0 (pad/sink) is not in raw sequence
            return SimpleNamespace(input_ids=list(range(1, seq_len + 1)))

        def convert_ids_to_tokens(self, ids):
            return [str(i) for i in ids]

    def _run_with_hooks(input_ids, fwd_hooks):
        acts = torch.randn(1, input_ids.shape[1], d_model)
        for hook_name, hook_fn in fwd_hooks:
            hook_fn(acts, hook_name)

    model = MagicMock()
    model.cfg.n_layers = n_layers
    model.cfg.device = "cpu"
    model.tokenizer = _Tokenizer()
    model.run_with_hooks.side_effect = _run_with_hooks
    return model


def _pairs(n: int, template: str = "T0") -> list[ConceptPair]:
    return [
        ConceptPair(
            prompt_pos=f"calc: {10 + i}+{10 + i}= ",
            prompt_neg=f"calc: {10 + i}+{i}= ",
            label_pos="carry",
            label_neg="no_carry",
            template=template,
        )
        for i in range(n)
    ]


def test_pos_neg_counts_equal():
    """pos and neg bucket lengths must always be equal after extraction."""
    model = _make_model()
    pairs = _pairs(10)
    layers = [0, 1, 2, 3]

    results = extract_layer_deltas_generic(
        model,
        pairs,
        layers,
        device=torch.device("cpu"),
        dtype=torch.float32,
        per_template=False,
        anchor_mode="last",
    )

    ld = results["all"]
    assert ld.n_pairs == 10
    for layer in layers:
        assert layer in ld.delta
        assert ld.delta[layer].shape == (8,)


def test_pos_neg_counts_equal_with_skipped():
    """Pairs with mismatched token lengths are skipped; remaining counts still match."""

    class _UnevenTokenizer:
        """Returns different lengths for pos vs neg on the first pair."""

        pad_token_id = 0
        bos_token_id = None
        eos_token_id = None
        all_special_ids = [0]

        def __init__(self):
            self._call_count = 0

        def __call__(self, text, add_special_tokens=False):
            self._call_count += 1
            # First pos/neg pair: different lengths → should be skipped
            length = 5 if self._call_count <= 2 else 6
            if self._call_count == 2:
                length = 7  # mismatch for first pair
            return SimpleNamespace(input_ids=list(range(1, length + 1)))

        def convert_ids_to_tokens(self, ids):
            return [str(i) for i in ids]

    model = _make_model()
    model.tokenizer = _UnevenTokenizer()

    pairs = _pairs(5)
    layers = [0, 1]

    results = extract_layer_deltas_generic(
        model,
        pairs,
        layers,
        device=torch.device("cpu"),
        dtype=torch.float32,
        per_template=False,
        anchor_mode="last",
    )

    ld = results["all"]
    # First pair was skipped; remaining 4 should be equal on both sides
    assert ld.skipped == 1
    assert ld.n_pairs == 4

def test_default_delimiter_anchor_raises_when_no_delimiter():
    """Default delimiter mode should fail loudly instead of falling back to last token."""
    model = _make_model()
    pairs = _pairs(1)

    with pytest.raises(ValueError, match="Could not resolve delimiter anchor"):
        extract_layer_deltas_generic(
            model,
            pairs,
            [0],
            device=torch.device("cpu"),
            dtype=torch.float32,
            per_template=False,
        )
