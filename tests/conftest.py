"""Shared pytest fixtures/helpers."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def tokenizer_reachable(name: str) -> bool:
    """Whether an HF tokenizer can actually be loaded (cached locally or fetchable).

    Even the "tiny" synthetic-weight test configs still need a real tokenizer
    (e.g. gpt2), so tests using them can't run in an environment with neither
    a local HF cache nor network access -- true for a plain CI runner, but not
    for this project's usual HPC/RDS environment (see CLAUDE.md).
    """
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(name)
        return True
    except Exception:
        return False
