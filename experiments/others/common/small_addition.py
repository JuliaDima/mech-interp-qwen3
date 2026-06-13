"""Small-addition-model helpers shared by miscellaneous experiments."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from experiments.addition.dataset_generation.generate_dataset_with_predictions import (
    TEMPLATES,
    TemplateID,
)
log = logging.getLogger("small_addition")


class SmallAdditionTransformer(nn.Module):
    """Small transformer wrapper used by downstream addition experiments."""

    def __init__(
        self,
        n_layers: int = 2,
        n_heads: int = 3,
        d_model: int = 256,
        vocab_size: int = 15,
        max_seq_len: int = 64,
        device: torch.device | str | None = None,
        use_rope: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.use_rope = use_rope

        d_head = d_model // n_heads
        if use_rope and d_head % 2 != 0:
            raise ValueError(
                f"RoPE requires d_head to be even, but got d_head={d_head} "
                f"(d_model={d_model}, n_heads={n_heads})."
            )

        from transformer_lens import HookedTransformer, HookedTransformerConfig

        config = HookedTransformerConfig(
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_head=d_head,
            d_mlp=d_model * 4,
            d_vocab=vocab_size,
            n_ctx=max_seq_len,
            act_fn="gelu",
            normalization_type="LN",
            positional_embedding_type="rotary" if use_rope else "standard",
            device=str(device) if device is not None else "cpu",
        )
        self.model = HookedTransformer(config)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens)


_QM_VOCAB: list[str] = [str(i) for i in range(10)] + ["+", "-", "=", "P", "M"]
_QM_CHAR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(_QM_VOCAB)}
_QM_CHAR_TO_IDX["+"] = _QM_VOCAB.index("+")
_QM_ANSWER_PLUS_IDX = 13
_QM_ANSWER_MINUS_IDX = 14


def _qm_tokenize(text: str) -> list[int]:
    tokens: list[int] = []
    i = 0
    eq_seen = False
    while i < len(text):
        ch = text[i]
        if ch == "=":
            tokens.append(_QM_CHAR_TO_IDX["="])
            eq_seen = True
        elif eq_seen and i == text.index("=") + 1:
            tokens.append(_QM_ANSWER_PLUS_IDX if ch == "+" else _QM_ANSWER_MINUS_IDX)
        elif ch.isdigit():
            tokens.append(int(ch))
        elif ch == "+":
            tokens.append(_QM_CHAR_TO_IDX["+"])
        elif ch == "-":
            tokens.append(_QM_CHAR_TO_IDX["-"])
        i += 1
    return tokens


def load_addition_dataset(
    dataset_path: str,
    max_samples: int | None = None,
    num_digits: int = 5,
) -> list[dict[str, Any]]:
    """Load an addition JSONL dataset, or generate deterministic fallback samples."""
    path = Path(dataset_path)
    samples: list[dict[str, Any]] = []

    if path.exists():
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    break
        return samples

    log.warning(
        "Dataset not found at %s; generating %d-digit samples on the fly",
        dataset_path,
        num_digits,
    )
    fallback_n = min(max_samples if max_samples else 200_000, 50_000)
    random.seed(42)
    seen: set[tuple[int, int]] = set()
    while len(samples) < fallback_n:
        d_a = random.randint(1, num_digits)
        d_b = random.randint(1, num_digits)
        a = random.randint(10 ** (d_a - 1) if d_a > 1 else 0, 10**d_a - 1)
        b = random.randint(10 ** (d_b - 1) if d_b > 1 else 0, 10**d_b - 1)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        samples.append(
            {
                "prompt": TEMPLATES[TemplateID.T0].format(a=a, b=b),
                "answer": str(a + b),
                "a": a,
                "b": b,
                "template": "T0",
            }
        )
    return samples


def get_small_model_tokenizer(model: Any, max_len: int | None = None):
    """Return a tokenizer for QuantaMaths or scratch small-addition models."""
    n_ctx = max_len if max_len is not None else model.model.cfg.n_ctx

    if hasattr(model, "_tokenizer"):
        tok_fn = model._tokenizer

        def tokenize_qm(text: str, max_l: int = n_ctx) -> list[int]:
            toks = tok_fn(text)
            if len(toks) < max_l:
                toks += [0] * (max_l - len(toks))
            return toks[:max_l]

        return tokenize_qm

    vocab = ["<PAD>", "<BOS>", "<EOS>"] + [str(i) for i in range(10)] + ["+", "=", " "]
    c2i = {c: i for i, c in enumerate(vocab)}

    def tokenize_scratch(text: str, max_l: int = n_ctx) -> list[int]:
        toks = [c2i.get(c, 0) for c in text]
        if len(toks) < max_l:
            toks += [0] * (max_l - len(toks))
        return toks[:max_l]

    return tokenize_scratch


def _load_small_model(
    args: argparse.Namespace, path: Path, device: torch.device
) -> SmallAdditionTransformer:
    log.info("Loading small model from %s", path)
    sd = torch.load(path, map_location=device)

    first_key = next(iter(sd.keys()))
    has_prefix = first_key.startswith("model.")

    def _get(key: str):
        full_key = f"model.{key}" if has_prefix else key
        return sd[full_key]

    vocab_size = _get("embed.W_E").shape[0]
    d_model = _get("embed.W_E").shape[1]
    use_rope = any(k.endswith("attn.rotary_sin") for k in sd)
    n_ctx = _get("blocks.0.attn.rotary_sin").shape[0] if use_rope else _get("pos_embed.W_pos").shape[0]
    prefix = "model.blocks." if has_prefix else "blocks."
    n_layers = sum(1 for k in sd if k.startswith(prefix) and k.endswith(".ln1.w"))
    n_heads = _get("blocks.0.attn.W_Q").shape[0]

    model = SmallAdditionTransformer(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        vocab_size=vocab_size,
        max_seq_len=n_ctx,
        device=device,
        use_rope=use_rope,
    )

    if has_prefix:
        model.load_state_dict(sd)
    else:
        model.model.load_state_dict(sd)

    model.to(device)
    hub_model_id = getattr(args, "hub_model", "").strip()
    if hub_model_id and vocab_size == 15:
        import re

        match = re.search(r"_d(\d+)_", hub_model_id)
        model._n_digits = int(match.group(1)) if match else (n_ctx - 4) // 3  # type: ignore[attr-defined]
        model._tokenizer = _qm_tokenize  # type: ignore[attr-defined]
        log.info("Restored QuantaMaths tokenizer (n_digits=%d)", model._n_digits)

    return model


def _load_small_sae(
    args: argparse.Namespace, path: Path, device: torch.device
) -> SingleLayerTranscoder:
    from mechinterp_qwen3.transcoder.single_layer_transcoder import load_relu_transcoder

    log.info("Loading small SAE from %s", path)
    sae_layer = (
        args.small_sae_layer if args.small_sae_layer is not None else args.small_model_layers - 1
    )
    return load_relu_transcoder(str(path), layer=sae_layer, lazy_encoder=False, lazy_decoder=False)
