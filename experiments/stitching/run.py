"""Model stitching experiment for addition circuits — Option B (SAE-mediated).

This experiment trains a small SingleLayerTranscoder (SAE) on a small addition model
(Quirke & Barez, ICLR 2024), fits an affine map from small-model MLP outputs to
large-model (Qwen3-4B) MLP outputs, and patches Qwen3-4B's hook_mlp_out to test
whether the small model's carry circuits transfer.

Background:
    - Chen et al. (arXiv 2506.06609) demonstrate that affine maps faithfully transfer
      features between models of different sizes.  We apply this at the MLP-output
      level rather than the residual-stream level, which is more targeted and avoids
      the residual-stream `interference` discussed in that paper.
    - Quirke & Barez (ICLR 2024) show that carry computation in small addition
      transformers is maximally active at the '=' token position.

Pipeline:
    1. Train small addition model          (Quirke & Barez, ICLR 2024)
    1.5. Train small SAE on small model's MLP layer
    2. Collect small SAE reconstructed MLP outputs at '='
    3. Collect large model (Qwen3-4B) MLP outputs at '='
    4. Fit affine map: small_mlp_out (d_small) -> large_mlp_out (d_large)
    5. Inject via hook_mlp_out patching; measure accuracy + KL
    6. Compare attribution graphs before/after stitching (placeholder)

Usage:
    python experiments/stitching/run.py --all
    python experiments/stitching/run.py --all --dry-run
    python experiments/stitching/run.py --train-small
    python experiments/stitching/run.py --train-sae

References:
    - Quirke & Barez (ICLR 2024): https://arxiv.org/abs/2310.13121
    - Chen et al. (arXiv 2506.06609): Affine stitching methodology
    - Anthropic (2025): Biology paper for attribution graph comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.addition.dataset_generation.generate_dataset_with_predictions import (  # noqa: E402
    TEMPLATES,
    TemplateID,
)
from mechinterp_qwen3.attribution_model import AttributionModel  # noqa: E402
from mechinterp_qwen3.transcoder.single_layer_transcoder import SingleLayerTranscoder  # noqa: E402
from mechinterp_qwen3.utils.config_utils import (  # noqa: E402
    add_config_args,
    load_config,
    print_config,
    set_parser_defaults_from_config,
)
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub  # noqa: E402
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype  # noqa: E402
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input  # noqa: E402
from mechinterp_qwen3.utils_seed import seed_everything  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stitching.run")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

from experiments.stitching.utils import (  # noqa: E402
    get_small_model_tokenizer,
    identify_cascading_carry_cases,
    load_addition_dataset,
    plot_stitching_results,
)

# ---------------------------------------------------------------------------
# Step 1: Train Small Addition Model
# ---------------------------------------------------------------------------


class SmallAdditionTransformer(nn.Module):
    """Small transformer for learning addition, following Quirke & Barez (ICLR 2024).

    Args:
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_model: Model dimension
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        device: Device to place model on
        use_rope: If True, use Rotary Position Embeddings (RoPE) instead of learned
                  absolute positional embeddings. RoPE is used by Qwen models and may
                  improve alignment for stitching experiments.
    """

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

        # RoPE requires d_head to be even (rotates pairs of dimensions)
        d_head = d_model // n_heads
        if use_rope and d_head % 2 != 0:
            raise ValueError(
                f"RoPE requires d_head to be even, but got d_head={d_head} "
                f"(d_model={d_model}, n_heads={n_heads}). "
                f"Please adjust d_model or n_heads so that d_model/n_heads is even."
            )

        _device_str = str(device) if device is not None else "cpu"
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
            device=_device_str,
        )
        self.model = HookedTransformer(config)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens)


# ---------------------------------------------------------------------------
# Load QuantaMaths pretrained model from HuggingFace (PhilipQuirke/QuantaMaths_*)
# ---------------------------------------------------------------------------

# QuantaMaths vocabulary (15 tokens, per-digit tokenization).
# Format: "33357+82243=+115600"  (5-digit: each char is one token)
# Token ordering as observed from embed.W_E shape (15,) in the released model.pth.
_QM_VOCAB: list[str] = [str(i) for i in range(10)] + ["+", "-", "=", "P", "M"]
# P = positive sign (+) prefix on answer, M = negative sign (-) prefix on answer
_QM_CHAR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(_QM_VOCAB)}
# Map both sign characters to the same tokens used in training
_QM_CHAR_TO_IDX["+"] = _QM_VOCAB.index("+")  # "+" operator
# Answer sign: Quirke uses tokens at index 13 (+) and 14 (-) for the answer sign
_QM_ANSWER_PLUS_IDX = 13
_QM_ANSWER_MINUS_IDX = 14


def _qm_tokenize(text: str) -> list[int]:
    """Tokenize a QuantaMaths question string to token indices.

    Input format: "33357+82243=+115600" (per-digit, one char per token).
    The answer sign (+ or -) uses dedicated vocab entries 13 and 14.
    """
    tokens: list[int] = []
    i = 0
    eq_seen = False
    while i < len(text):
        ch = text[i]
        if ch == "=":
            tokens.append(_QM_CHAR_TO_IDX["="])
            eq_seen = True
        elif eq_seen and i == text.index("=") + 1:
            # First character after '=' is the answer sign
            tokens.append(_QM_ANSWER_PLUS_IDX if ch == "+" else _QM_ANSWER_MINUS_IDX)
        elif ch.isdigit():
            tokens.append(int(ch))
        elif ch == "+":
            tokens.append(_QM_CHAR_TO_IDX["+"])
        elif ch == "-":
            tokens.append(_QM_CHAR_TO_IDX["-"])
        i += 1
    return tokens


def _qm_make_sample(a: int, b: int, n_digits: int) -> dict[str, Any]:
    """Create a dict sample in QuantaMaths format."""
    total = a + b
    a_str = str(a).zfill(n_digits)
    b_str = str(b).zfill(n_digits)
    ans_digits = n_digits + 1  # n-digit + n-digit can produce (n+1)-digit answer
    ans_str = "+" + str(total).zfill(ans_digits)  # always positive for addition
    prompt = f"{a_str}+{b_str}="
    full = f"{a_str}+{b_str}={ans_str}"
    return {
        "prompt": prompt,
        "full": full,
        "a": a,
        "b": b,
        "answer": str(total),
        "a_str": a_str,
        "b_str": b_str,
    }


def _qm_to_large_dicts(
    small_extraction_samples: list[str],
    large_model: AttributionModel,
    n_digits: int,
) -> list[dict[str, Any]]:
    """Convert QuantaMaths extraction strings into Qwen3-4B prompt dicts.

    Parses "DDDDD+DDDDD=+DDDDDD" into (a, b) and formats using the large
    model's own tokenizer (same chat/math prompt style as the dataset).
    This ensures that both models will process *identical* addition problems.
    """
    dicts: list[dict[str, Any]] = []
    for text in small_extraction_samples:
        eq = text.index("=")
        plus = text.index("+")
        a = int(text[:plus])
        b = int(text[plus + 1 : eq])
        total = a + b
        # Qwen3-4B prompt format matches T0 template
        prompt = TEMPLATES[TemplateID.T0].format(a=a, b=b)
        dicts.append({"prompt": prompt, "a": a, "b": b, "answer": str(total)})
    return dicts


def load_quanta_maths_model(
    hub_model_id: str,
    device: torch.device,
    n_digits: int | None = None,
) -> tuple[SmallAdditionTransformer, int]:
    """Download and load a QuantaMaths pretrained model from HuggingFace.

    The model is stored as a raw TransformerLens state-dict in ``model.pth``.
    We reconstruct the HookedTransformerConfig from the weight shapes, load the
    weights, and wrap in ``SmallAdditionTransformer``.

    Naming convention: ``QuantaMaths_add_d5_l1_h3_t15K_s372001``
      · d5 = 5 digits, l2 = 2 layers, h3 = 3 heads, t15K = 15k steps

    Tokenizer is the QuantaMaths per-digit scheme (vocab size = 15).

    Returns:
        (model, n_digits) where n_digits is parsed from the hub_model_id or the
        weight shapes.
    """
    from huggingface_hub import hf_hub_download

    log.info("Downloading QuantaMaths model: %s", hub_model_id)
    pth_path = hf_hub_download(hub_model_id, "model.pth")
    sd = torch.load(pth_path, map_location="cpu")

    # Infer architecture from weight shapes
    W_E: torch.Tensor = sd["embed.W_E"]  # (vocab_size, d_model)
    W_pos: torch.Tensor = sd["pos_embed.W_pos"]  # (n_ctx, d_model)
    W_Q0: torch.Tensor = sd["blocks.0.attn.W_Q"]  # (n_heads, d_model, d_head)
    W_in: torch.Tensor = sd["blocks.0.mlp.W_in"]  # (d_model, d_mlp)

    vocab_size = W_E.shape[0]
    d_model = W_E.shape[1]
    n_ctx = W_pos.shape[0]
    n_heads = W_Q0.shape[0]
    d_head = W_Q0.shape[2]
    d_mlp = W_in.shape[1]
    n_layers = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".ln1.w"))

    # Infer n_digits: n_ctx = 2*n_digits + 2 (question) + n_digits + 2 (answer) = 3*n_digits+4
    # Actually n_ctx = (2*n_digits + 2) + (n_digits + 2) = 3*n_digits + 4
    if n_digits is None:
        import re

        m = re.search(r"_d(\d+)_", hub_model_id)
        n_digits = int(m.group(1)) if m else (n_ctx - 4) // 3

    log.info(
        "QuantaMaths model: d_model=%d, n_layers=%d, n_heads=%d, d_head=%d, "
        "d_mlp=%d, vocab=%d, n_ctx=%d, n_digits=%d",
        d_model,
        n_layers,
        n_heads,
        d_head,
        d_mlp,
        vocab_size,
        n_ctx,
        n_digits,
    )

    # Wrap in SmallAdditionTransformer, then load weights
    wrapper = SmallAdditionTransformer(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        vocab_size=vocab_size,
        max_seq_len=n_ctx,
        device=device,
    )
    # HookedTransformer uses strict=False to allow mask/IGNORE keys (non-param buffers)
    missing, unexpected = wrapper.model.load_state_dict(sd, strict=False)
    if missing:
        log.warning("Missing keys when loading QuantaMaths weights: %s", missing)
    log.info("QuantaMaths model loaded successfully from %s", hub_model_id)

    wrapper.model.to(device)
    wrapper.model.eval()

    wrapper._n_digits = n_digits  # type: ignore[attr-defined]
    wrapper._tokenizer = _qm_tokenize  # type: ignore[attr-defined]
    wrapper._make_sample = _qm_make_sample  # type: ignore[attr-defined]
    return wrapper, n_digits


def train_small_model(
    n_layers: int,
    n_heads: int,
    d_model: int,
    epochs: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    num_digits: int = 5,
    dry_run: bool = False,
    use_rope: bool = False,
) -> tuple[SmallAdditionTransformer, list[str], list[str], list[str]]:
    """Train a small transformer on n-digit addition (Quirke & Barez, ICLR 2024).

    Format: "12345+67890=+080235" (QuantaMaths format: zero-padded with answer sign).

    Args:
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_model: Model dimension
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        dtype: Data type for model
        num_digits: Number of digits for addition problems
        dry_run: If True, use small dataset for quick testing
        use_rope: If True, use Rotary Position Embeddings (RoPE) instead of standard
                  absolute positional embeddings. RoPE may improve alignment with Qwen.

    Returns:
        (model, train_samples_str, val_samples_str, ood_samples_str)
    """
    import random

    positional_type = "RoPE" if use_rope else "absolute"
    log.info(
        "Training small addition model (Quirke & Barez, ICLR 2024) with %s positional embeddings",
        positional_type,
    )

    # QuantaMaths vocabulary (15 tokens exactly)
    # 0-9, +, -, =, P (answer +), M (answer -)
    vocab = [str(i) for i in range(10)] + ["+", "-", "=", "P", "M"]
    vocab_size = len(vocab)  # Must be exactly 15
    assert vocab_size == 15, f"QuantaMaths requires exactly 15 tokens, got {vocab_size}"

    # Map characters to indices
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    # Answer signs use dedicated tokens at the end
    answer_plus_idx = vocab.index("P")  # Token 13
    answer_minus_idx = vocab.index("M")  # Token 14

    model = SmallAdditionTransformer(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        vocab_size=vocab_size,
        device=device,
        use_rope=use_rope,
    )
    model.model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    random.seed(42)
    max_val = 10**num_digits - 1

    num_train = (
        50_000 if not dry_run else 100
    )  # 50k covers 5% of 1M 3-digit pairs #TODO: remove hardcoded values and use config
    num_val = 500 if not dry_run else 20
    num_ood = 200 if not dry_run else 20

    log.info("Generating %d-digit addition data (range [0, %d])", num_digits, max_val)

    def make_sample(a: int, b: int) -> str:
        """Format sample in QuantaMaths format: "12345+67890=+080235"."""
        total = a + b
        a_str = str(a).zfill(num_digits)
        b_str = str(b).zfill(num_digits)
        ans_digits = num_digits + 1  # n-digit + n-digit can produce (n+1)-digit
        ans_str = "+" + str(total).zfill(ans_digits)  # Always positive for addition
        return f"{a_str}+{b_str}={ans_str}"

    train_samples = []
    for _ in range(num_train):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        train_samples.append(make_sample(a, b))

    val_samples: list[str] = []
    val_set = set(train_samples)
    attempts = 0
    while len(val_samples) < num_val and attempts < num_val * 10:
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        t = make_sample(a, b)
        if t not in val_set:
            val_samples.append(t)
            val_set.add(t)
        attempts += 1

    ood_min = 10**num_digits  # TODO: remove hardcoded values and use config. Print ood grid.
    ood_max = 10 ** (num_digits + 1) - 1
    ood_samples = []
    for _ in range(num_ood):
        a = random.randint(ood_min, ood_max)
        b = random.randint(ood_min, ood_max)
        ood_samples.append(make_sample(a, b))

    if dry_run:
        train_samples = train_samples[:10]
        val_samples = val_samples[:5]
        ood_samples = ood_samples[:5]
        epochs = min(epochs, 100)

    def tokenize(text: str) -> list[int]:
        """Tokenize QuantaMaths format: "12345+67890=+080235"."""
        tokens: list[int] = []
        eq_idx = text.index("=")
        for i, ch in enumerate(text):
            if ch == "=":
                tokens.append(char_to_idx["="])
            elif i == eq_idx + 1:
                # First char after '=' is answer sign (+ or -)
                tokens.append(answer_plus_idx if ch == "+" else answer_minus_idx)
            elif ch.isdigit():
                tokens.append(int(ch))
            elif ch == "+":
                tokens.append(char_to_idx["+"])
            elif ch == "-":
                tokens.append(char_to_idx["-"])
        return tokens

    def evaluate(split: list[str], max_eval: int = 200) -> float:
        """Batch-evaluate accuracy on up to max_eval samples."""
        model.model.eval()
        subset = split[:max_eval]
        # Pad all sequences to the same length for batched inference
        ids_list = [tokenize(t) for t in subset]
        max_len = max(len(ids) for ids in ids_list)
        # Pad with zeros (or could use a specific PAD token if we had one)
        ids_batch = torch.zeros((len(ids_list), max_len), device=device, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            ids_batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        correct = total = 0
        with torch.no_grad():
            logits_batch = model.model(ids_batch)  # (B, L, V) — single forward pass
        for i, text in enumerate(subset):
            eq = text.find("=")
            if eq == -1:
                continue
            # Answer starts after '=' sign
            answer_part = text[eq + 1 :]
            for k, ch in enumerate(answer_part):
                pos = eq + k
                if pos < logits_batch.shape[1] - 1:
                    pred = int(logits_batch[i, pos].argmax())
                    # First character is answer sign (P or M)
                    if k == 0:
                        expected = answer_plus_idx if ch == "+" else answer_minus_idx
                    elif ch.isdigit():
                        expected = int(ch)
                    else:
                        continue
                    if pred == expected:
                        correct += 1
                    total += 1
        model.model.train()
        return 100.0 * correct / total if total > 0 else 0.0

    batch_size = 32
    best_val_acc = 0.0

    for epoch in range(epochs):
        random.shuffle(train_samples)
        total_loss = 0.0

        for i in range(0, len(train_samples), batch_size):
            batch = train_samples[i : i + batch_size]
            # Pad batch to same length (use zeros, which is digit '0')
            ids_list = [tokenize(t) for t in batch]
            max_len = max(len(ids) for ids in ids_list)
            ids_batch = torch.zeros((len(ids_list), max_len), device=device, dtype=torch.long)
            for j, ids in enumerate(ids_list):
                ids_batch[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)

            logits = model.model(ids_batch)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = ids_batch[:, 1:].contiguous()
            # No ignore_index since all tokens (including padding 0s) are valid
            loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            val_acc = evaluate(val_samples)
            ood_acc = evaluate(ood_samples)
            avg_loss = total_loss / len(train_samples)
            log.info(
                "Epoch %d/%d: Loss=%.4f Val=%.2f%% OOD=%.2f%%",
                epoch + 1,
                epochs,
                avg_loss,
                val_acc,
                ood_acc,
            )
            best_val_acc = max(best_val_acc, val_acc)

    log.info("Training complete. Best val accuracy: %.2f%%", best_val_acc)

    model._n_digits = num_digits  # type: ignore[attr-defined]
    model._tokenizer = tokenize  # type: ignore[attr-defined]
    model._make_sample = make_sample  # type: ignore[attr-defined]

    return model, train_samples, val_samples, ood_samples


# ---------------------------------------------------------------------------
# Step 1.5: Train Small SAE (SingleLayerTranscoder) on small model MLP outputs
# ---------------------------------------------------------------------------


def train_small_sae(
    small_model: SmallAdditionTransformer,
    small_extraction_samples: list[str],
    device: torch.device,
    d_transcoder: int = 4096,
    epochs: int = 500,
    lr: float = 1e-3,
    sae_layer: int | None = None,
    dry_run: bool = False,
) -> SingleLayerTranscoder:
    """Train a SingleLayerTranscoder (SAE) on the small model's MLP outputs.

    The SAE learns to reconstruct hook_mlp_out from hook_resid_mid at the '='
    token position.  This matches the large model's TranscoderSet convention:
        feature_input_hook  = 'mlp.hook_in'  (≡ hook_resid_mid)
        feature_output_hook = 'mlp.hook_out' (≡ hook_mlp_out)

    Args:
        small_model: Trained SmallAdditionTransformer
        small_extraction_samples: list[str] of "a+b=answer" strings
        d_transcoder: SAE feature count.  Default 4096 ≈ 16× d_model (healthy ratio).
        sae_layer: Which small-model layer to fit (default: n_layers - 1, i.e. last)
        dry_run: Use minimal data / epochs
    """
    if sae_layer is None:
        sae_layer = small_model.n_layers - 1
    d_model = small_model.d_model

    log.info(
        "Training small SAE: layer=%d, d_model=%d, d_transcoder=%d",
        sae_layer,
        d_model,
        d_transcoder,
    )

    tokenize = get_small_model_tokenizer(small_model)

    samples = small_extraction_samples
    if dry_run:
        samples = samples[:20]
        epochs = min(epochs, 50)

    # ---- Collect paired (resid_mid, mlp_out) at '=' ----
    log.info("Collecting (resid_mid, mlp_out) pairs from small model for SAE training…")
    small_model.model.eval()
    mlp_in_list: list[torch.Tensor] = []
    mlp_out_list: list[torch.Tensor] = []

    with torch.no_grad():
        for text in tqdm(samples, desc="Collecting SAE training data"):
            eq = text.find("=")
            if eq == -1:
                continue
            ids = torch.tensor([tokenize(text)], device=device, dtype=torch.long)
            cache: dict[str, torch.Tensor] = {}

            def _hook_in(act: torch.Tensor, hook, _pos: int = eq, cache=cache) -> torch.Tensor:
                cache["in"] = act[:, _pos, :].clone()
                return act

            def _hook_out(act: torch.Tensor, hook, _pos: int = eq, cache=cache) -> torch.Tensor:
                cache["out"] = act[:, _pos, :].clone()
                return act

            with small_model.model.hooks(
                fwd_hooks=[
                    (f"blocks.{sae_layer}.hook_resid_mid", _hook_in),
                    (f"blocks.{sae_layer}.hook_mlp_out", _hook_out),
                ]
            ):
                small_model.model(ids)

            if "in" in cache and "out" in cache:
                mlp_in_list.append(cache["in"].squeeze(0))
                mlp_out_list.append(cache["out"].squeeze(0))

    if not mlp_in_list:
        raise ValueError("No training data collected for SAE — check sample format.")

    X = torch.stack(mlp_in_list).to(device=device, dtype=torch.float32)  # (n, d_model)
    Y = torch.stack(mlp_out_list).to(device=device, dtype=torch.float32)  # (n, d_model)
    log.info(
        "SAE training data: %d samples, MLP-in range [%.3f, %.3f]",
        X.shape[0],
        X.min().item(),
        X.max().item(),
    )

    # ---- Create and train SAE ----
    # NOTE: SingleLayerTranscoder inits all weights to zero,
    # use a hand-rolled module with proper Kaiming init, then copy weights back.
    class _SimpleSAE(nn.Module):
        """Minimal ReLU SAE (encode→ReLU→decode) with Kaiming weight init."""

        def __init__(self, d_in: int, d_tc: int):
            super().__init__()
            self.W_enc = nn.Parameter(torch.empty(d_in, d_tc))
            self.b_enc = nn.Parameter(torch.zeros(d_tc))
            self.W_dec = nn.Parameter(torch.empty(d_tc, d_in))
            self.b_dec = nn.Parameter(torch.zeros(d_in))
            nn.init.kaiming_uniform_(self.W_enc, a=0.01)
            nn.init.kaiming_uniform_(self.W_dec, a=0.01)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return F.relu(x @ self.W_enc + self.b_enc)

        def decode(self, f: torch.Tensor) -> torch.Tensor:
            return f @ self.W_dec + self.b_dec

    sae = _SimpleSAE(d_model, d_transcoder).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    batch_size = min(128, X.shape[0])
    best_loss = float("inf")

    sae.train()
    for epoch in range(epochs):
        perm = torch.randperm(X.shape[0], device=device)
        epoch_loss = 0.0
        n_active_sum = 0.0
        n_batches = 0

        for start in range(0, X.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            x_b, y_b = X[idx], Y[idx]

            features = sae.encode(x_b)  # (B, d_tc)
            recon = sae.decode(features)  # (B, d_model)

            recon_loss = F.mse_loss(recon, y_b)
            l1_loss = features.abs().mean()
            loss = recon_loss + 1e-4 * l1_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()

            epoch_loss += recon_loss.item() * len(idx)
            n_active_sum += (features > 0).float().sum(dim=-1).mean().item()
            n_batches += 1

        avg_loss = epoch_loss / X.shape[0]
        avg_active = n_active_sum / max(n_batches, 1)
        best_loss = min(best_loss, avg_loss)

        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            log.info(
                "SAE Epoch %d/%d: ReconLoss=%.4f  ActiveFeatures≈%.1f/%d",
                epoch + 1,
                epochs,
                avg_loss,
                avg_active,
                d_transcoder,
            )

    log.info("SAE training complete. Best recon loss: %.4f", best_loss)

    # Copy trained weights into a SingleLayerTranscoder for save/load compatibility
    # (SingleLayerTranscoder W_enc shape: (d_tc, d_in); _SimpleSAE W_enc: (d_in, d_tc))
    slt = SingleLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=F.relu,
        layer_idx=sae_layer,
        skip_connection=False,
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        # _SimpleSAE: W_enc (d_in, d_tc), W_dec (d_tc, d_in)
        # SingleLayerTranscoder: W_enc (d_tc, d_in), W_dec (d_tc, d_in)
        _dt = slt.W_enc.dtype  # cast to whatever dtype SLT uses
        slt.W_enc.copy_(sae.W_enc.T.to(_dt))  # (d_in, d_tc) -> (d_tc, d_in)
        slt.W_dec.copy_(sae.W_dec.to(_dt))  # (d_tc, d_in) same shape
        slt.b_enc.copy_(sae.b_enc.to(_dt))
        slt.b_dec.copy_(sae.b_dec.to(_dt))
    slt.eval()
    return slt


# ---------------------------------------------------------------------------
# Step 2: Collect small SAE reconstructed MLP outputs at '='
# ---------------------------------------------------------------------------


def collect_small_sae_outputs(
    small_model: SmallAdditionTransformer,
    small_sae: SingleLayerTranscoder,
    samples: list[str],
    device: torch.device,
    sae_layer: int | None = None,
) -> torch.Tensor:
    """Run small model → SAE encode → decode → return reconstructed MLP outputs.

    Returns tensor of shape [n_samples, d_model_small].
    """
    if sae_layer is None:
        sae_layer = small_model.n_layers - 1

    tokenize = get_small_model_tokenizer(small_model)

    outputs: list[torch.Tensor] = []
    small_model.model.eval()
    small_sae.eval()

    with torch.no_grad():
        for text in tqdm(samples, desc="Collecting small SAE outputs"):
            eq = text.find("=")
            if eq == -1:
                continue
            ids = torch.tensor([tokenize(text)], device=device, dtype=torch.long)
            cache: dict[str, torch.Tensor] = {}

            def _hook(act: torch.Tensor, hook, _pos: int = eq, cache=cache) -> torch.Tensor:
                cache["v"] = act[:, _pos, :].clone()
                return act

            with small_model.model.hooks(fwd_hooks=[(f"blocks.{sae_layer}.hook_resid_mid", _hook)]):
                small_model.model(ids)

            if "v" not in cache:
                continue

            resid_mid = cache["v"].to(dtype=torch.float32)  # (1, d_small)
            features = small_sae.encode(resid_mid)  # (1, d_tc)
            recon = small_sae.decode(features)  # (1, d_small)
            outputs.append(recon.squeeze(0).cpu())

    result = torch.stack(outputs) if outputs else torch.zeros(0, small_model.d_model)
    log.info("Collected %d small-SAE MLP-output vectors, shape %s", result.shape[0], result.shape)
    return result


# ---------------------------------------------------------------------------
# Step 3: Collect large model MLP outputs at '='
# ---------------------------------------------------------------------------


def collect_large_mlp_outputs(
    large_model: AttributionModel,
    samples: list[dict[str, Any]],
    large_layers: list[int],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Collect Qwen3-4B hook_mlp_out activations at the last input token position.

    Hooks into blocks.{layer}.hook_mlp_out (block-level, fires after AttributionMLP
    including any permanent skip-connection hooks).

    Returns dict {large_layer: tensor[n_samples, d_model_large]}.
    """
    log.info("Collecting large model MLP outputs at layers %s", large_layers)
    acts_by_layer: dict[int, list[torch.Tensor]] = {layer_idx: [] for layer_idx in large_layers}

    large_model.eval()
    with torch.no_grad():
        for sample in tqdm(samples, desc="Collecting large MLP outputs"):
            prompt = sample["prompt"]
            tokens = tokenize_qwen_input(prompt, large_model.tokenizer, device=device).unsqueeze(0)
            eq_pos = tokens.shape[1] - 1  # last token before generation

            cache: dict[int, torch.Tensor] = {}

            def make_hook(layer_idx: int, eq_pos=eq_pos, cache=cache):
                def _hook(
                    act: torch.Tensor, hook, _pos: int = eq_pos, _cache=cache
                ) -> torch.Tensor:
                    _cache[layer_idx] = act[:, _pos, :].clone()
                    return act

                return _hook

            hooks = [
                (f"blocks.{layer_idx}.hook_mlp_out", make_hook(layer_idx))
                for layer_idx in large_layers
            ]
            with large_model.hooks(fwd_hooks=hooks):
                large_model(tokens)

            for layer_idx in large_layers:
                if layer_idx in cache:
                    acts_by_layer[layer_idx].append(cache[layer_idx].cpu())

    result: dict[int, torch.Tensor] = {}
    for layer_idx, lst in acts_by_layer.items():
        if lst:
            result[layer_idx] = torch.cat(lst, dim=0)
            log.info("  Layer %d: %s", layer_idx, result[layer_idx].shape)
    return result


# ---------------------------------------------------------------------------
# CCA helper (SVCCA proxy, as per Chen et al. arXiv 2506.06609)
# ---------------------------------------------------------------------------


def compute_cca_score(
    X: np.ndarray, Y: np.ndarray, n_components: int = 20
) -> float:  # TODO: write in docs
    """Mean top-k canonical correlation between X and Y (SVCCA proxy).

    Projects X and Y onto their top-k PCA subspaces, then computes the Frobenius
    similarity between the two orthonormal bases via SVD of their cross-gram matrix.
    Singular values of this matrix are the principal angles; mean cosine gives a
    score in [0, 1].  Returns 1.0 when X and Y span the same subspace.

    Follows Chen et al. (arXiv 2506.06609) layer-selection criterion.
    """
    from numpy.linalg import svd

    n = X.shape[0]
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)

    k = min(n_components, n - 1, X_c.shape[1], Y_c.shape[1])
    if k < 1:
        return 0.0

    # Orthonormal bases for the top-k PCA subspaces
    Ux, _, _ = svd(X_c, full_matrices=False)
    Uy, _, _ = svd(Y_c, full_matrices=False)
    Qx = Ux[:, :k]  # (n, k) — orthonormal columns
    Qy = Uy[:, :k]  # (n, k) — orthonormal columns

    # Principal angles between the two subspaces
    # Singular values of Qx^T @ Qy are the cosines of principal angles
    sv = svd(Qx.T @ Qy, compute_uv=False)  # (k,), values in [0, 1]
    return float(np.clip(sv, 0.0, 1.0).mean())


# ---------------------------------------------------------------------------
# Step 4: Fit affine maps (MLP-output space)
# ---------------------------------------------------------------------------


def fit_mlp_output_maps(
    small_sae_outputs: torch.Tensor,  # [n, d_small]
    large_mlp_outputs: dict[int, torch.Tensor],  # {large_layer: [n, d_large]}
    target_large_layers: list[int],
) -> dict[int, dict[str, Any]]:
    """Fit affine maps: small_mlp_out (d_small) -> large_mlp_out (d_large).

    Uses multi-output Ridge regression (Fix #2: replaces the O(d_large) per-dim loop).
    Also reports SVCCA score for layer selection (Fix #5, as per Chen et al.).

    Returns dict {large_layer: {'W': (d_large, d_small), 'b': (d_large,), 'r2', 'cca'}}.
    """
    log.info("Fitting MLP-output affine maps for large layers %s", target_large_layers)
    X = small_sae_outputs.float().cpu().numpy()  # [n, d_small] — cast bf16→f32 for sklearn
    maps: dict[int, dict[str, Any]] = {}

    for large_layer in target_large_layers:
        if large_layer not in large_mlp_outputs:
            log.warning("No activations for large layer %d — skipping", large_layer)
            continue

        Y = large_mlp_outputs[large_layer].float().cpu().numpy()  # [n, d_large] — cast bf16→f32

        # FIX #2: single multi-output Ridge fit (was a loop over d_large dims)
        ridge = Ridge(alpha=1e-4, fit_intercept=True)
        ridge.fit(X, Y)
        W = ridge.coef_  # (d_large, d_small)
        b = ridge.intercept_  # (d_large,)

        Y_pred = X @ W.T + b
        r2 = float(r2_score(Y, Y_pred))
        cca = compute_cca_score(X, Y)

        maps[large_layer] = {
            "W": torch.tensor(W, dtype=torch.float32),
            "b": torch.tensor(b, dtype=torch.float32),
            "r2": r2,
            "cca": cca,
        }
        log.info("  Layer %d: R²=%.4f  CCA=%.4f", large_layer, r2, cca)

    return maps


# ---------------------------------------------------------------------------
# Step 5: Inject and verify
# ---------------------------------------------------------------------------


def inject_and_verify(
    small_model: SmallAdditionTransformer,
    small_sae: SingleLayerTranscoder,
    large_model: AttributionModel,
    stitch_maps: dict[int, dict[str, Any]],
    samples: list[dict[str, Any]],
    sae_layer: int,
    cascading_carry_threshold: int,
    device: torch.device,
    plot_dataset: bool = False,
    out_root: Path | str | None = None,
) -> dict[str, Any]:
    """Inject small SAE MLP outputs into Qwen3-4B via hook_mlp_out patching.

    For each sample:
      1. Run small model → resid_mid at '=' → SAE encode+decode → small_mlp_out
      2. Affine map: stitched = small_mlp_out @ W.T + b  (→ d_large)
      3. Patch blocks.{large_layer}.hook_mlp_out with stitched
      4. Measure accuracy (FIX #7: exact token-ID match) and KL (FIX #1, #6)
    """
    log.info("Running SAE-mediated MLP injection verification")

    # Select best large layer by CCA, then R² as fallback (FIX #5)
    def _score(item: tuple[int, dict]) -> float:
        return item[1].get("cca", item[1].get("r2", 0.0))

    best_large_layer, best_map = max(stitch_maps.items(), key=_score)
    score_key = "cca" if "cca" in best_map else "r2"
    W = best_map["W"].to(device)
    b_vec = best_map["b"].to(device)
    log.info("Best large layer: %d (%s=%.4f)", best_large_layer, score_key, best_map[score_key])

    tokenize_small = get_small_model_tokenizer(small_model)

    cascading = identify_cascading_carry_cases(samples, cascading_carry_threshold)
    cascading_total = sum(cascading)

    correct_before = correct_after = 0
    casc_before = casc_after = 0
    total_kl = 0.0
    n_valid = 0

    tf_probs_before = []
    tf_probs_after = []
    tf_a_vals = []
    tf_b_vals = []

    small_model.model.eval()
    small_sae.eval()
    large_model.eval()

    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc="SAE injection")):
            a_val = sample.get("a", 0)
            b_val = sample.get("b", 0)
            answer = sample.get("answer", str(a_val + b_val))
            prompt = sample.get("prompt", TEMPLATES[TemplateID.T0].format(a=a_val, b=b_val))

            # ---- small model → SAE → small_mlp_out ----
            # Use correct QuantaMaths format: zero-padded with answer sign
            n_digits = getattr(small_model, "_n_digits", 5)
            a_str = str(a_val).zfill(n_digits)
            b_str = str(b_val).zfill(n_digits)
            total = a_val + b_val
            ans_digits = n_digits + 1
            ans_str = "+" + str(total).zfill(ans_digits)
            small_text = f"{a_str}+{b_str}={ans_str}"
            eq_small = small_text.find("=")
            if eq_small == -1:
                continue

            ids_small = torch.tensor([tokenize_small(small_text)], device=device, dtype=torch.long)
            cache_sm: dict[str, torch.Tensor] = {}

            def _hook_sm(
                act: torch.Tensor, hook, _pos: int = eq_small, cache_sm=cache_sm
            ) -> torch.Tensor:
                cache_sm["v"] = act[:, _pos, :].clone()
                return act

            with small_model.model.hooks(
                fwd_hooks=[(f"blocks.{sae_layer}.hook_resid_mid", _hook_sm)]
            ):
                small_model.model(ids_small)

            if "v" not in cache_sm:
                continue

            resid_mid = cache_sm["v"].to(dtype=torch.float32)  # (1, d_small)
            feats = small_sae.encode(resid_mid)  # (1, d_tc)
            small_mlp_out = small_sae.decode(feats)  # (1, d_small)

            # ---- affine map → large MLP output space ----
            stitched = (small_mlp_out @ W.T + b_vec).to(dtype=large_model.cfg.dtype)  # (1, d_large)

            # ---- large model: before / after (Teacher Forced) ----
            large_tokens = tokenize_qwen_input(prompt, large_model.tokenizer, device).unsqueeze(0)
            eq_large = large_tokens.shape[1] - 1

            if not answer:
                continue
            ans_encoded = large_model.tokenizer(
                answer, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            full_tokens = torch.cat([large_tokens, ans_encoded], dim=1)

            logits_before = large_model(full_tokens)

            def patch_hook(
                act: torch.Tensor,
                hook,
                _val: torch.Tensor = stitched,
                _pos: int = eq_large,
            ) -> torch.Tensor:
                # During generate() TransformerLens runs one token at a time
                # (KV-cached), so act has shape (1, 1, d_model) for every
                # decoding step after the prefill.  Only patch during the
                # prefill pass, when the full prompt is still in context.
                if act.shape[1] <= _pos:
                    return act
                act = act.clone()
                act[:, _pos, :] = _val
                return act

            with large_model.hooks(
                fwd_hooks=[(f"blocks.{best_large_layer}.hook_mlp_out", patch_hook)]
            ):
                logits_after = large_model(full_tokens)

            # Accuracy: token-ID match at the first answer position (teacher forced).
            # answer_toks contains all token IDs that make up the answer string; we
            # check whether the model's top-1 prediction at eq_large is among them.
            # This is faster than generate() and avoids decoding artifacts.
            answer_toks = large_model.tokenizer(answer, add_special_tokens=False).input_ids
            pred_before_tok = int(logits_before[0, eq_large, :].argmax())
            pred_after_tok = int(logits_after[0, eq_large, :].argmax())
            is_correct_before = pred_before_tok in answer_toks
            is_correct_after = pred_after_tok in answer_toks

            if is_correct_before:
                correct_before += 1
                if cascading[i]:
                    casc_before += 1
            if is_correct_after:
                correct_after += 1
                if cascading[i]:
                    casc_after += 1

            # ---- Teacher Forcing Analysis (Average P(correct) over the whole answer) ----
            prob_b_seq = []
            prob_a_seq = []
            for k in range(ans_encoded.shape[1]):
                t_id = ans_encoded[0, k].item()
                pos = eq_large + k
                prob_b_seq.append(F.softmax(logits_before[0, pos], dim=-1)[t_id].item())
                prob_a_seq.append(F.softmax(logits_after[0, pos], dim=-1)[t_id].item())

            tf_probs_before.append(float(np.mean(prob_b_seq)))
            tf_probs_after.append(float(np.mean(prob_a_seq)))
            tf_a_vals.append(a_val)
            tf_b_vals.append(b_val)

            # ---- KL divergence ----
            p = F.softmax(logits_before[0, eq_large], dim=-1).clamp(min=1e-10)
            q = F.softmax(logits_after[0, eq_large], dim=-1).clamp(min=1e-10)
            kl = F.kl_div(q.log(), p, reduction="sum", log_target=False).item()
            total_kl += kl
            n_valid += 1

    # Plot if requested
    if plot_dataset and out_root is not None:
        plot_dir = Path(out_root) / "plots" / "stitching"
        plot_stitching_results(tf_a_vals, tf_b_vals, tf_probs_before, tf_probs_after, plot_dir)

    denom = max(n_valid, 1)
    metrics: dict[str, Any] = {
        "best_large_layer": best_large_layer,
        "best_score": best_map.get("cca", best_map.get("r2", 0.0)),
        "score_type": score_key,
        "before_acc": 100.0 * correct_before / denom,
        "after_acc": 100.0 * correct_after / denom,
        "cascading_before_acc": 100.0 * casc_before / max(cascading_total, 1),
        "cascading_after_acc": 100.0 * casc_after / max(cascading_total, 1),
        "kl_divergence": total_kl / denom,
        "n_samples": n_valid,
    }

    log.info("Before injection: %.2f%% accuracy", metrics["before_acc"])
    log.info("After  injection: %.2f%% accuracy", metrics["after_acc"])
    log.info("Cascading carry before: %.2f%%", metrics["cascading_before_acc"])
    log.info("Cascading carry after:  %.2f%%", metrics["cascading_after_acc"])
    log.info("Avg KL(before||after): %.4f", metrics["kl_divergence"])
    return metrics


# ---------------------------------------------------------------------------
# Step 6: Attribution Graph Comparison (placeholder)
# ---------------------------------------------------------------------------


def compare_attribution_graphs(  # TODO: implement
    large_model: AttributionModel,
    samples: list[dict[str, Any]],
    stitch_maps: dict[int, dict[str, Any]],
    out_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run attribution pipeline before and after stitching (placeholder)."""
    log.warning("Attribution graph comparison is a placeholder — integrate with run_attribution.py")
    comparison: dict[str, Any] = {
        "top_20_features_before": [],
        "top_20_features_after": [],
        "overlap": 0,
        "carry_feature_improvement": 0.0,
    }
    torch.save({"placeholder": True}, out_root / "graph_before.pt")
    torch.save({"placeholder": True}, out_root / "graph_after.pt")
    return comparison


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stitching/run.py",
        description="Model stitching experiment (Option B: SAE-mediated MLP injection)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    phases = p.add_argument_group("Phases")
    phases.add_argument("--train-small", action="store_true", help="Train small addition model")
    phases.add_argument(
        "--train-sae", action="store_true", help="Train small SAE on small model MLP"
    )
    phases.add_argument(
        "--collect-small", action="store_true", help="Collect small SAE MLP outputs"
    )
    phases.add_argument(
        "--collect-large", action="store_true", help="Collect large model MLP outputs"
    )
    phases.add_argument("--fit-stitch", action="store_true", help="Fit affine maps")
    phases.add_argument("--verify", action="store_true", help="Verify with activation patching")
    phases.add_argument("--compare-graphs", action="store_true", help="Compare attribution graphs")
    phases.add_argument(
        "--plot-dataset", action="store_true", help="Plot teacher-forced P(correct) vs examples"
    )
    phases.add_argument("--all", action="store_true", help="Run all phases")

    model_args = p.add_argument_group("Model")
    model_args.add_argument("--model", default="Qwen/Qwen3-4B")
    model_args.add_argument("--transcoder_set", default="mwhanna/qwen3-4b-transcoders")
    model_args.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"]
    )
    model_args.add_argument("--device", default=None)

    small_args = p.add_argument_group("Small Model")
    small_args.add_argument(
        "--hub-model",
        default="PhilipQuirke/QuantaMaths_add_d5_l1_h3_t15K_s372001",
        help=(
            "HuggingFace model ID for a pretrained QuantaMaths addition model. "
            "Set to empty string '' to train from scratch instead. "
            "Default: PhilipQuirke/QuantaMaths_add_d5_l1_h3_t15K_s372001 "
            "(d_model=510, 5-digit, 1L3H, already with ~99%% val accuracy)."
        ),
    )
    small_args.add_argument("--small_model_layers", type=int, default=2)
    small_args.add_argument("--small_model_heads", type=int, default=3)
    small_args.add_argument("--small_model_d_model", type=int, default=256)
    small_args.add_argument(
        "--small_model_epochs",
        type=int,
        default=30,
        help="Training epochs. Paper uses ~1.5M datums: 50K samples × 30 epochs = 1.5M",
    )
    small_args.add_argument("--small_model_lr", type=float, default=1e-3)
    small_args.add_argument(
        "--small_model_num_digits",
        type=int,
        default=5,
        help="Only used when training from scratch (--hub-model='')",
    )
    small_args.add_argument(
        "--small_model_use_rope",
        action="store_true",
        help=(
            "Use Rotary Position Embeddings (RoPE) instead of absolute positional embeddings "
            "when training small model from scratch. RoPE matches Qwen's positional encoding "
            "and may improve representation alignment for stitching. Only used when --hub-model=''."
        ),
    )

    sae_args = p.add_argument_group("Small SAE")
    sae_args.add_argument(
        "--small_sae_d_transcoder",
        type=int,
        default=4096,
        help="SAE feature count (default 4096 ≈ 16× d_model=256)",
    )
    sae_args.add_argument("--small_sae_epochs", type=int, default=500)
    sae_args.add_argument("--small_sae_lr", type=float, default=1e-3)
    sae_args.add_argument(
        "--small_sae_layer",
        type=int,
        default=None,
        help="Small model layer to train SAE on (default: last layer)",
    )

    stitch_args = p.add_argument_group("Stitching")
    stitch_args.add_argument(
        "--stitch_layer_pairs",
        type=str,
        default="[14, 16, 18]",
        help="Large model layers to probe (list of ints)",
    )
    stitch_args.add_argument("--held_out_fraction", type=float, default=0.2)
    stitch_args.add_argument("--cascading_carry_threshold", type=int, default=2)
    stitch_args.add_argument(
        "--num_verify_samples",
        type=int,
        default=1000,
        help="Number of samples to run for Phase 5 injection verification (default 1000)",
    )

    data_args = p.add_argument_group("Dataset")
    data_args.add_argument("--dataset_path", default="data/addition_dataset_stitching.jsonl")

    out_args = p.add_argument_group("Output")
    out_args.add_argument("--out_root", default="runs/stitching")

    misc = p.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--dry-run", action="store_true", help="Use 10 samples only")

    add_config_args(p)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()

    pre, _ = parser.parse_known_args()
    pos_config = None
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml") and not sys.argv[1].startswith("-"):
        pos_config = sys.argv[1]
        sys.argv.pop(1)

    config_path = pre.config or pos_config
    config = load_config(config_path)
    set_parser_defaults_from_config(parser, config, section="stitching_experiment")

    args = parser.parse_args()
    if args.config is None and config_path:
        args.config = config_path

    # Parse large layer list
    if isinstance(args.stitch_layer_pairs, str):
        args.stitch_layer_pairs = json.loads(args.stitch_layer_pairs)

    print_config(args, title="Stitching Experiment Configuration")

    run_all = args.all
    do_train = run_all or args.train_small
    do_train_sae = run_all or args.train_sae
    do_collect_sm = run_all or args.collect_small
    do_collect_lg = run_all or args.collect_large
    do_fit = run_all or args.fit_stitch
    do_verify = run_all or args.verify
    do_compare = run_all or args.compare_graphs

    if not any(
        [do_train, do_train_sae, do_collect_sm, do_collect_lg, do_fit, do_verify, do_compare]
    ):
        parser.print_help()
        sys.exit(0)

    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else get_default_device()
    dtype = parse_dtype(args.dtype)
    log.info("Device: %s  dtype: %s", device, dtype)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
    }

    # ---- Load dataset (dict samples for large model) ----
    max_samples = 10 if args.dry_run else None
    dict_samples = load_addition_dataset(
        args.dataset_path, max_samples, num_digits=args.small_model_num_digits
    )
    log.info("Loaded %d dict samples", len(dict_samples))

    n_train = int(len(dict_samples) * (1 - args.held_out_fraction))
    train_dict = dict_samples[:n_train]
    test_dict = dict_samples[n_train:]
    log.info("Split: train=%d test=%d", len(train_dict), len(test_dict))

    sae_layer = args.small_sae_layer  # None → defaults to n_layers-1 in functions

    # ---- Paths ----
    small_model_path = out_root / "small_model.pt"
    small_sae_path = out_root / "small_sae.safetensors"
    small_sae_out_path = out_root / "small_sae_outputs.pt"
    large_mlp_out_path = out_root / "large_mlp_outputs.pt"
    stitch_maps_path = out_root / "stitch_maps.pt"
    small_samples_path = out_root / "training_samples.pt"

    # ---- Load large model once (reuse across steps) ----
    # Only load if we need it (steps 3, 5, or 6)
    large_model: AttributionModel | None = None
    if do_collect_lg or do_verify or do_compare:
        log.info("=" * 60)
        log.info("Loading large model (will be reused across steps)")
        log.info("=" * 60)
        large_model = _load_large_model(args, dtype, device=device)
        log.info("Large model loaded successfully - will reuse for all steps")

    # ========================================================================
    # Step 1: Train small model
    # ========================================================================
    small_model: SmallAdditionTransformer | None = None
    small_extraction_samples: list[str] = []

    hub_model_id = getattr(args, "hub_model", "").strip()

    if do_train:
        log.info("=" * 60)
        log.info("Step 1: Small addition model")
        log.info("=" * 60)

        if hub_model_id:
            # ---- Load pretrained QuantaMaths model from HuggingFace ----
            log.info(
                "Using pretrained model: %s (skipping training)",
                hub_model_id,
            )
            small_model, n_digits_hub = load_quanta_maths_model(hub_model_id, device=device)
            args.small_model_d_model = small_model.d_model  # Use the actual pretrained dimension
            log.info(
                "Overwriting args.small_model_d_model with actual pretrained dimension: %d",
                small_model.d_model,
            )
            args.small_model_num_digits = n_digits_hub
            log.info(
                "Overwriting args.small_model_num_digits with actual pretrained digits: %d",
                n_digits_hub,
            )
            import random as _rnd

            _rnd.seed(args.seed)
            _max = 10**n_digits_hub - 1
            n_extraction = 10 if args.dry_run else 5_000
            small_extraction_samples = [
                _qm_make_sample(_rnd.randint(0, _max), _rnd.randint(0, _max), n_digits_hub)["full"]
                for _ in range(n_extraction)
            ]
            log.info(
                "Generated %d QuantaMaths extraction samples (n_digits=%d)",
                len(small_extraction_samples),
                n_digits_hub,
            )
            torch.save(small_model.state_dict(), small_model_path)
            torch.save({"train": small_extraction_samples}, small_samples_path)
            metrics["small_model_hub"] = hub_model_id
        else:
            # ---- Train from scratch ----
            small_model, tr_str, val_str, ood_str = train_small_model(
                n_layers=args.small_model_layers,
                n_heads=args.small_model_heads,
                d_model=args.small_model_d_model,
                epochs=args.small_model_epochs,
                lr=args.small_model_lr,
                device=device,
                dtype=dtype,
                num_digits=args.small_model_num_digits,
                dry_run=args.dry_run,
                use_rope=args.small_model_use_rope,
            )
            torch.save(small_model.state_dict(), small_model_path)
            torch.save({"train": tr_str, "val": val_str, "ood": ood_str}, small_samples_path)
            small_extraction_samples = tr_str
            metrics["small_model_trained"] = True
            log.info("Saved small model to %s", small_model_path)

    # ========================================================================
    # Step 1.5: Train small SAE
    # ========================================================================
    small_sae: SingleLayerTranscoder | None = None

    if do_train_sae:
        log.info("=" * 60)
        log.info("Step 1.5: Training small SAE")
        log.info("=" * 60)
        if small_model is None:
            small_model = _load_small_model(args, small_model_path, device)
        if not small_extraction_samples:
            small_extraction_samples = _load_small_samples(small_samples_path)

        small_sae = train_small_sae(
            small_model=small_model,
            small_extraction_samples=small_extraction_samples,
            device=device,
            d_transcoder=args.small_sae_d_transcoder,
            epochs=args.small_sae_epochs,
            lr=args.small_sae_lr,
            sae_layer=sae_layer,
            dry_run=args.dry_run,
        )
        small_sae.to_safetensors(str(small_sae_path))
        log.info("Saved small SAE to %s", small_sae_path)
        metrics["small_sae_trained"] = True

    # ========================================================================
    # Step 2: Collect small SAE outputs
    # ========================================================================
    small_sae_outputs: torch.Tensor | None = None

    if do_collect_sm:
        log.info("=" * 60)
        log.info("Step 2: Collecting small SAE MLP outputs")
        log.info("=" * 60)
        if small_model is None:
            small_model = _load_small_model(args, small_model_path, device)
        if small_sae is None:
            small_sae = _load_small_sae(args, small_sae_path, device)
        if not small_extraction_samples:
            small_extraction_samples = _load_small_samples(small_samples_path)

        # Limit to 5000 samples for speed
        small_extraction_samples_limited = small_extraction_samples[:5000]
        log.info(
            "Using %d samples for collection (limited from %d)",
            len(small_extraction_samples_limited),
            len(small_extraction_samples),
        )

        _sae_layer = sae_layer if sae_layer is not None else small_model.n_layers - 1
        small_sae_outputs = collect_small_sae_outputs(
            small_model, small_sae, small_extraction_samples_limited, device, _sae_layer
        )
        torch.save(small_sae_outputs, small_sae_out_path)
        log.info("Saved small SAE outputs: %s", small_sae_outputs.shape)
        metrics["small_sae_outputs_collected"] = True

    # ========================================================================
    # Step 3: Collect large model MLP outputs
    # ========================================================================
    large_mlp_outputs: dict[int, torch.Tensor] | None = None

    if do_collect_lg:
        log.info("=" * 60)
        log.info("Step 3: Collecting large model MLP outputs")
        log.info("=" * 60)
        # Large model already loaded at the start - reuse it
        if large_model is None:
            raise RuntimeError("Large model should have been loaded at start of main()")
        large_layers: list[int] = list(args.stitch_layer_pairs)

        # Build matched samples: same (a,b) pairs as the small model saw
        hub_model_id = getattr(args, "hub_model", "").strip()
        if hub_model_id and small_extraction_samples:
            # Limit to 5000 samples for speed (must match small SAE collection)
            small_extraction_samples_limited = small_extraction_samples[:5000]
            # Parse QuantaMaths strings → Qwen3-4B prompt dicts (same problems, different format)
            _n_dig = (
                getattr(small_model, "_n_digits", args.small_model_num_digits) if small_model else 5
            )
            large_collect_samples = _qm_to_large_dicts(
                small_extraction_samples_limited, large_model, _n_dig
            )
            log.info(
                "Using %d matched QuantaMaths samples for large model collection",
                len(large_collect_samples),
            )
        else:
            large_collect_samples = train_dict[
                : min(len(small_extraction_samples) if small_extraction_samples else 5_000, 5_000)
            ]

        large_mlp_outputs = collect_large_mlp_outputs(
            large_model, large_collect_samples, large_layers, device
        )
        torch.save(large_mlp_outputs, large_mlp_out_path)
        log.info("Saved large MLP outputs to %s", large_mlp_out_path)
        metrics["large_mlp_outputs_collected"] = True

    # ========================================================================
    # Step 4: Fit affine maps
    # ========================================================================
    stitch_maps: dict[int, dict[str, Any]] | None = None

    if do_fit:
        log.info("=" * 60)
        log.info("Step 4: Fitting MLP-output affine maps")
        log.info("=" * 60)
        if small_sae_outputs is None:
            small_sae_outputs = torch.load(small_sae_out_path)
        if large_mlp_outputs is None:
            large_mlp_outputs = torch.load(large_mlp_out_path)

        stitch_maps = fit_mlp_output_maps(
            small_sae_outputs,
            large_mlp_outputs,
            list(args.stitch_layer_pairs),
        )
        torch.save(stitch_maps, stitch_maps_path)
        metrics["r2_scores"] = {str(k): v["r2"] for k, v in stitch_maps.items()}
        metrics["cca_scores"] = {str(k): v["cca"] for k, v in stitch_maps.items()}

    # ========================================================================
    # Step 5: Inject and verify
    # ========================================================================
    if do_verify:
        log.info("=" * 60)
        log.info("Step 5: SAE injection verification")
        log.info("=" * 60)
        if small_model is None:
            small_model = _load_small_model(args, small_model_path, device)
        if small_sae is None:
            small_sae = _load_small_sae(args, small_sae_path, device)
        if stitch_maps is None:
            stitch_maps = torch.load(stitch_maps_path)
        # Large model already loaded at the start - reuse it
        if large_model is None:
            raise RuntimeError("Large model should have been loaded at start of main()")

        # Slice test_dict to avoid extremely long verification runs
        hub_model_id = getattr(args, "hub_model", "").strip()
        if hub_model_id:
            # Generate a fresh set of matched problems for verification (QuantaMaths format)
            _n_dig = (
                getattr(small_model, "_n_digits", args.small_model_num_digits) if small_model else 5
            )
            _verify_count = min(args.num_verify_samples, 2000)  # Cap at 2k for sanity
            log.info("Generating %d matched QuantaMaths samples for verification", _verify_count)
            _raw_verify = [
                _qm_make_sample(
                    random.randint(0, 10**_n_dig - 1), random.randint(0, 10**_n_dig - 1), _n_dig
                )["full"]
                for _ in range(_verify_count)
            ]
            _verify_samples = _qm_to_large_dicts(_raw_verify, large_model, _n_dig)
        else:
            _verify_samples = test_dict[: args.num_verify_samples]

        log.info(
            "Running injection verification on %d samples (out of %d available)",
            len(_verify_samples),
            len(_verify_samples) if hub_model_id else len(test_dict),
        )

        _sae_layer = sae_layer if sae_layer is not None else small_model.n_layers - 1
        patch_metrics = inject_and_verify(
            small_model=small_model,
            small_sae=small_sae,
            large_model=large_model,
            stitch_maps=stitch_maps,
            samples=_verify_samples,
            sae_layer=_sae_layer,
            cascading_carry_threshold=args.cascading_carry_threshold,
            device=device,
            plot_dataset=True,
            out_root=out_root,
        )
        metrics["patching"] = patch_metrics

    # ========================================================================
    # Step 6: Attribution graph comparison
    # ========================================================================
    if do_compare:
        log.info("=" * 60)
        log.info("Step 6: Attribution graph comparison")
        log.info("=" * 60)
        # Large model already loaded at the start - reuse it
        if large_model is None:
            raise RuntimeError("Large model should have been loaded at start of main()")
        if stitch_maps is None:
            stitch_maps = torch.load(stitch_maps_path)
        graph_cmp = compare_attribution_graphs(
            large_model, test_dict, stitch_maps, out_root, device
        )
        metrics["graph_comparison"] = graph_cmp

    # ---- Save metrics ----
    metrics_path = out_root / "info.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Saved metrics to %s", metrics_path)

    print("\n" + "=" * 60)
    print(f"Stitching experiment complete. Outputs in: {out_root}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Load helpers (avoid code duplication in main)
# ---------------------------------------------------------------------------


def _load_small_model(
    args: argparse.Namespace, path: Path, device: torch.device
) -> SmallAdditionTransformer:
    log.info("Loading small model from %s", path)
    sd = torch.load(path, map_location=device)

    # Check for 'model.' prefix and strip it if it exists (for flat QuantaMaths checkpoints)
    first_key = next(iter(sd.keys()))
    has_prefix = first_key.startswith("model.")

    def _get(key: str):
        full_key = f"model.{key}" if has_prefix else key
        return sd[full_key]

    # Infer architecture from checkpoint
    vocab_size = _get("embed.W_E").shape[0]
    d_model = _get("embed.W_E").shape[1]

    # RoPE models store rotary_sin/cos per block instead of a learned pos_embed
    use_rope = any(k.endswith("attn.rotary_sin") for k in sd)
    if use_rope:
        # rotary_sin shape: (n_ctx, d_head/2) — first dim is max sequence length
        n_ctx = _get("blocks.0.attn.rotary_sin").shape[0]
    else:
        n_ctx = _get("pos_embed.W_pos").shape[0]

    # Count layers robustly
    prefix = "model.blocks." if has_prefix else "blocks."
    n_layers = sum(1 for k in sd if k.startswith(prefix) and k.endswith(".ln1.w"))
    n_heads = _get("blocks.0.attn.W_Q").shape[0]

    m = SmallAdditionTransformer(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        vocab_size=vocab_size,
        max_seq_len=n_ctx,
        device=device,
        use_rope=use_rope,
    )

    # Load correctly
    if has_prefix:
        m.load_state_dict(sd)
    else:
        m.model.load_state_dict(sd)

    m.to(device)
    # Restore QuantaMaths tokenizer attributes if this is a hub model
    hub_model_id = getattr(args, "hub_model", "").strip()
    if hub_model_id and vocab_size == 15:  # QuantaMaths vocab is always 15
        import re as _re

        _m = _re.search(r"_d(\d+)_", hub_model_id)
        m._n_digits = int(_m.group(1)) if _m else (n_ctx - 4) // 3  # type: ignore[attr-defined]
        m._tokenizer = _qm_tokenize  # type: ignore[attr-defined]
        m._make_sample = _qm_make_sample  # type: ignore[attr-defined]
        log.info("Restored QuantaMaths tokenizer (n_digits=%d)", m._n_digits)

    return m


def _load_small_sae(
    args: argparse.Namespace, path: Path, device: torch.device
) -> SingleLayerTranscoder:
    from mechinterp_qwen3.transcoder.single_layer_transcoder import load_relu_transcoder

    log.info("Loading small SAE from %s", path)
    sae_layer = (
        args.small_sae_layer if args.small_sae_layer is not None else args.small_model_layers - 1
    )
    return load_relu_transcoder(str(path), layer=sae_layer, lazy_encoder=False, lazy_decoder=False)


def _load_small_samples(path: Path) -> list[str]:
    log.info("Loading small-model training samples from %s", path)
    return torch.load(path)["train"]


def _load_large_model(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device | None = None
) -> AttributionModel:
    log.info("Loading large model %s with transcoders %s", args.model, args.transcoder_set)
    transcoder, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=True
    )
    kwargs: dict = {"dtype": dtype}
    if device is not None:
        kwargs["device"] = device
    return AttributionModel.from_pretrained_and_transcoders(args.model, transcoder, **kwargs)


if __name__ == "__main__":
    main()
