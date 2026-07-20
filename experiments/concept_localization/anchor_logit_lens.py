"""Logit lens at a concept's anchor: what token would the model output if generation
stopped at layer L, at that token position, applying the model's own final RMSNorm.

Computes this separately for the mean "positive" and mean "negative" residual stream
(from the cached sweep_residuals.npz + pos_mask), then reports the top-K tokens for
each and the top-K tokens by (pos_logits - neg_logits) difference — the latter isolates
what the concept promotes specifically, factoring out generic continuation tokens that
both conditions promote equally.

Qwen3-4B ties input/output embeddings (tie_word_embeddings=True), so the unembedding
matrix is exactly model.embed_tokens.weight; both it and the final model.norm.weight
are read directly off the safetensors shards — no model load needed.

Usage:
    python -m experiments.concept_localization.anchor_logit_lens \
        --anchor_dir runs/concept_localization/gcd/gcd_T0/anchor_rank3_pos6 --top_k 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

_MODEL_SNAPSHOT = (
    "/rds/user/eid23/hpc-work/p28/cache/hf/hub/models--Qwen--Qwen3-4B"
    "/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)
_RMS_EPS = 1e-6


def _load_unembed_norm_tokenizer():
    index = json.loads((Path(_MODEL_SNAPSHOT) / "model.safetensors.index.json").read_text())

    shard = index["weight_map"]["model.embed_tokens.weight"]
    with safe_open(str(Path(_MODEL_SNAPSHOT) / shard), framework="pt") as f:
        W_U = f.get_tensor("model.embed_tokens.weight").float()  # (vocab, d_model), tied embeddings

    shard = index["weight_map"]["model.norm.weight"]
    with safe_open(str(Path(_MODEL_SNAPSHOT) / shard), framework="pt") as f:
        norm_w = f.get_tensor("model.norm.weight").float()  # (d_model,)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_SNAPSHOT)
    return W_U, norm_w, tokenizer


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x / torch.sqrt(var + _RMS_EPS) * weight


def _topk_tokens(scores: torch.Tensor, tokenizer, k: int) -> list[str]:
    top = scores.topk(k)
    return [f"{tokenizer.decode([i.item()])!r}({v.item():+.2f})" for v, i in zip(top.values, top.indices)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor_dir", type=Path, required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    npz = np.load(args.anchor_dir / "sweep" / "sweep_residuals.npz")
    pos_mask = npz["pos_mask"].astype(bool)
    layers = sorted(int(k[3:]) for k in npz.files if k.startswith("H_L"))

    W_U, norm_w, tokenizer = _load_unembed_norm_tokenizer()

    print(f"\n=== Logit lens (pos vs neg, RMSNorm applied) — {args.anchor_dir} ===")
    for layer in layers:
        H = torch.from_numpy(npz[f"H_L{layer}"].astype(np.float32))
        mean_pos = H[pos_mask].mean(dim=0)
        mean_neg = H[~pos_mask].mean(dim=0)

        logits_pos = W_U @ _rms_norm(mean_pos, norm_w)
        logits_neg = W_U @ _rms_norm(mean_neg, norm_w)
        diff = logits_pos - logits_neg

        print(f"\nL{layer:>2}")
        print(f"  pos  top: " + "  ".join(_topk_tokens(logits_pos, tokenizer, args.top_k)))
        print(f"  neg  top: " + "  ".join(_topk_tokens(logits_neg, tokenizer, args.top_k)))
        print(f"  diff top (pos-neg): " + "  ".join(_topk_tokens(diff, tokenizer, args.top_k)))


if __name__ == "__main__":
    main()
