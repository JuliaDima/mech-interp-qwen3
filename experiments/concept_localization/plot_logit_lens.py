"""Logit lens for concept localization.

Projects the residual stream at each layer through the final layer norm and
unembedding matrix to track how the model's answer prediction evolves across
layers.

Two figures are produced per concept+template:

  logit_lens_diff_<template>.pdf
      Mean logit(pos_answer) - logit(neg_answer) at the last prompt token,
      across layers, for positive vs negative examples.  Identifies the layer
      at which the model commits to its answer.

  logit_lens_topk_<template>.pdf
      Heatmap of top-5 predicted tokens at each layer at the last prompt
      token (one row per rank, one column per layer) for a single representative
      positive example.

Usage
-----
    python experiments/concept_localization/plot_logit_lens.py --concept perfect_square
    python experiments/concept_localization/plot_logit_lens.py --concept perfect_square --template T1
    python experiments/concept_localization/plot_logit_lens.py --all
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import experiments.plot_style as ps

_MODEL   = "Qwen/Qwen3-4B"
BASE     = _REPO_ROOT / "runs" / "concept_localization"
TOP_K_TOK = 5   # tokens to show in the heatmap


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_name: str = _MODEL, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device} …")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, device_map=device,
    )
    model.eval()
    return model, tok, device


# ── Logit-lens forward pass ────────────────────────────────────────────────────

@torch.no_grad()
def run_logit_lens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: list[str],
    device: str,
    anchor_pos: int | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Return logits[n_examples, n_layers+1, vocab] at a fixed token position.

    anchor_pos: token index to probe (0-based, before padding).  None means
                the last non-padding token of each example (default).
    """
    results = []

    for start in range(0, len(prompts), batch_size):
        batch  = prompts[start : start + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True,
                     add_special_tokens=False).to(device)
        outputs = model(**inputs, output_hidden_states=True)

        hs = outputs.hidden_states  # (n_layers+1,) each [batch, seq, d]

        # Resolve the position for each example in the batch.
        # With left-padding, token index `anchor_pos` in the original (unpadded)
        # prompt maps to (pad_len + anchor_pos) in the padded tensor.
        seq_lens = inputs.attention_mask.sum(dim=1)       # [batch]
        pad_lens  = inputs.input_ids.shape[1] - seq_lens  # [batch]
        if anchor_pos is None:
            positions = seq_lens - 1 + pad_lens           # last real token
        else:
            positions = pad_lens + anchor_pos             # fixed index in unpadded seq

        batch_logits = []
        for h in hs:
            normed = model.model.norm(h)           # [batch, seq, d]
            logits = model.lm_head(normed)         # [batch, seq, vocab]
            per_example = torch.stack(
                [logits[b, positions[b]] for b in range(len(batch))]
            )  # [batch, vocab]
            batch_logits.append(per_example.cpu().float().numpy())

        batch_arr = np.stack(batch_logits, axis=1)  # [batch, n_layers+1, vocab]
        results.append(batch_arr)

    return np.concatenate(results, axis=0)   # [n_examples, n_layers+1, vocab]


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_pairs(concept: str, template: str, n_per_template: int = 40):
    """Import the concept dataset module and return pairs for the given template."""
    mod = importlib.import_module(f"data.concept_datasets.{concept}_dataset")
    generate_fn = None
    for name in dir(mod):
        if name.startswith("generate_") and name.endswith("_pairs"):
            generate_fn = getattr(mod, name)
            break
    if generate_fn is None:
        raise ValueError(f"No generate_*_pairs function found in {concept}_dataset")
    all_pairs = generate_fn(n_per_template=n_per_template, templates=[template])
    return [p for p in all_pairs if p.template == template]


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_logit_diff(
    concept: str,
    template: str,
    layers: np.ndarray,
    pos_diffs: np.ndarray,
    neg_diffs: np.ndarray,
    pos_answer: str,
    neg_answer: str,
    out_path: Path,
    probe_label: str = "last token",
) -> None:
    """Line plot: mean logit diff (pos_answer - neg_answer) across layers."""
    ps.apply()
    fig, ax = plt.subplots(figsize=(8, 4))

    for diffs, color, label in [
        (pos_diffs, ps.VIOLET, f"positive examples  (answer = {pos_answer!r})"),
        (neg_diffs, ps.TEAL,   f"negative examples  (answer = {neg_answer!r})"),
    ]:
        mean = diffs.mean(axis=0)
        std  = diffs.std(axis=0)
        ax.plot(layers, mean, color=color, lw=2.0, label=label)
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.15)

    ax.axhline(0, color=ps.GRAY, lw=0.8, ls="--")
    ax.set_xlabel("layer", fontsize=9)
    ax.set_ylabel(f"logit({pos_answer}) − logit({neg_answer})", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="x", color=ps.GRAY, alpha=0.18, lw=0.5)

    # Mark the layer where the gap first opens consistently
    gap = pos_diffs.mean(axis=0) - neg_diffs.mean(axis=0)
    threshold = 0.25 * gap.max()
    crossing = next((i for i, g in enumerate(gap) if g > threshold), None)
    if crossing is not None:
        ax.axvline(layers[crossing], color=ps.RED, lw=0.9, ls=":", alpha=0.8,
                   label=f"gap > 25 % max at layer {layers[crossing]}")

    fig.suptitle(
        f"{concept}  |  template {template}  |  logit lens at {probe_label}",
        fontsize=10, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_topk_heatmap(
    concept: str,
    template: str,
    layers: np.ndarray,
    logits_one: np.ndarray,   # [n_layers+1, vocab] for a single example
    tok: AutoTokenizer,
    pos_tok_id: int,
    neg_tok_id: int,
    out_path: Path,
    top_k: int = TOP_K_TOK,
    probe_label: str = "last token",
) -> None:
    """Heatmap: top-k predicted tokens at each layer for one example."""
    n_steps = logits_one.shape[0]
    top_ids  = np.argsort(logits_one, axis=1)[:, -top_k:][:, ::-1]  # [steps, k]
    tokens   = [[tok.decode([tid]) for tid in row] for row in top_ids]

    # Build a colour matrix: highlight pos/neg answer tokens distinctly
    def tok_color(t_id):
        if t_id == pos_tok_id:
            return ps.VIOLET
        if t_id == neg_tok_id:
            return ps.TEAL
        return ps.GRAY

    ps.apply()
    fig, ax = plt.subplots(figsize=(max(8, n_steps * 0.45), top_k * 0.9 + 1.2))

    for rank in range(top_k):
        for step in range(n_steps):
            t_id  = top_ids[step, rank]
            raw   = tokens[step][rank]
            # escape non-ASCII so matplotlib doesn't warn about missing glyphs
            label = raw.encode("ascii", errors="replace").decode("ascii").strip() or f"<{t_id}>"
            color = tok_color(t_id)
            ax.text(step, top_k - 1 - rank, label, ha="center", va="center",
                    fontsize=7, color=color,
                    fontweight="bold" if color != ps.GRAY else "normal")

    ax.set_xlim(-0.5, n_steps - 0.5)
    ax.set_ylim(-0.5, top_k - 0.5)
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"rank {top_k - i}" for i in range(top_k)], fontsize=7)
    ax.set_xlabel("layer", fontsize=9)
    ax.grid(color=ps.GRAY, alpha=0.15, lw=0.4)

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=ps.VIOLET, label=f"pos answer  ({tok.decode([pos_tok_id])!r})"),
        Patch(color=ps.TEAL,   label=f"neg answer  ({tok.decode([neg_tok_id])!r})"),
        Patch(color=ps.GRAY,   label="other token"),
    ], fontsize=7, loc="upper right")

    fig.suptitle(
        f"{concept}  |  template {template}  |  top-{top_k} tokens per layer at {probe_label}  (one positive example)",
        fontsize=9, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── Main driver ────────────────────────────────────────────────────────────────

_model_cache: tuple | None = None

def get_model(model_name: str = _MODEL):
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model(model_name)
    return _model_cache


def run_concept(
    concept: str,
    template: str = "T0",
    n_examples: int = 40,
    model_name: str = _MODEL,
    anchor_pos: int | None = None,
) -> None:
    model, tok, device = get_model(model_name)
    pairs = load_pairs(concept, template, n_per_template=n_examples)
    if not pairs:
        print(f"  [{concept}/{template}] no pairs found — skipping")
        return

    print(f"  [{concept}/{template}] {len(pairs)} pairs  anchor_pos={anchor_pos!r}")

    pos_prompts = [p.prompt_pos for p in pairs]
    neg_prompts = [p.prompt_neg for p in pairs]
    pos_answer  = pairs[0].predict_pos
    neg_answer  = pairs[0].predict_neg

    pos_tok_id = tok.encode(pos_answer, add_special_tokens=False)[0]
    neg_tok_id = tok.encode(neg_answer, add_special_tokens=False)[0]

    pos_logits = run_logit_lens(model, tok, pos_prompts, device, anchor_pos)  # [N, L+1, V]
    neg_logits = run_logit_lens(model, tok, neg_prompts, device, anchor_pos)

    n_steps = pos_logits.shape[1]
    layers  = np.arange(n_steps)   # 0 = embedding, 1..n = after each layer

    pos_diffs = pos_logits[:, :, pos_tok_id] - pos_logits[:, :, neg_tok_id]
    neg_diffs = neg_logits[:, :, pos_tok_id] - neg_logits[:, :, neg_tok_id]

    pos_label = "last token" if anchor_pos is None else f"pos {anchor_pos}"
    out_dir = BASE / concept
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{template}" if anchor_pos is None else f"{template}_pos{anchor_pos}"

    plot_logit_diff(
        concept, template, layers, pos_diffs, neg_diffs,
        pos_answer, neg_answer,
        out_dir / f"logit_lens_diff_{suffix}.pdf",
        probe_label=pos_label,
    )
    plot_topk_heatmap(
        concept, template, layers, pos_logits[0],
        tok, pos_tok_id, neg_tok_id,
        out_dir / f"logit_lens_topk_{suffix}.pdf",
        probe_label=pos_label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--concept", help="Concept name (must have a *_dataset.py file)")
    group.add_argument("--all", action="store_true",
                       help="Run for every concept with emergence.npy")
    parser.add_argument("--template",   default="T0")
    parser.add_argument("--n_examples", type=int, default=40)
    parser.add_argument("--model",      default=_MODEL)
    parser.add_argument("--anchor_pos", type=int, default=None,
                        help="Token position to probe (0-based in unpadded prompt). "
                             "Default: last token.")
    args = parser.parse_args()

    if args.all:
        concepts = sorted(p.parent.name for p in BASE.glob("*/emergence.npy"))
        print(f"Found {len(concepts)} concepts")
    else:
        concepts = [args.concept]

    for concept in concepts:
        try:
            run_concept(concept, args.template, args.n_examples, args.model,
                        anchor_pos=args.anchor_pos)
        except Exception as e:
            print(f"  [{concept}] ERROR: {e}")


if __name__ == "__main__":
    main()
