"""
Visualise whether learned soft-prompt prefix vectors are interpretable.

For each of the three evaluated concepts (residue_class, balanced_parentheses,
causal_direction) the script:

  1.  Loads the learned prefix matrix P ∈ R^{k×d}.
  2.  Computes cosine similarity between every prefix position and every token
      in the embedding matrix (Qwen3-4B uses tied embeddings, so the embedding
      matrix doubles as the unembedding / logit-lens projection).
  3.  Produces two figures:

      prefix_interpretability_nn.pdf
          Three-panel heatmap (one per concept): rows = prefix positions,
          columns = top-5 nearest tokens ranked by cosine similarity.
          Cell colour encodes cosine similarity; cell text shows the decoded
          token string.  Gives a qualitative read on whether the vectors are
          "word-like".

      prefix_interpretability_maxcos.pdf
          Single panel: max cosine similarity per prefix position, grouped by
          concept.  Low max cosine (~0.3–0.5) indicates the vector lives in a
          region of embedding space not well-covered by any real token.

Run on CPU — no GPU or full model load required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
for _p in [str(_REPO), str(_REPO / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiments.plot_style as ps

# ── paths ──────────────────────────────────────────────────────────────────────
_SNAP = (
    "/rds/user/eid23/hpc-work/p28/cache/hf/hub/"
    "models--Qwen--Qwen3-4B/snapshots/"
    "1cfa9a7208912126459214e8b04321603b3df60c"
)
_SHARD = Path(_SNAP) / "model-00001-of-00003.safetensors"
_RUNS = _REPO / "runs" / "soft_prompt"
_OUT = _RUNS / "figures"

CONCEPTS = ["residue_class", "balanced_parentheses", "causal_direction"]
LABELS = {
    "residue_class": "Residue class",
    "balanced_parentheses": "Balanced parentheses",
    "causal_direction": "Causal direction",
}
TOP_K = 5  # nearest tokens to display per prefix position


# ── data loading ───────────────────────────────────────────────────────────────

def load_embedding(shard: Path) -> np.ndarray:
    from safetensors import safe_open
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        W = f.get_tensor("model.embed_tokens.weight").float().numpy()
    print(f"  embed matrix: {W.shape}")
    return W


def load_tokenizer(snap: str):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snap, trust_remote_code=True)


def load_prefix(concept: str) -> np.ndarray:
    path = _RUNS / concept / "prefix_soft_prompt.pt"
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    return ckpt["state_dict"]["prefix"].float().numpy()  # (k, d)


def tok_str(tok_id: int, tokenizer) -> str:
    s = tokenizer.decode([tok_id]).strip()
    # Replace non-ASCII tokens with a short placeholder so they render cleanly
    if not s or not all(ord(c) < 128 for c in s):
        return "[…]"
    return s[:10]


# ── figure 1: nearest-token heatmap ───────────────────────────────────────────

def fig_nn_heatmap(W_E_n: np.ndarray, tokenizer) -> plt.Figure:
    """
    Three-panel heatmap: rows = prefix positions, columns = nearest-token rank.
    Colour = cosine similarity; text = decoded token.
    """
    ps.apply()
    n = len(CONCEPTS)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 5.2))

    for ax, concept in zip(axes, CONCEPTS):
        P = load_prefix(concept)
        k = P.shape[0]
        P_n = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)

        cos = P_n @ W_E_n.T  # (k, vocab)
        top_idx = np.argsort(-cos, axis=1)[:, :TOP_K]
        top_cos = cos[np.arange(k)[:, None], top_idx]

        vmax = top_cos.max()
        im = ax.imshow(top_cos, cmap=ps.CMAP_SEQ, vmin=0.0, vmax=vmax, aspect="auto")

        for i in range(k):
            for j in range(TOP_K):
                v = top_cos[i, j]
                tok = tok_str(int(top_idx[i, j]), tokenizer)
                ax.text(
                    j, i, tok,
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if v > 0.55 * vmax else "#1a1a1a",
                )

        ax.set_xticks(range(TOP_K))
        ax.set_xticklabels([f"#{r + 1}" for r in range(TOP_K)], fontsize=8)
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"$p_{{{i}}}$" for i in range(k)], fontsize=8)
        ax.set_xlabel("Nearest token rank", fontsize=9)
        ax.set_title(LABELS[concept], fontsize=11, fontweight="bold", pad=6)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    axes[0].set_ylabel("Prefix position", fontsize=9)
    fig.suptitle(
        "Nearest embedding-space tokens per learned prefix vector",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


# ── figure 2: max cosine per position ─────────────────────────────────────────

def fig_maxcos(W_E_n: np.ndarray) -> plt.Figure:
    """
    Grouped bar chart: max cosine similarity to any real token, per prefix
    position and concept.  Low values indicate the vector is in a region of
    embedding space not covered by any token.
    """
    ps.apply()
    fig, ax = plt.subplots(figsize=(9, 3.8))

    colors = [ps.NAVY, ps.TEAL, ps.VIOLET]
    width = 0.26
    k = None

    for ci, (concept, color) in enumerate(zip(CONCEPTS, colors)):
        P = load_prefix(concept)
        k = P.shape[0]
        P_n = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        cos = P_n @ W_E_n.T
        max_cos = cos.max(axis=1)
        xs = np.arange(k) + ci * width
        ax.bar(xs, max_cos, width=width, color=color, alpha=0.85,
               label=LABELS[concept])

    ax.set_xticks(np.arange(k) + width)
    ax.set_xticklabels([f"$p_{{{i}}}$" for i in range(k)], fontsize=9)
    ax.set_xlabel("Prefix position", fontsize=10)
    ax.set_ylabel("Max cosine similarity to vocab", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color=ps.GRAY, lw=0.8, linestyle="--", alpha=0.5)
    # annotate the "novel" threshold used in the literature
    ax.axhline(0.5, color=ps.RED, lw=1.0, linestyle=":", alpha=0.7,
               label="novel threshold (0.5)")
    ax.legend(fontsize=9, ncol=2)
    ax.set_title(
        "Distance from token manifold: max cosine similarity per prefix position",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    _OUT.mkdir(parents=True, exist_ok=True)

    print("Loading embedding matrix…")
    W_E = load_embedding(_SHARD)
    W_E_n = W_E / (np.linalg.norm(W_E, axis=1, keepdims=True) + 1e-8)

    print("Loading tokenizer…")
    tok = load_tokenizer(_SNAP)

    print("Figure 1: nearest-token heatmap…")
    f1 = fig_nn_heatmap(W_E_n, tok)
    for ext in ("pdf", "png"):
        p = _OUT / f"prefix_interpretability_nn.{ext}"
        f1.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  → {p}")
    plt.close(f1)

    print("Figure 2: max cosine per position…")
    f2 = fig_maxcos(W_E_n)
    for ext in ("pdf", "png"):
        p = _OUT / f"prefix_interpretability_maxcos.{ext}"
        f2.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  → {p}")
    plt.close(f2)

    print("Done.")


if __name__ == "__main__":
    main()
