"""Plot lookup validation results: candidate scores and inhibition curve.

Usage:
  python experiments/addition/plot_lookup_validation.py \
      --validation_dir runs/addition/lookup_validation \
      --operand_plots_dir runs/addition/operand_plots/operand_plots \
      --out_dir runs/addition/lookup_validation/plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from experiments.plot_style import GRAY, MAUVE, NAVY, RED, TEAL, VIOLET, apply  # noqa: E402

UNIFORM_BASELINE = 0.02  # 200/10000 cells = 2%


def plot_candidate_scores(candidates: list[dict], out_path: Path) -> None:
    """Bar chart of lookup concentration scores for top-k candidates."""
    apply()
    fig, ax = plt.subplots(figsize=(7, 3.5))

    labels = [f"L{c['layer']}\nF{c['feat_idx']}" for c in candidates]
    scores = [c["score"] for c in candidates]

    bars = ax.bar(range(len(candidates)), scores, color=NAVY, alpha=0.82, width=0.65)
    # Highlight top candidate
    bars[0].set_color(VIOLET)
    bars[0].set_alpha(1.0)

    ax.axhline(UNIFORM_BASELINE, color=RED, linewidth=1.4, linestyle="--", label="Uniform baseline (2%)")
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Lookup concentration score $s_{\\ell,k}$")
    ax.set_xlabel("Feature (layer / index)")
    ax.set_title("Top-10 candidates ranked by (6,9) lookup specificity")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(scores) * 1.25)

    # Annotate the top bar
    ax.text(
        0, scores[0] + 0.0005,
        f"{scores[0]:.4f}",
        ha="center", va="bottom", fontsize=8, color=VIOLET,
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_inhibition_curve(inhibition: dict, out_path: Path) -> None:
    """Alpha sweep: Δlogit and Δprob vs alpha."""
    apply()
    results = inhibition["results"]
    alphas = [r["alpha"] for r in results]
    delta_logits = [r["delta_logit"] for r in results]
    delta_probs = [r["delta_prob"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    ax1.plot(alphas, delta_logits, "o-", color=NAVY, markersize=6, label="$\\Delta$logit")
    ax1.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax1.axvline(0, color=GRAY, linewidth=0.8, linestyle=":")
    ax1.set_xlabel("Inhibition scale $\\alpha$")
    ax1.set_ylabel("$\\Delta$ logit (target token)")
    ax1.set_title(f"Inhibition: L{inhibition['feature_layer']} F{inhibition['feature_idx']}")
    ax1.invert_xaxis()

    ax2.plot(alphas, delta_probs, "o-", color=TEAL, markersize=6, label="$\\Delta$prob")
    ax2.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax2.axvline(0, color=GRAY, linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Inhibition scale $\\alpha$")
    ax2.set_ylabel("$\\Delta$ probability (target token)")
    ax2.set_title("Effect on target-token probability")
    ax2.invert_xaxis()

    for ax in (ax1, ax2):
        for alpha, dv in zip(alphas, (delta_logits if ax is ax1 else delta_probs)):
            ax.annotate(
                f"{dv:+.3f}",
                (alpha, dv),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color=GRAY,
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_top_operand_matrices(
    candidates: list[dict],
    operand_plots_dir: Path,
    out_path: Path,
    n: int = 6,
) -> None:
    """Grid of the top-n candidate operand matrices side by side."""
    apply()
    candidates = candidates[:n]
    ncols = 3
    nrows = (len(candidates) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.array(axes).flatten()

    for i, cand in enumerate(candidates):
        npy = operand_plots_dir / f"L{cand['layer']:02d}_F{cand['feat_idx']:06d}.npy"
        if not npy.exists():
            axes[i].set_visible(False)
            continue
        mat = np.load(str(npy))
        im = axes[i].imshow(mat.T, origin="lower", aspect="auto", cmap="viridis")
        axes[i].set_title(
            f"L{cand['layer']} F{cand['feat_idx']}\nscore={cand['score']:.4f}  peak={cand['max_activation']:.1f}",
            fontsize=8,
        )
        axes[i].set_xlabel("a", fontsize=8)
        axes[i].set_ylabel("b", fontsize=8)
        fig.colorbar(im, ax=axes[i], pad=0.02, shrink=0.85)

    # Hide unused axes
    for j in range(len(candidates), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Top-6 candidate operand matrices  (ones-digit target: a%10=6, b%10=9)",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--validation_dir", default="runs/addition/lookup_validation")
    p.add_argument("--operand_plots_dir", default="runs/addition/operand_plots/operand_plots")
    p.add_argument("--out_dir", default="runs/addition/lookup_validation/plots")
    args = p.parse_args()

    val_dir = Path(args.validation_dir)
    ops_dir = Path(args.operand_plots_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(val_dir / "candidates.json") as f:
        cands_data = json.load(f)
    with open(val_dir / "inhibition_results.json") as f:
        inhibition = json.load(f)

    plot_candidate_scores(cands_data["candidates"], out_dir / "candidate_scores.pdf")
    plot_inhibition_curve(inhibition, out_dir / "inhibition_curve.pdf")
    plot_top_operand_matrices(cands_data["candidates"], ops_dir, out_dir / "top_operand_matrices.pdf")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
