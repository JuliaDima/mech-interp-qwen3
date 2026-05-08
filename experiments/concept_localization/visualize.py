"""Visualization for concept localization results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn.functional as F

import experiments.plot_style as ps
from experiments.concept_localization.analyze import FeatureMatch
from experiments.concept_localization.extract_deltas import LayerDeltas

_TEMPLATE_COLORS = [ps.NAVY, ps.TEAL, ps.MAUVE]


def plot_norm_and_alignment(
    results: dict[str, LayerDeltas],
    out_path: Path,
    concept: str = "carry",
) -> None:
    """Two-panel plot: delta norm by layer (left) and inter-layer cos-sim (right).

    Per-template curves are shown as thin lines; the aggregate "all" as bold.
    """
    ps.apply()
    fig, (ax_norm, ax_cos) = plt.subplots(1, 2, figsize=(14, 5))

    tmpl_keys = [k for k in results if k != "all"]
    for i, key in enumerate(["all"] + tmpl_keys):
        ld = results.get(key)
        if ld is None or not ld.delta:
            continue
        layers = sorted(ld.delta.keys())
        norms = [ld.delta[l].norm().item() for l in layers]
        if key == "all":
            ax_norm.plot(layers, norms, label="all templates", color=ps.VIOLET, linewidth=2.5)
        else:
            color = _TEMPLATE_COLORS[(i - 1) % len(_TEMPLATE_COLORS)]
            ax_norm.plot(layers, norms, label=key, color=color, linewidth=1.0, alpha=0.7)

    ps.phase_vlines(ax_norm)
    ax_norm.set_xlabel("Layer")
    ax_norm.set_ylabel("‖δ‖")
    ax_norm.set_title(f"Delta norm — {concept}")
    ax_norm.legend()

    # Inter-layer cosine similarity for aggregate only
    if "all" in results and results["all"].delta:
        ld = results["all"]
        layers = sorted(ld.delta.keys())
        cos_sims: list[float] = []
        mid_layers: list[float] = []
        for i in range(len(layers) - 1):
            a = ld.delta[layers[i]]
            b = ld.delta[layers[i + 1]]
            cos_sims.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
            mid_layers.append((layers[i] + layers[i + 1]) / 2)

        ax_cos.plot(mid_layers, cos_sims, color=ps.NAVY, linewidth=1.8)
        ax_cos.axhline(0, color=ps.GRAY, linestyle="--", linewidth=0.8)
        ps.phase_vlines(ax_cos)
        ax_cos.set_xlabel("Layer (midpoint)")
        ax_cos.set_ylabel("cos_sim(δ_l, δ_{l+1})")
        ax_cos.set_title(f"Inter-layer delta alignment — {concept}")
        ax_cos.set_ylim(-1.05, 1.05)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_template_consistency(
    consistency: dict[int, dict[str, float]],
    out_path: Path,
    concept: str = "carry",
) -> None:
    """Plot cross-template delta cosine similarity at each layer."""
    if not consistency:
        return

    ps.apply()
    layers = sorted(consistency.keys())
    pair_keys = sorted({k for row in consistency.values() for k in row})

    fig, ax = plt.subplots(figsize=(12, 4))
    for i, pk in enumerate(pair_keys):
        vals = [consistency[l].get(pk, float("nan")) for l in layers]
        color = _TEMPLATE_COLORS[i % len(_TEMPLATE_COLORS)]
        ax.plot(layers, vals, label=pk, color=color, linewidth=1.5)

    ax.axhline(1.0, color=ps.GRAY, linestyle="--", linewidth=0.8)
    ps.phase_vlines(ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos_sim")
    ax.set_title(f"Template consistency — {concept}")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_projections(
    projections: dict[int, list[FeatureMatch]],
    out_path: Path,
    top_k: int = 15,
    concept: str = "carry",
) -> None:
    """Scatter: x=layer, y=feature_id, color=cos_sim."""
    xs, ys, cs = [], [], []
    for layer, matches in projections.items():
        for m in matches[:top_k]:
            xs.append(layer)
            ys.append(m.feature_id)
            cs.append(m.cos_sim)

    if not xs:
        return

    ps.apply()
    fig, ax = plt.subplots(figsize=(max(10, len(projections) * 0.5), 5))
    sc = ax.scatter(xs, ys, c=cs, cmap=ps.CMAP_DIV, vmin=-1, vmax=1, s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="cos_sim(δ, W_enc row)")
    ps.phase_vlines(ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Feature ID")
    ax.set_title(f"Top-{top_k} features aligned with {concept} delta")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_top_features_per_layer(
    projections: dict[int, list[FeatureMatch]],
    out_path: Path,
    top_k: int = 5,
    concept: str = "carry",
) -> None:
    """Bar chart: top-k cos_sim values at each layer (sorted by |cos_sim|)."""
    layers = sorted(projections.keys())
    n = len(layers)
    if n == 0:
        return

    ps.apply()
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5), squeeze=False)

    for idx, layer in enumerate(layers):
        ax = axes[idx // cols][idx % cols]
        matches = projections[layer][:top_k]
        labels = [str(m.feature_id) for m in matches]
        vals = [m.cos_sim for m in matches]
        colors = [ps.RED if v > 0 else ps.NAVY for v in vals]
        ax.barh(labels, vals, color=colors)
        ax.axvline(0, color=ps.GRAY, linewidth=0.6)
        ax.set_xlim(-1, 1)
        ax.set_title(f"L{layer}", fontsize=8)
        ax.tick_params(axis="both", labelsize=6)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f"Top-{top_k} {concept} features by layer (cos_sim with δ)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
