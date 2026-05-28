"""Visualization for concept localization results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn.functional as F

import experiments.plot_style as ps
from experiments.concept_localization.analyze import FeatureMatch
from experiments.concept_localization.extract_deltas import LayerDeltas

_TEMPLATE_COLORS = [ps.NAVY, ps.TEAL, ps.MAUVE]


def _norm01(vals: list[float]) -> list[float]:
    """Scale to [0, 1] by absolute peak; preserves sign and zero-crossings."""
    peak = max(abs(v) for v in vals) if vals else 1.0
    return [v / peak if peak > 0 else 0.0 for v in vals]


def plot_norm_and_alignment(
    results: dict[str, LayerDeltas],
    out_path: Path,
    concept: str = "carry",
) -> None:
    """Norm + inter-layer alignment (top row) and template consistency (bottom).

    Top row: peak-normalised delta norm, optionally activation-normalised norm,
    and inter-layer cosine similarity.  Bottom: pairwise cross-template cosine
    similarity across layers.

    Normalised norm = ‖δ_l‖ / E[‖h_l‖], removing residual-stream growth bias.
    Falls back to raw ‖δ‖ when mean_act_norm is unavailable (old runs).
    Per-template curves are shown as thin lines; the aggregate "all" as bold.
    """
    ps.apply()
    ld_all = results.get("all")
    use_norm = bool(ld_all and ld_all.mean_act_norm)
    ncols = 3 if use_norm else 2
    tmpl_keys = [k for k in results if k != "all"]

    # Template-consistency pairs
    tc_pairs = [(t1, t2) for i, t1 in enumerate(tmpl_keys) for t2 in tmpl_keys[i + 1:]]
    tc: dict[str, tuple[list[int], list[float]]] = {}
    for t1, t2 in tc_pairs:
        ld1, ld2 = results.get(t1), results.get(t2)
        if ld1 is None or ld2 is None:
            continue
        shared = sorted(set(ld1.delta.keys()) & set(ld2.delta.keys()))
        vals = [
            F.cosine_similarity(
                ld1.delta[l].float().unsqueeze(0),
                ld2.delta[l].float().unsqueeze(0),
            ).item()
            for l in shared
        ]
        tc[f"{t1} vs {t2}"] = (shared, vals)

    has_tc = bool(tc)
    n_rows = 2 + use_norm + has_tc  # raw, (act_norm?), cos_sim, (tc?)
    fig, axs = plt.subplots(n_rows, 1, figsize=(5, 5 * n_rows),
                            sharex=True,
                            gridspec_kw={"hspace": 0.08})
    axs = list(axs) if n_rows > 1 else [axs]
    row = 0
    ax_raw = axs[row]; row += 1
    ax_normed = axs[row] if use_norm else None
    if use_norm: row += 1
    ax_cos = axs[row]; row += 1
    ax_tc = axs[row] if has_tc else None

    # peak of the "all" series — used to normalise every curve to [0, 1]
    ld_all_obj = results.get("all")
    all_raw = (
        [ld_all_obj.delta[l].norm().item() for l in sorted(ld_all_obj.delta.keys())]
        if ld_all_obj and ld_all_obj.delta
        else []
    )
    peak_norm = max(all_raw) if all_raw else 1.0

    for i, key in enumerate(["all"] + tmpl_keys):
        ld = results.get(key)
        if ld is None or not ld.delta:
            continue
        layers = sorted(ld.delta.keys())
        raw = [ld.delta[l].norm().item() for l in layers]
        peak_normed = [r / peak_norm for r in raw]
        is_agg = key == "all"
        lw = 2.5 if is_agg else 0.9
        ls = "-" if is_agg else "--"
        alpha = 1.0 if is_agg else 0.65
        color = ps.VIOLET if is_agg else _TEMPLATE_COLORS[(i - 1) % len(_TEMPLATE_COLORS)]
        label = "all templates" if is_agg else key

        ax_raw.plot(layers, peak_normed, label=label, color=color, linewidth=lw, linestyle=ls, alpha=alpha)

        if use_norm and ax_normed is not None and ld.mean_act_norm:
            normed = [r / ld.mean_act_norm.get(l, 1.0) for l, r in zip(layers, raw)]
            ax_normed.plot(layers, normed, label=label, color=color, linewidth=lw, linestyle=ls, alpha=alpha)

        if is_agg and ld.mean_pair_cos:
            cos_layers = sorted(ld.mean_pair_cos.keys())
            cos_vals = [ld.mean_pair_cos[l] for l in cos_layers]
            ax_raw.plot(cos_layers, cos_vals, color=ps.TEAL, linewidth=1.6,
                        linestyle=":", label=r"mean $\cos(\delta_i,\,\bar{\delta}_l)$", zorder=4)
            if use_norm and ax_normed is not None:
                ax_normed.plot(cos_layers, cos_vals, color=ps.TEAL, linewidth=1.6,
                               linestyle=":", label=r"mean $\cos(\delta_i,\,\bar{\delta}_l)$", zorder=4)

    ax_raw.set_ylabel(r"$\|\delta_l\| / \max_l(\|\delta_l\|)$")
    ax_raw.set_title(f"{concept} — norm trajectory", fontsize=11)
    ax_raw.set_ylim(bottom=0)
    ax_raw.legend(fontsize=8)

    if use_norm and ax_normed is not None:
        ax_normed.set_ylabel(r"$\|\delta_l\| / \mathbb{E}\|\mathbf{h}_l\|$")
        ax_normed.legend(fontsize=8)

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
        ax_cos.set_ylabel(r"$\cos(\delta_l,\,\delta_{l+1})$")
        ax_cos.set_ylim(-1.05, 1.05)
        if not has_tc:
            ax_cos.set_xlabel("Layer")

    # Template consistency (bottom panel — carries the x-axis label)
    if ax_tc is not None:
        colors = [ps.NAVY, ps.TEAL, ps.MAUVE, ps.GRAY]
        for (label, (layers_tc, vals)), col in zip(tc.items(), colors):
            ax_tc.plot(layers_tc, vals, color=col, lw=1.8, label=label)
        ax_tc.axhline(1.0, color=ps.GRAY, lw=0.7, ls="--", alpha=0.5)
        ax_tc.set_xlabel("Layer")
        ax_tc.set_ylabel("Template consistency")
        ax_tc.set_ylim(top=1.05)
        ax_tc.legend(fontsize=8, loc="lower left")

    fig.savefig(out_path, bbox_inches="tight")
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
    abs_max = max(abs(v) for v in cs) if cs else 1.0
    sc = ax.scatter(xs, ys, c=cs, cmap=ps.CMAP_DIV, vmin=-abs_max, vmax=abs_max, s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label=f"cos_sim(δ, W_enc row)  [max={abs_max:.3f}]")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Feature ID")
    ax.set_title(f"Top-{top_k} features aligned with {concept} delta")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_causal_overlay(
    causal_results: dict,
    delta_norms: dict[int, float],
    out_path: Path | None = None,
    concept: str = "",
    *,
    ax: plt.Axes | None = None,
    mean_act_norms: dict[int, float] | None = None,
) -> None:
    """Single-axis normalised overlay of all three causal signals.

    causal_results: dict[str, CausalScores] with keys "all" + per-template.

    All signals are scaled to [-1, 1] by their absolute peak.  Per-template
    patching and grad·δ curves are shown as thin lines; the aggregate "all"
    is bold.  The spread between template lines is the robustness indicator —
    no misleading within-template std band.

    If mean_act_norms is provided, delta_norms are divided by E[‖h‖] per layer
    before peak-normalisation, removing residual-stream growth bias.

    If `ax` is provided the plot is drawn into it and the caller owns the
    figure lifecycle (no savefig/close).  When `ax` is None a new figure is
    created, saved to `out_path`, and closed.
    """
    agg = causal_results["all"]
    layers = agg.layers
    tmpl_keys = [k for k in causal_results if k != "all"]

    if mean_act_norms:
        dn = [delta_norms.get(l, 0.0) / mean_act_norms.get(l, 1.0) for l in layers]
        dn_label = r"$\|\delta_l\| / \mathbb{E}\|\mathbf{h}_l\|$  (correlation)"
    else:
        dn = [delta_norms.get(l, 0.0) for l in layers]
        dn_label = r"$\|\delta_l\|$  (correlation)"
    pm_all = [agg.patching_mean.get(l, 0.0) for l in layers]
    gm_all = [agg.grad_dot_delta_mean.get(l, 0.0) for l in layers]

    peak_dn = max(abs(v) for v in dn) or 1.0
    peak_pm = max(abs(v) for v in pm_all) or 1.0
    peak_gm = max(abs(v) for v in gm_all) or 1.0

    own_fig = ax is None
    if own_fig:
        ps.apply()
        fig, ax = plt.subplots(figsize=(11, 4.5))

    # Delta norm (no template variants — it's the same for all)
    ax.plot(
        layers,
        [v / peak_dn for v in dn],
        color=ps.NAVY,
        linewidth=2.2,
        label=dn_label,
        zorder=2,
    )

    # Per-template thin lines — each template gets its own colour
    for i, t in enumerate(tmpl_keys):
        cs = causal_results[t]
        pm_t = [cs.patching_mean.get(l, 0.0) / peak_pm for l in layers]
        gm_t = [cs.grad_dot_delta_mean.get(l, 0.0) / peak_gm for l in layers]
        c = _TEMPLATE_COLORS[i % len(_TEMPLATE_COLORS)]
        ax.plot(layers, pm_t, color=c, linewidth=0.9, alpha=0.5, linestyle="--")
        ax.plot(layers, gm_t, color=c, linewidth=0.9, alpha=0.5, linestyle=":")

    # Aggregate bold lines
    ax.plot(
        layers,
        [v / peak_pm for v in pm_all],
        color=ps.VIOLET,
        linewidth=2.2,
        label=r"Activation patching  $\Delta(\mathrm{logit}^+ - \mathrm{logit}^-)$",
        zorder=3,
    )
    ax.plot(
        layers,
        [v / peak_gm for v in gm_all],
        color=ps.TEAL,
        linewidth=2.2,
        label="Gradient · δ",
        zorder=3,
    )

    ax.axhline(0, color=ps.GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalised signal  (each scaled to peak = 1)")
    ax.set_title(
        f"Causal signals vs. delta norm — {concept}  "
        f"(n={agg.n_pairs} pairs, {len(tmpl_keys)} templates)"
    )
    ax.legend(loc="upper left")

    if own_fig:
        fig.tight_layout()
        if out_path is not None:
            fig.savefig(out_path)
        plt.close(fig)


def plot_causal_overlay_grid(
    entries: list[tuple[str, dict, dict[int, float], dict[int, float]]],
    out_path: Path,
    ncols: int = 3,
) -> None:
    """Combined grid of causal overlay subplots, one per concept.

    entries: list of (concept_name, causal_results, delta_norms, mean_act_norms)
    mean_act_norms may be an empty dict for old runs (falls back to raw norms).
    """
    n = len(entries)
    nrows = (n + ncols - 1) // ncols

    ps.apply()
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 4.5 * nrows))
    axes_flat = list(axes.flatten()) if n > 1 else [axes]

    for i, entry in enumerate(entries):
        concept, causal_results, delta_norms = entry[0], entry[1], entry[2]
        mean_act_norms = entry[3] if len(entry) > 3 else {}
        plot_causal_overlay(
            causal_results,
            delta_norms,
            concept=concept,
            ax=axes_flat[i],
            mean_act_norms=mean_act_norms or None,
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def assemble_png_grid(
    entries: list[tuple[str, Path]],
    out_path: Path,
    ncols: int = 3,
    title: str = "",
) -> None:
    """Load existing PNGs and tile them into a labelled grid figure.

    entries: list of (label, png_path)
    """
    import matplotlib.image as mpimg

    n = len(entries)
    nrows = (n + ncols - 1) // ncols

    ps.apply()
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 5 * nrows))
    axes_flat = list(axes.flatten()) if n > 1 else [axes]

    for i, (label, png_path) in enumerate(entries):
        img = mpimg.imread(str(png_path))
        axes_flat[i].imshow(img)
        axes_flat[i].axis("off")
        axes_flat[i].set_title(label, fontsize=11, pad=4)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_causal_efficiency(
    causal_results: dict,
    delta_norms: dict[int, float],
    out_path: Path,
    concept: str = "",
) -> None:
    """Causal efficiency = grad·δ / ‖δ‖ per layer, shown per template.

    causal_results: dict[str, CausalScores] with keys "all" + per-template.

    Per-template thin lines show robustness; aggregate "all" is bold with
    shaded positive/negative regions.  Dividing by ‖δ‖ removes magnitude
    bias and shows purely directional alignment with the output gradient.
    """
    agg = causal_results["all"]
    layers = agg.layers
    tmpl_keys = [k for k in causal_results if k != "all"]

    def _efficiency(cs, norms):
        return [
            cs.grad_dot_delta_mean.get(l, 0.0) / norms.get(l, 1.0)
            if norms.get(l, 0.0) > 1e-6
            else 0.0
            for l in layers
        ]

    eff_all = _efficiency(agg, delta_norms)

    ps.apply()
    fig, ax = plt.subplots(figsize=(11, 4.0))

    # Shade aggregate positive / negative regions
    ax.fill_between(
        layers,
        eff_all,
        0,
        where=[e >= 0 for e in eff_all],
        alpha=0.15,
        color=ps.TEAL,
        label="Causally aligned",
    )
    ax.fill_between(
        layers,
        eff_all,
        0,
        where=[e < 0 for e in eff_all],
        alpha=0.15,
        color=ps.RED,
        label="Anti-causal",
    )

    # Per-template thin lines — each template gets its own colour
    for i, t in enumerate(tmpl_keys):
        eff_t = _efficiency(causal_results[t], delta_norms)
        ax.plot(
            layers,
            eff_t,
            color=_TEMPLATE_COLORS[i % len(_TEMPLATE_COLORS)],
            linewidth=0.9,
            alpha=0.55,
            linestyle="--",
            label=t,
        )

    # Aggregate bold
    ax.plot(layers, eff_all, color=ps.TEAL, linewidth=2.2, zorder=3, label="mean (all templates)")
    ax.axhline(0, color=ps.GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$(\nabla_h \,[\mathrm{logit}^+ - \mathrm{logit}^-] \cdot \delta)\,/\,\|\delta\|$")
    ax.set_title(
        f"Causal efficiency — {concept}  (n={agg.n_pairs} pairs)\n"
        "How much of each layer's delta is pointing at the output"
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_causal_scores(
    causal_results: dict,
    delta_norms: dict[int, float],
    out_path: Path,
    concept: str = "",
    mean_act_norms: dict[int, float] | None = None,
) -> None:
    """Three-panel causal analysis plot with per-template lines.

    causal_results: dict[str, CausalScores] with keys "all" + per-template.

    Left:   Activation patching Δ(logit^+ − logit^−) — bold aggregate + thin per-template.
    Centre: Gradient·δ — bold aggregate + thin per-template.
    Right:  Delta norm ‖δ‖ for reference.
    """
    agg = causal_results["all"]
    layers = agg.layers
    tmpl_keys = [k for k in causal_results if k != "all"]

    ps.apply()
    fig, axes = plt.subplots(3, 1, figsize=(5, 15), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    dn = [delta_norms.get(l, 0.0) for l in layers]

    def _draw_panel(ax, attr, color, ylabel, norm_by=None, bottom=False):
        for i, t in enumerate(tmpl_keys):
            cs = causal_results[t]
            vals = [getattr(cs, attr + "_mean").get(l, 0.0) for l in layers]
            if norm_by is not None:
                vals = [v / n if n > 1e-6 else 0.0 for v, n in zip(vals, norm_by)]
            tc = _TEMPLATE_COLORS[i % len(_TEMPLATE_COLORS)]
            ax.plot(layers, vals, color=tc, linewidth=0.9, alpha=0.55, linestyle="--", label=t)
        vals_all = [getattr(agg, attr + "_mean").get(l, 0.0) for l in layers]
        if norm_by is not None:
            vals_all = [v / n if n > 1e-6 else 0.0 for v, n in zip(vals_all, norm_by)]
        ax.plot(layers, vals_all, color=color, linewidth=2.2, label="all", zorder=3)
        ax.axhline(0, color=ps.GRAY, linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        if bottom:
            ax.set_xlabel("Layer")

    _draw_panel(axes[0], "patching", ps.VIOLET,
                r"$\Delta(\mathrm{logit}^+ - \mathrm{logit}^-)$")
    _draw_panel(axes[1], "grad_dot_delta", ps.TEAL,
                r"$(\nabla_h\,[\mathrm{logit}^+ - \mathrm{logit}^-] \cdot \delta)\;/\;\|\delta\|$",
                norm_by=dn)

    # Bottom panel: delta norm
    ax = axes[2]
    dn_raw = [delta_norms.get(l, 0.0) for l in layers]
    peak_raw = max(dn_raw) if dn_raw else 1.0
    dn_raw_01 = [v / peak_raw if peak_raw > 0 else 0.0 for v in dn_raw]
    ax.plot(layers, dn_raw_01, color=ps.NAVY, linewidth=2.0,
            label=r"$\|\delta_l\| / \max_l(\|\delta_l\|)$")
    if mean_act_norms:
        dn_anorm = [delta_norms.get(l, 0.0) / mean_act_norms.get(l, 1.0) for l in layers]
        peak_anorm = max(dn_anorm) if dn_anorm else 1.0
        dn_anorm_01 = [v / peak_anorm if peak_anorm > 0 else 0.0 for v in dn_anorm]
        ax.plot(layers, dn_anorm_01, color=ps.TEAL, linewidth=1.4, linestyle="--",
                label=r"$(\|\delta_l\| / \mathbb{E}\|\mathbf{h}_l\|) / \max_l(\|\delta_l\| / \mathbb{E}\|\mathbf{h}_l\|)$")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalised to [0, 1]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    axes[0].set_title(
        f"{concept}  —  causal analysis  (n={agg.n_pairs} pairs, {len(tmpl_keys)} templates)",
        fontsize=11,
    )
    axes[1].set_title("Gradient–delta alignment", fontsize=9, pad=3)
    axes[2].set_title(
        "Concept delta norm  (mean across all pairs and templates)", fontsize=9, pad=3
    )
    fig.savefig(out_path, bbox_inches="tight")
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
        vals = [m.projection for m in matches]
        colors = [ps.RED if v > 0 else ps.NAVY for v in vals]
        ax.barh(labels, vals, color=colors)
        ax.axvline(0, color=ps.GRAY, linewidth=0.6)
        ax.set_xlim(-1, 1)
        ax.set_title(f"L{layer}", fontsize=8)
        ax.tick_params(axis="both", labelsize=6)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(rf"Top-{top_k} {concept} features by layer  ($\hat{{W}}_{{\mathrm{{enc}}}}\,\delta_l$ projection)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
