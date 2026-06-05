"""Concept localisation visualisations.

Per-concept figures (saved inside each concept run directory):
  cross_layer_sim.pdf        — per-template L×L cos(δ_i, δ_j) heatmaps
  template_consistency.pdf   — per-layer template consistency curves

Cross-concept figures (saved to runs/concept_localization/):
  cross_layer_sim_all.png    — grid of L×L heatmaps for every concept
  cross_concept_sim.png      — concept×concept cosine at key layers

Usage
-----
    python -m experiments.concept_localization.plot_localisation --concept carry
    python -m experiments.concept_localization.plot_localisation --concept all
    python -m experiments.concept_localization.plot_localisation --heatmaps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

CONCEPTS = [
    "carry",
    "gcd",
    "residue_class",
    "transitive_ordering",
    "conservation",
    "causal_direction",
    "negation_scope",
    "balanced_parentheses",
    "decimal_termination",
    "doppler_shift",
    "dot_product_sign",
    "geometric_series",
    "momentum_conservation",
    "perfect_square",
    "syllogism",
    "triangle_inequality",
    "wave_interference",
]
SYMBOLIC_SUBSET = [
    "carry",
    "residue_class",
    "gcd",
    "perfect_square",
    "decimal_termination",
    "dot_product_sign",
    "triangle_inequality",
    "transitive_ordering",
    "balanced_parentheses",
    "syllogism",
]
from experiments.plot_style import CMAP_DIV, GRAY, MAUVE, NAVY, TEAL, VIOLET, apply

apply()

_TMPL_COLORS = {"T0": NAVY, "T1": TEAL, "T2": MAUVE}
_RUNS = _REPO_ROOT / "runs" / "concept_localization"
_PHASES_JSON = _RUNS / "phases.json"


def _concept_phases(concept: str, template: str = "T0") -> list[int]:
    """Return phase boundary layer indices from phases.json, or [] if absent."""
    if not _PHASES_JSON.exists():
        return []
    with open(_PHASES_JSON) as f:
        db = json.load(f)
    return [int(x) for x in db.get(concept, {}).get(template, [])]




def _concept_run_dir(concept: str, template: str = "T0") -> Path:
    """Directory containing per-anchor runs for a concept/template."""
    base = _RUNS / concept
    if "/" in concept or base.name.endswith(f"_{template}"):
        return base
    templated = base / f"{concept}_{template}"
    return templated if templated.exists() else base


def _concept_root_dir(concept: str) -> Path:
    """Top-level concept directory for aggregate/mixed-template comparison plots."""
    if "/" in concept:
        return _RUNS / concept.split("/", 1)[0]
    return _RUNS / concept


def _load(concept_dir: Path):
    results_path = concept_dir / "results.json"
    deltas_path = concept_dir / "deltas.pt"
    if not results_path.exists() or not deltas_path.exists():
        return None, None
    with open(results_path) as f:
        res = json.load(f)
    deltas = torch.load(deltas_path, map_location="cpu", weights_only=False)
    return res, deltas


def _vecs(deltas: dict, key: str, n_layers: int) -> torch.Tensor:
    """Stack layer deltas for a given key into (n_layers, d_model)."""
    bucket = deltas.get(key, {})
    d_model = next(iter(bucket.values())).shape[-1] if bucket else 1
    rows = []
    for l in range(n_layers):
        rows.append(bucket[l].float() if l in bucket else torch.zeros(d_model))
    return torch.stack(rows)


def _plot_cross_layer_sim_core(
    deltas: dict,
    n_layers: int,
    anchor_tok: str,
    concept: str,
    out_path: Path,
) -> None:
    """Shared rendering logic for cross-layer cosine similarity figures."""
    tmpl_keys = [k for k in deltas if k != "all"]
    keys = ["all"] + tmpl_keys
    n_panels = len(keys)

    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 4.5), gridspec_kw={"wspace": 0.4}
    )
    if n_panels == 1:
        axes = [axes]

    for ax, key in zip(axes, keys, strict=False):
        D = _vecs(deltas, key, n_layers)
        D_n = F.normalize(D.float(), dim=-1)
        mat = (D_n @ D_n.T).numpy()

        vmin = float(np.percentile(mat, 2))
        vmax = 1.0
        cmap = "Blues" if vmin >= 0 else CMAP_DIV
        if vmin < 0:
            abs_max = max(abs(vmin), vmax)
            vmin, vmax = -abs_max, abs_max

        im = ax.imshow(
            mat,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$\cos(\delta_i,\, \delta_j)$", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        tick_step = max(1, n_layers // 7)
        ticks = list(range(0, n_layers, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_yticklabels(ticks, fontsize=7)
        ax.set_xlabel("Layer $j$", fontsize=9)
        ax.set_ylabel("Layer $i$", fontsize=9)
        ax.grid(False)

        label = "templates: all" if key == "all" else f"template: {key}"
        ax.set_title(f'{label}   anchor: {anchor_tok}', fontsize=9, pad=5)

    fig.suptitle(f"Cross-layer cosine similarity — {concept}", fontsize=12, y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_cross_layer_sim(concept: str, template: str = "T0") -> None:
    out_dir = _concept_run_dir(concept, template)
    res, deltas = _load(out_dir)
    if res is None:
        print(f"  [skip] {concept}: no data")
        return

    cfg = res.get("config", {})
    anchor_tok = cfg.get("anchor_token", cfg.get("anchor_mode", "?"))
    n_layers = len(res["sharpness"]["norm_by_layer"])
    _plot_cross_layer_sim_core(
        deltas, n_layers, anchor_tok, concept, out_dir / "cross_layer_sim.pdf"
    )


def plot_cross_layer_sim_per_anchor(concept: str, template: str = "T0") -> None:
    """Plot cross-layer cosine similarity for every anchor_rank*_pos* directory."""
    import re as _re
    concept_dir = _concept_run_dir(concept, template)
    anchor_dirs = sorted(concept_dir.glob("anchor_rank*_pos*"))
    if not anchor_dirs:
        print(f"  [skip] {concept}: no per-anchor directories")
        return

    for anchor_dir in anchor_dirs:
        m = _re.match(r"anchor_rank(\d+)_pos(\d+)", anchor_dir.name)
        if not m:
            continue
        rank, pos = int(m.group(1)), int(m.group(2))
        res, deltas = _load(anchor_dir)
        if res is None:
            print(f"  [skip] {anchor_dir.name}: no data")
            continue
        cfg = res.get("config", {})
        tok = cfg.get("anchor_token", cfg.get("anchor_mode", "?"))
        anchor_tok = f"rank{rank} pos{pos} '{tok}'"
        n_layers = len(res["sharpness"]["norm_by_layer"])
        out_path = anchor_dir / "cross_layer_sim.png"
        _plot_cross_layer_sim_core(deltas, n_layers, anchor_tok, concept, out_path)


def plot_cross_layer_sim_data(
    layer_deltas: dict,
    n_layers: int,
    anchor_tok: str,
    concept: str,
    out_path: Path,
) -> None:
    """Plot cross-layer cosine similarity from in-memory LayerDeltas objects.

    layer_deltas: dict mapping key -> LayerDeltas (as returned by extract_layer_deltas_generic).
    """
    deltas = {key: ld.delta for key, ld in layer_deltas.items()}
    _plot_cross_layer_sim_core(deltas, n_layers, anchor_tok, concept, out_path)


def plot_template_consistency(concept: str) -> None:
    out_dir = _concept_root_dir(concept)
    res, deltas = _load(out_dir)
    if res is None:
        print(f"  [skip] {concept}: no data")
        return

    cfg = res.get("config", {})
    anchor_tok = cfg.get("anchor_token", cfg.get("anchor_mode", "?"))
    n_layers = len(res["sharpness"]["norm_by_layer"])
    layers = np.arange(n_layers)
    # norm_by_layer already activation-normalised when mean_act_norm was available
    norms = np.array([res["sharpness"]["norm_by_layer"][str(l)] for l in layers])
    is_normalised = res["sharpness"].get("normalised", False)
    mean_act_norm = res.get("mean_act_norm", {})
    tmpl_keys = [k for k in deltas if k != "all"]

    # Per-template norms from deltas.pt — apply same normalisation as "all"
    tmpl_norms = {}
    for t in tmpl_keys:
        arr = np.zeros(n_layers)
        bucket = deltas.get(t, {})
        for l in range(n_layers):
            if l in bucket:
                raw = bucket[l].float().norm().item()
                scale = mean_act_norm.get(str(l), 1.0) if is_normalised else 1.0
                arr[l] = raw / scale if scale > 0 else raw
        tmpl_norms[t] = arr

    # Template consistency: pairwise cosine per layer, computed from deltas
    pairs = [(t1, t2) for i, t1 in enumerate(tmpl_keys) for t2 in tmpl_keys[i + 1 :]]
    tc: dict[str, np.ndarray] = {}
    for t1, t2 in pairs:
        vals = []
        b1, b2 = deltas.get(t1, {}), deltas.get(t2, {})
        for l in range(n_layers):
            if l in b1 and l in b2:
                a, b = b1[l].float(), b2[l].float()
                vals.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
            else:
                vals.append(float("nan"))
        tc[f"{t1} vs {t2}"] = np.array(vals)

    if not tc:
        print(f"  [skip] {concept}: no template consistency data")
        return

    fig, ax_tc = plt.subplots(1, 1, figsize=(9, 4))
    ax_tc.spines["top"].set_visible(False)
    ax_tc.spines["right"].set_visible(False)

    colors = [NAVY, TEAL, MAUVE, GRAY]
    for (label, vals), col in zip(tc.items(), colors, strict=False):
        ax_tc.plot(layers, vals, color=col, lw=1.8, label=label)
    ax_tc.axhline(1.0, color=GRAY, lw=0.7, ls="--", alpha=0.5)
    ax_tc.set_ylabel("Template consistency", fontsize=10)
    ax_tc.set_xlabel("Layer", fontsize=10)
    ax_tc.set_ylim(top=1.05)
    ax_tc.legend(fontsize=8, loc="lower left")

    fig.suptitle(f"{concept} — template consistency", fontsize=13, y=1.01)
    out = out_dir / "template_consistency.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# Semantic ordering for cross-concept heatmap: group by concept family
_CONCEPT_ORDER = [
    "carry",
    "gcd",
    "residue_class",
    "perfect_square",
    "decimal_termination",
    "dot_product_sign",
    "conservation",
    "geometric_series",
    "wave_interference",
    "doppler_shift",
    "momentum_conservation",
    "triangle_inequality",
    "balanced_parentheses",
    "negation_scope",
    "syllogism",
    "transitive_ordering",
    "causal_direction",
]
_GROUP_SIZES = [6, 6, 5]  # modular / physical / logical
_GROUP_LABELS = ["Modular / arithmetic", "Physical / continuous", "Logical / linguistic"]


def plot_cross_layer_sim_grid(concepts: list[str], ncols: int = 6) -> None:
    """Grid of L×L cross-layer cosine matrices for all concepts (aggregate key)."""
    nrows = (len(concepts) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols, 2.7 * nrows),
        gridspec_kw={"hspace": 0.55, "wspace": 0.25},
    )
    axes_flat = list(axes.flat) if nrows > 1 else list(axes) if ncols > 1 else [axes]
    last_im = None

    for i, concept in enumerate(concepts):
        ax = axes_flat[i]
        pt = _RUNS / concept / "deltas.pt"
        rj = _RUNS / concept / "results.json"
        if not pt.exists():
            ax.set_visible(False)
            continue

        data = torch.load(pt, map_location="cpu", weights_only=False)
        agg = data.get("all", {})
        if not agg:
            ax.set_visible(False)
            continue

        n_layers = max(int(k) for k in agg) + 1
        d_model = next(iter(agg.values())).shape[-1]
        D = torch.zeros(n_layers, d_model)
        for k, v in agg.items():
            D[int(k)] = v.float()
        D_n = F.normalize(D, dim=-1)
        mat = (D_n @ D_n.T).numpy()

        last_im = ax.imshow(
            mat,
            vmin=-1,
            vmax=1,
            cmap=CMAP_DIV,
            aspect="equal",
            origin="upper",
            interpolation="nearest",
        )

        # Phase boundaries from phases.json as violet dashed lines on both axes.
        # A boundary at layer pb sits at pb-0.5 in heatmap coordinates.
        for pb in _concept_phases(concept):
            ax.axhline(pb - 0.5, color=VIOLET, lw=0.8, ls="--", alpha=0.80, zorder=3)
            ax.axvline(pb - 0.5, color=VIOLET, lw=0.8, ls="--", alpha=0.80, zorder=3)

        tick_step = max(1, n_layers // 4)
        ticks = list(range(0, n_layers, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, fontsize=5)
        ax.set_yticklabels(ticks, fontsize=5)
        ax.tick_params(length=2, pad=1)
        ax.set_title(concept.replace("_", " "), fontsize=7.5, pad=3)
        ax.grid(False)

    for j in range(len(concepts), nrows * ncols):
        axes_flat[j].set_visible(False)

    if last_im is not None:
        cb = fig.colorbar(
            last_im,
            ax=axes_flat,
            fraction=0.012,
            pad=0.02,
            shrink=0.6,
        )
        cb.set_label(r"$\cos(\delta_l,\,\delta_{l'})$", fontsize=8)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(
        r"Cross-layer cosine similarity $\cos(\delta_l,\,\delta_{l'})$ — all concepts",
        fontsize=11,
        y=1.01,
    )
    out = _RUNS / "cross_layer_sim_all.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--concept", default="carry", help="Concept name/path, 'all', or 'symbolic'")
    ap.add_argument("--template", default="T0",
                    help="Template-specific run dir to use for per-anchor plots")
    ap.add_argument("--runs_dir", default=None, help="Override runs directory")
    ap.add_argument(
        "--heatmaps",
        action="store_true",
        help="Generate cross-layer grid and cross-concept similarity heatmaps",
    )
    ap.add_argument(
        "--key_layers",
        nargs="+",
        type=int,
        default=[10, 17, 22, 30],
        help="Layers for cross-concept similarity plot",
    )
    args = ap.parse_args()

    global _RUNS
    if args.runs_dir:
        _RUNS = Path(args.runs_dir)

    if args.heatmaps:
        print("Generating cross-layer sim grid…")
        plot_cross_layer_sim_grid(CONCEPTS)
        return

    if args.concept == "all":
        concepts = CONCEPTS
    elif args.concept == "symbolic":
        concepts = SYMBOLIC_SUBSET
    else:
        concepts = [args.concept]

    for concept in concepts:
        print(f"— {concept}")
        plot_cross_layer_sim_per_anchor(concept, template=args.template)
        plot_template_consistency(concept)


if __name__ == "__main__":
    main()
