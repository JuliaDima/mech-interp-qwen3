"""Concept localisation visualisations.

Produces two figures per concept run:
  cross_layer_sim.pdf   — per-template heatmaps of cos(δ_i, δ_j)
  template_consistency.pdf   — template consistency

Usage
-----
    python -m experiments.concept_localization.plot_localisation
    python -m experiments.concept_localization.plot_localisation --concept residue_class
    python -m experiments.concept_localization.plot_localisation --concept all
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

from experiments.concept_localization.run_concept import CONCEPTS, SYMBOLIC_SUBSET
from experiments.plot_style import CMAP_DIV, GRAY, MAUVE, NAVY, RED, TEAL, VIOLET, apply

apply()

_TMPL_COLORS = {"T0": NAVY, "T1": TEAL, "T2": MAUVE}
_RUNS = _REPO_ROOT / "runs" / "concept_localization"


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


def plot_cross_layer_sim(concept: str) -> None:
    out_dir = _RUNS / concept
    res, deltas = _load(out_dir)
    if res is None:
        print(f"  [skip] {concept}: no data")
        return

    cfg = res.get("config", {})
    anchor_tok = cfg.get("anchor_token", cfg.get("anchor_mode", "?"))
    n_layers = len(res["sharpness"]["norm_by_layer"])
    tmpl_keys = [k for k in deltas if k != "all"]
    keys = ["all"] + tmpl_keys
    n_panels = len(keys)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5),
                             gridspec_kw={"wspace": 0.4})
    if n_panels == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        D = _vecs(deltas, key, n_layers)
        D_n = F.normalize(D.float(), dim=-1)
        mat = (D_n @ D_n.T).numpy()

        vmin = float(np.percentile(mat, 2))
        vmax = 1.0
        cmap = "Blues" if vmin >= 0 else CMAP_DIV
        if vmin < 0:
            abs_max = max(abs(vmin), vmax)
            vmin, vmax = -abs_max, abs_max

        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap,
                       aspect="auto", origin="upper", interpolation="nearest")
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

        label = "all templates" if key == "all" else key
        ax.set_title(f'{label}   anchor: "{anchor_tok}"', fontsize=9, pad=5)

    fig.suptitle(f"Cross-layer cosine similarity — {concept}", fontsize=12, y=1.02)
    out = out_dir / "cross_layer_sim.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_template_consistency(concept: str) -> None:
    out_dir = _RUNS / concept
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
    pairs = [(t1, t2) for i, t1 in enumerate(tmpl_keys) for t2 in tmpl_keys[i+1:]]
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
    for (label, vals), col in zip(tc.items(), colors):
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


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--concept", default="carry",
                    help="Concept name, 'all', or 'symbolic'")
    ap.add_argument("--runs_dir", default=None,
                    help="Override runs directory")
    args = ap.parse_args()

    global _RUNS
    if args.runs_dir:
        _RUNS = Path(args.runs_dir)

    if args.concept == "all":
        concepts = CONCEPTS
    elif args.concept == "symbolic":
        concepts = SYMBOLIC_SUBSET
    else:
        concepts = [args.concept]

    for concept in concepts:
        print(f"— {concept}")
        plot_cross_layer_sim(concept)
        plot_template_consistency(concept)


if __name__ == "__main__":
    main()
