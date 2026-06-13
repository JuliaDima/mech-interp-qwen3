"""Per-anchor emergence and anchor-layer-grid plots for any concept with an emergence.npy file.

Provides two figures:

  emergence_per_anchor.pdf

  anchor_layer_grid_<template>_top<k>.pdf
      

Usage
-----
    python experiments/concept_localization/plot_emergence_per_anchor.py --concept doppler_shift
    python experiments/concept_localization/plot_emergence_per_anchor.py --all
    python experiments/concept_localization/plot_emergence_per_anchor.py --concept carry --top_k 6 --highlight_k 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import experiments.plot_style as ps
from experiments.concept_localization.peak_layers import select_peak_layers, PeakLayerResult

BASE = _REPO_ROOT / "runs" / "concept_localization"


def _non_monotonicity(curve: np.ndarray) -> float:
    """Prominence of the most prominent local peak.

    For each interior position i, computes min(rise to i, fall from i), where
    rise = c_i - min(c[:i]) and fall = c_i - min(c[i+1:]).  Returns the maximum
    across all positions.  A monotone curve scores near zero; a sharp bump that
    both climbs and descends significantly scores high.
    """
    best = 0.0
    for i in range(1, len(curve) - 1):
        rise = curve[i] - curve[:i].min()
        fall = curve[i] - curve[i + 1:].min()
        if rise > 0 and fall > 0:
            best = max(best, min(rise, fall))
    return float(best)


def _build_slots(norms_raw, labels):
    """Return a list of plot slots.

    Each slot is either:
      ("prefix", prefix_label, zero_indices)  — collapsed zero-signal prefix
      ("anchor", idx)                          — individual active anchor
    """
    n = norms_raw.shape[0]
    first_active = next((i for i in range(n) if norms_raw[i].max() > 1e-8), n)
    slots = []
    if first_active > 0:
        prefix_label = "".join(labels[i] for i in range(first_active))
        slots.append(("prefix", prefix_label, list(range(first_active))))
    for i in range(first_active, n):
        slots.append(("anchor", i))
    return slots


def _peak_norm(vals: list[float]) -> list[float]:
    peak = max(abs(v) for v in vals) if vals else 1.0
    return [v / peak if peak > 1e-12 else 0.0 for v in vals]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_concept_anchor_data(concept: str) -> dict | None:
    """Load and compute all anchor data from emergence.npy.

    Returns a dict with keys:
        norms_raw, act_normed, layers, labels, labels_neg,
        template_key, template_str, prompt_annotated,
        active, non_mono, nm_ranks  (pos_idx -> 1-based rank among active),
        slots  (as produced by _build_slots)
    Returns None if emergence.npy is missing.
    """
    path = BASE / concept / "emergence.npy"
    if not path.exists():
        return None

    d            = np.load(path, allow_pickle=True).item()
    norms_raw    = d["norms_raw"]
    act_raw      = d["act_norms_raw"]
    layers       = np.array(d["layers"])
    labels       = d.get("token_labels_pos", [str(i) for i in range(norms_raw.shape[0])])
    labels_neg   = d.get("token_labels_neg", labels)
    template_key = d.get("template", "T0")

    template_str: str | None = None
    try:
        import importlib
        mod = importlib.import_module(f"data.concept_datasets.{concept}_dataset")
        templates = getattr(mod, "TEMPLATES", {})
        if template_key in templates:
            template_str = templates[template_key][0]
    except Exception:
        pass
    if template_str is None:
        template_str = "".join(str(t) for t in labels)

    actual_prompt = "".join(labels)
    _parts = re.split(r'\{(\w+)\}', template_str)
    _group_map: list[str] = []
    _pattern_parts: list[str] = []
    for i, p in enumerate(_parts):
        if i % 2 == 0:
            _pattern_parts.append(re.escape(p))
        else:
            _pattern_parts.append(f"(?P<_v{len(_group_map)}>.+?)")
            _group_map.append(p)
    _pattern = "".join(_pattern_parts) + "$"
    _m = re.match(_pattern, actual_prompt, re.DOTALL)

    var_token_positions: dict[str, list[int]] = {}
    if _m:
        char_pos = 0
        tok_char_ends = []
        for tok in labels:
            char_pos += len(tok)
            tok_char_ends.append(char_pos)
        tok_char_starts = [0] + tok_char_ends[:-1]
        for g_idx, var_name in enumerate(_group_map):
            try:
                cs, ce = _m.start(f"_v{g_idx}"), _m.end(f"_v{g_idx}")
                idxs = [i for i, (s, e) in enumerate(zip(tok_char_starts, tok_char_ends))
                        if s < ce and e > cs]
                var_token_positions.setdefault(var_name, [])
                var_token_positions[var_name].extend(
                    i for i in idxs if i not in var_token_positions[var_name])
            except IndexError:
                pass

    all_var_tok_idxs = {i for idxs in var_token_positions.values() for i in idxs}
    prompt_annotated = "".join(
        f"{{{tok}}}" if i in all_var_tok_idxs else tok
        for i, tok in enumerate(labels)
    )

    row_max    = act_raw.max(axis=1, keepdims=True).clip(min=1e-8)
    act_normed = act_raw / row_max

    active   = [i for i in range(norms_raw.shape[0]) if norms_raw[i].max() > 1e-8]
    non_mono = {i: _non_monotonicity(act_normed[i]) for i in active}
    by_nm    = sorted(active, key=lambda i: non_mono[i], reverse=True)
    nm_ranks = {idx: r + 1 for r, idx in enumerate(by_nm)}

    raw_mean_cos = d.get("mean_cos")
    mean_cos_map = {}
    if raw_mean_cos is not None:
        for i in active:
            if i < len(raw_mean_cos):
                mean_cos_map[i] = float(raw_mean_cos[i])

    return dict(
        norms_raw=norms_raw, act_normed=act_normed,
        layers=layers, labels=labels, labels_neg=labels_neg,
        template_key=template_key, template_str=template_str,
        prompt_annotated=prompt_annotated,
        active=active, non_mono=non_mono, nm_ranks=nm_ranks,
        mean_cos=mean_cos_map,
        slots=_build_slots(norms_raw, labels),
    )


# ── Emergence per-anchor plot ──────────────────────────────────────────────────

def _draw_anchor_subplot(ax, layers, norms_raw_i, act_normed_i, title, highlight=False):
    l1, = ax.plot(layers, norms_raw_i, color=ps.VIOLET, lw=1.6, label="raw ‖δ‖")
    ax.set_ylabel("raw ‖δ‖", fontsize=7, color=ps.VIOLET)
    ax.tick_params(axis="y", labelcolor=ps.VIOLET, labelsize=7)
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    l3, = ax2.plot(layers, act_normed_i, color=ps.TEAL, lw=1.4, ls=":", label="double-norm")
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis="y", labelcolor=ps.TEAL, labelsize=7)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor(ps.TEAL)
    ax2.set_ylabel("double-norm", fontsize=7, color=ps.TEAL)

    title_kw = dict(fontsize=9, pad=4)
    if highlight:
        title_kw.update(color=ps.RED, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(ps.RED)
            spine.set_linewidth(1.8)
        for spine in ax2.spines.values():
            spine.set_edgecolor(ps.RED)
            spine.set_linewidth(1.8)
    ax.set_title(title, **title_kw)
    return l1, l3


def plot_emergence_per_anchor(concept: str) -> Path | None:
    data = load_concept_anchor_data(concept)
    if data is None:
        print(f"  [{concept}] emergence.npy not found — skipping")
        return None

    norms_raw        = data["norms_raw"]
    act_normed       = data["act_normed"]
    layers           = data["layers"]
    labels           = data["labels"]
    template_key     = data["template_key"]
    template_str     = data["template_str"]
    prompt_annotated = data["prompt_annotated"]
    active           = data["active"]
    non_mono         = data["non_mono"]
    slots            = data["slots"]

    top3_idx = sorted(active, key=lambda i: non_mono[i], reverse=True)[:3]
    ranks    = {idx: rank + 1 for rank, idx in enumerate(top3_idx)}

    n_slots = len(slots)
    NCOLS   = 4
    NROWS   = (n_slots + NCOLS - 1) // NCOLS

    ps.apply()
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 3.2, NROWS * 2.6), sharex=True)
    if NROWS * NCOLS == 1:
        axes = np.array([[axes]])
    axes_flat = axes.flat

    legend_handles = [
        Line2D([0], [0], color=ps.VIOLET, lw=1.6, label="raw ‖δ‖"),
        Line2D([0], [0], color=ps.TEAL,   lw=1.4, ls=":", label="double-norm"),
    ]

    for slot_idx, slot in enumerate(slots):
        ax = axes_flat[slot_idx]
        ax.set_xlim(layers[0], layers[-1])
        if slot_idx >= (NROWS - 1) * NCOLS:
            ax.set_xlabel("layer", fontsize=8)

        if slot[0] == "prefix":
            _, prefix_label, _ = slot
            ax.plot(layers, np.zeros_like(layers, dtype=float), color=ps.GRAY, lw=1.0, ls="--")
            ax.set_ylim(0, 1)
            ax.set_title(f"prefix  '{prefix_label}'", fontsize=8, pad=4, color=ps.GRAY)
            ax.tick_params(axis="y", labelsize=7)
            ax2 = ax.twinx()
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="y", labelcolor=ps.TEAL, labelsize=7)
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_edgecolor(ps.TEAL)
            ax2.set_ylabel("double-norm", fontsize=7, color=ps.TEAL)
        else:
            _, idx = slot
            label  = repr(labels[idx]) if idx < len(labels) else str(idx)
            is_top = idx in ranks
            nm     = non_mono.get(idx, 0.0)
            title  = (
                f"#{ranks[idx]}  pos {idx}  {label}  nm={nm:.2f}"
                if is_top else f"pos {idx}  {label}  nm={nm:.2f}"
            )
            _draw_anchor_subplot(ax, layers, norms_raw[idx], act_normed[idx], title,
                                 highlight=is_top)

        if slot_idx == 0:
            ax.legend(handles=legend_handles, fontsize=7, loc="upper left")

    for slot_idx in range(n_slots, NROWS * NCOLS):
        axes_flat[slot_idx].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    fig.text(0.5, 0.995, f"{concept} — delta norm per anchor  (top-3 by non-monotonicity highlighted)",
             ha="center", va="top", fontsize=10, fontweight="bold", transform=fig.transFigure)
    fig.text(0.5, 0.965, f"template {template_key}:  {template_str}",
             ha="center", va="top", fontsize=8, color=ps.GRAY, style="italic",
             transform=fig.transFigure)
    fig.text(0.5, 0.938, f"prompt:  {prompt_annotated}  — bracketed tokens are variables",
             ha="center", va="top", fontsize=8, color=ps.NAVY, transform=fig.transFigure)

    out = BASE / concept / "emergence_per_anchor.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{concept}] saved → {out}")
    return out


# ── Anchor-layer grid plot ─────────────────────────────────────────────────────

PANEL_W = 4.2
PANEL_H = 3.6
N_ROWS  = 5

ROW_LABELS = [
    "E_dec projection",
    "delta trajectory",
    "layer cosine sim",
    "null comparison",
    "causal overlay",
]


def _load_anchor_dir(anchor_dir: Path, template: str) -> dict | None:
    results_path = anchor_dir / "results.json"
    deltas_path  = anchor_dir / "deltas.pt"
    if not results_path.exists() or not deltas_path.exists():
        return None
    results = json.loads(results_path.read_text())
    raw    = torch.load(deltas_path, map_location="cpu", weights_only=False)
    deltas = raw.get(template, raw.get("all", {}))
    if not deltas:
        return None
    null_path = anchor_dir / "null" / "null_permutation.json"
    null = json.loads(null_path.read_text()) if null_path.exists() else None
    return {"results": results, "deltas": deltas, "null": null}


def discover_anchors(concept: str, template: str, top_k: int) -> list[dict]:
    """Select top-k anchors by combined score (null+patching+grad).

    Stage 1 (pre-pipeline): make_gif ranks candidates by mean_cos and submits
    --candidates jobs.  Stage 2 (here, post-pipeline): from those anchor dirs,
    keep only those with a valid combined score and return the top-k.
    """
    data = load_concept_anchor_data(concept)
    if data is None:
        return []

    norms_raw  = data["norms_raw"]
    act_normed = data["act_normed"]
    layers     = [int(l) for l in data["layers"]]
    labels     = data["labels"]

    # ── candidate pool: only dirs that exist on disk ─────────────────────────
    template_dir = BASE / concept / f"{concept}_{template}"
    pos_to_dir: dict[int, Path] = {}
    pos_to_init_rank: dict[int, int] = {}
    if template_dir.exists():
        for ad in template_dir.glob("anchor_rank*_pos*"):
            m = re.fullmatch(r"anchor_rank(\d+)_pos(\d+)", ad.name)
            if m:
                pos  = int(m.group(2))
                rank = int(m.group(1))
                pos_to_dir[pos]       = ad
                pos_to_init_rank[pos] = rank

    # ── build entries for all available anchor dirs ───────────────────────────
    # combined_score = peak score when valid, 0.0 when null-failing (sorts to right)
    entries = []
    for pos_idx, anchor_dir in pos_to_dir.items():
        if pos_idx >= norms_raw.shape[0]:
            continue
        pr   = _load_peak_result(anchor_dir, template)
        supp = _load_anchor_dir(anchor_dir, template)
        token = labels[pos_idx] if pos_idx < len(labels) else str(pos_idx)
        if pr is not None and pr.valid and pr.peak_scores:
            combined_score = float(pr.peak_scores[0])
        else:
            combined_score = 0.0   # null-failing or incomplete — shown but not highlighted
        entries.append({
            "pos":            pos_idx,
            "token":          token,
            "combined_score": combined_score,
            "init_rank":      pos_to_init_rank.get(pos_idx, 0),
            "norms_raw":      norms_raw[pos_idx],
            "act_normed":     act_normed[pos_idx],
            "layers":         layers,
            "dir":            anchor_dir,
            "results":        supp["results"] if supp else {},
            "deltas":         supp["deltas"]  if supp else {},
            "null":           supp["null"]    if supp else None,
            "peak_result":    pr,
        })

    # ── select top_k by combined score, assign ranks ─────────────────────────
    sorted_all = sorted(entries, key=lambda e: e["combined_score"], reverse=True)
    top = sorted_all[:top_k]
    for crank, e in enumerate(top, start=1):
        e["combined_rank"] = crank

    # Display columns in token-position order (left = earlier in prompt)
    return sorted(top, key=lambda e: e["pos"])


def _load_peak_result(anchor_dir: Path | None, template: str) -> PeakLayerResult | None:
    if anchor_dir is None:
        return None
    try:
        return select_peak_layers(anchor_dir, template=template)
    except Exception:
        return None


def _draw_feature_projection(ax, results: dict, layers: list[int],
                              show_colorbar: bool = True) -> list:
    top_k = int(results.get("config", {}).get("top_k", 15))
    xs, ys, cs = [], [], []
    for layer_s, rows in results.get("top_features_by_layer", {}).items():
        layer = int(layer_s)
        for row in rows[:top_k]:
            xs.append(layer)
            ys.append(int(row["feature_id"]))
            cs.append(float(row.get("cos_sim", 0.0)))
    if xs:
        vmax = max(abs(v) for v in cs) or 1.0
        sc = ax.scatter(xs, ys, c=cs, cmap=ps.CMAP_DIV,
                        vmin=-vmax, vmax=vmax, s=14, alpha=0.85)
        if show_colorbar:
            cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.06)
            cb.set_label("E_dec cos", fontsize=6)
            cb.ax.tick_params(labelsize=5, length=0)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
    ax.set_ylabel("feature id", fontsize=7)
    return []


def _draw_delta_trajectory(ax, norms_raw_i: np.ndarray, act_normed_i: np.ndarray,
                            layers: list[int], show_legend: bool = False,
                            null: dict | None = None,
                            peak_result: PeakLayerResult | None = None) -> list:
    peak = float(norms_raw_i.max()) if norms_raw_i.max() > 1e-12 else 1.0
    norms_normed = norms_raw_i / peak
    ax.plot(layers, norms_normed, color=ps.VIOLET, lw=1.6, label=r"‖δ‖ / max‖δ‖")
    ax.set_ylabel(r"‖δ‖ / max‖δ‖", fontsize=7, color=ps.VIOLET)
    ax.tick_params(axis="y", labelcolor=ps.VIOLET, labelsize=6)
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    ax2.plot(layers, act_normed_i, color=ps.TEAL, lw=1.4, ls=":", label="double-norm")
    ax2.set_ylabel("double-norm", fontsize=7, color=ps.TEAL)
    ax2.tick_params(axis="y", labelcolor=ps.TEAL, labelsize=6)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor(ps.TEAL)
    ax2.set_ylim(bottom=0)

    # Null-excess overlay: fill under the norm curve, capped at how far above
    # null mean + 1 SD the real signal is (same scale as norms_normed).
    if null is not None:
        nlayers = [int(x) for x in null.get("layers", [])]
        real_n  = np.asarray(null.get("real_norms",  []), dtype=float)
        nulls   = np.asarray(null.get("null_norms",  []), dtype=float)
        if real_n.size > 0 and nulls.size > 0 and len(nlayers) == len(real_n):
            null_mean = nulls.mean(axis=0)
            null_std  = nulls.std(axis=0)
            excess    = np.maximum(0.0, real_n - (null_mean + null_std))
            ax.fill_between(nlayers, 0, excess,
                            color=ps.MAUVE, alpha=0.40,
                            label="excess above null+1SD")

    # Peak layer vlines: use select_peak_layers result if available, else fallback to argmax
    if peak_result is not None and peak_result.valid and peak_result.peak_layers:
        for rank, pl in enumerate(peak_result.peak_layers):
            lw = 1.1 if rank == 0 else 0.7
            ax.axvline(pl, color=ps.VIOLET, lw=lw, ls=":", alpha=0.85)
            ax.text(pl + 0.3, 0.92 - rank * 0.14, f"L{pl}",
                    fontsize=5, color=ps.VIOLET, alpha=0.85,
                    transform=ax.get_xaxis_transform())
    else:
        fallback = layers[int(np.argmax(norms_normed))]
        ax.axvline(fallback, color=ps.VIOLET, lw=0.8, ls=":", alpha=0.7)

    if show_legend:
        handles = [
            Line2D([0], [0], color=ps.VIOLET, lw=1.6, label=r"‖δ‖ / max‖δ‖"),
            Line2D([0], [0], color=ps.TEAL,   lw=1.4, ls=":", label="double-norm"),
        ]
        if null is not None:
            import matplotlib.patches as mpatches
            handles.append(mpatches.Patch(color=ps.MAUVE, alpha=0.40, label="excess above null+1SD"))
        ax.legend(handles=handles, fontsize=5, loc="upper left", framealpha=0.7)
    return [ax2]


def _draw_layer_cosine(ax, deltas: dict[int, torch.Tensor],
                        layers: list[int], show_colorbar: bool = True) -> list:
    mat = np.full((len(layers), len(layers)), np.nan)
    for i, li in enumerate(layers):
        if li not in deltas:
            continue
        ai = deltas[li].float().unsqueeze(0)
        for j, lj in enumerate(layers):
            if lj not in deltas:
                continue
            mat[i, j] = F.cosine_similarity(ai, deltas[lj].float().unsqueeze(0)).item()
    im = ax.imshow(
        mat, origin="lower", aspect="auto", cmap=ps.CMAP_DIV,
        vmin=-1, vmax=1,
        extent=[min(layers) - 0.5, max(layers) + 0.5,
                min(layers) - 0.5, max(layers) + 0.5],
    )
    if show_colorbar:
        cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.06)
        cb.set_label("cos", fontsize=6)
        cb.ax.tick_params(labelsize=5, length=0)
    ax.set_ylabel("layer", fontsize=7)
    return []


def _draw_null(ax, null: dict | None, layers: list[int], show_legend: bool = False) -> list:
    if not null:
        ax.text(0.5, 0.5, "no null", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color=ps.GRAY)
        ax.set_ylabel("norm", fontsize=7)
        return []
    nlayers = [int(x) for x in null.get("layers", layers)]
    real  = np.asarray(null.get("real_norms", []), dtype=float)
    nulls = np.asarray(null.get("null_norms", []), dtype=float)
    if real.size == 0 or nulls.size == 0:
        ax.text(0.5, 0.5, "incomplete", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        return []
    K    = nulls.shape[0]
    mean = nulls.mean(axis=0)
    std  = nulls.std(axis=0)
    ax.fill_between(nlayers, mean - std, mean + std, color=ps.GRAY, alpha=0.20,
                    label=f"null mean ± 1 SD  ({K} shuffles)")
    ax.plot(nlayers, np.percentile(nulls, 95, axis=0), color=ps.GRAY, lw=0.9, ls=":",
            label="null 95th percentile")
    ax.plot(nlayers, mean, color=ps.GRAY, lw=1.1)
    ax.plot(nlayers, real, color=ps.VIOLET, lw=1.8,
            label=r"real $\|\delta_i\|/\max\|\delta_i\|$")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("norm", fontsize=7)
    if show_legend:
        ax.legend(fontsize=5, loc="upper right", framealpha=0.7)
    return []


def _draw_causal(ax, results: dict, deltas: dict[int, torch.Tensor],
                  template: str, layers: list[int]) -> list:
    causal = results.get("causal")
    if not causal:
        ax.text(0.5, 0.5, "no causal", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color=ps.GRAY)
        ax.set_ylabel("signal", fontsize=7)
        return []
    cs = causal.get(template) or causal.get("all")
    if not cs:
        ax.text(0.5, 0.5, f"no {template}", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        return []
    patch = [float(cs.get("patching_mean",      {}).get(str(l), 0.0)) for l in layers]
    grad  = [float(cs.get("grad_dot_delta_mean", {}).get(str(l), 0.0)) for l in layers]
    delta = [float(deltas[l].norm().item()) if l in deltas else 0.0     for l in layers]
    ax.plot(layers, _peak_norm(delta), color=ps.NAVY,   lw=1.6, label="delta")
    ax.plot(layers, _peak_norm(patch), color=ps.VIOLET, lw=1.6, label="patch")
    ax.plot(layers, _peak_norm(grad),  color=ps.TEAL,   lw=1.6, label="grad·δ")
    ax.axhline(0, color=ps.GRAY, lw=0.7, ls="--")
    ax.set_ylabel("signal / peak", fontsize=7)
    ax.legend(fontsize=5, loc="upper left", framealpha=0.7)
    return []


def _apply_highlight(all_axes: list) -> None:
    for ax in all_axes:
        for spine in ax.spines.values():
            spine.set_edgecolor(ps.RED)
            spine.set_linewidth(1.8)


def plot_anchor_layer_grid(
    concept: str,
    template: str = "T0",
    top_k: int = 6,
    highlight_k: int = 3,
    out_path: Path | None = None,
) -> Path | None:
    anchors = discover_anchors(concept, template, top_k)
    if not anchors:
        print(f"  [{concept}] no anchors found for template={template} top_k={top_k}")
        return None

    K  = len(anchors)
    cd = load_concept_anchor_data(concept)
    template_str     = cd["template_str"]     if cd else template
    prompt_annotated = cd["prompt_annotated"] if cd else None

    ps.apply()
    fig, axes = plt.subplots(
        N_ROWS, K,
        figsize=(PANEL_W * K, PANEL_H * N_ROWS),
        gridspec_kw={"hspace": 0.22, "wspace": 0.35},
        squeeze=False,
    )

    for col, entry in enumerate(anchors):
        results      = entry["results"]
        deltas       = entry["deltas"]
        null         = entry["null"]
        layers       = entry["layers"]
        norms_raw_i  = entry["norms_raw"]
        act_normed_i = entry["act_normed"]
        combined_rank = entry.get("combined_rank", col + 1)
        pr           = entry.get("peak_result")
        is_null      = pr is not None and not pr.valid
        highlight    = combined_rank <= highlight_k and not is_null
        ticks        = list(range(min(layers), max(layers) + 1, 5))
        leftmost     = (col == 0)
        col_axes: list = []

        extra = _draw_feature_projection(axes[0, col], results, layers,
                                          show_colorbar=leftmost)
        col_axes += [axes[0, col]] + extra
        if leftmost:
            axes[0, col].set_title(ROW_LABELS[0], fontsize=8, pad=4)

        extra = _draw_delta_trajectory(axes[1, col], norms_raw_i, act_normed_i,
                                       layers, show_legend=leftmost, null=null,
                                       peak_result=entry.get("peak_result"))
        col_axes += [axes[1, col]] + extra
        if leftmost:
            axes[1, col].set_title(ROW_LABELS[1], fontsize=8, pad=4)

        extra = _draw_layer_cosine(axes[2, col], deltas, layers,
                                    show_colorbar=leftmost)
        col_axes += [axes[2, col]] + extra
        if leftmost:
            axes[2, col].set_title(ROW_LABELS[2], fontsize=8, pad=4)

        extra = _draw_null(axes[3, col], null, layers, show_legend=leftmost)
        col_axes += [axes[3, col]] + extra
        if leftmost:
            axes[3, col].set_title(ROW_LABELS[3], fontsize=8, pad=4)

        extra = _draw_causal(axes[4, col], results, deltas, template, layers)
        col_axes += [axes[4, col]] + extra
        if leftmost:
            axes[4, col].set_title(ROW_LABELS[4], fontsize=8, pad=4)

        if not leftmost:
            for ax in col_axes:
                ax.set_ylabel("")

        if highlight:
            _apply_highlight(col_axes)

        for row in range(N_ROWS):
            ax = axes[row, col]
            ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
            if row == N_ROWS - 1:
                ax.set_xlabel("layer", fontsize=7)
            ax.set_xticks(ticks)
            ax.tick_params(axis="x", labelsize=6, labelbottom=True)
            ax.tick_params(axis="y", labelsize=6, labelleft=True, labelright=False)
            ax.grid(axis="x", color=ps.GRAY, alpha=0.18, lw=0.5)

    fig_h    = PANEL_H * N_ROWS
    y1       = 0.995
    y2       = y1 - 0.23 / fig_h
    y3       = y2 - 0.21 / fig_h
    rect_top = y3 - 0.55 / fig_h

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()
    fig.subplots_adjust(top=rect_top)

    for col, entry in enumerate(anchors):
        combined_rank = entry.get("combined_rank", col + 1)
        pr        = entry.get("peak_result")
        is_null   = pr is not None and not pr.valid
        highlight = combined_rank <= highlight_k and not is_null
        hdr_color = ps.RED if highlight else (ps.GRAY if is_null else "black")

        init_rank  = entry.get("init_rank", "?")
        col_header = (
            f"pos {entry['pos']} '{entry['token']}'  "
            f"(init {init_rank} → rank {combined_rank})  score={entry['combined_score']:.2f}"
        )
        axes[0, col].annotate(
            col_header,
            xy=(0.5, 1), xycoords="axes fraction",
            xytext=(0, 18), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
            fontweight="bold" if highlight else "normal",
            color=hdr_color,
            annotation_clip=False,
        )

    fig.text(
        0.5, y1,
        f"{concept} — anchor layer summary  "
        f"(top {top_k} anchors by position;  top {highlight_k} by rank highlighted)",
        ha="center", va="top", fontsize=10, fontweight="bold",
        transform=fig.transFigure,
    )
    fig.text(
        0.5, y2,
        f"template {template}:  {template_str}",
        ha="center", va="top", fontsize=8, color=ps.GRAY, style="italic",
        transform=fig.transFigure,
    )
    y3_text = (f"prompt:  {prompt_annotated}  — bracketed tokens are variables"
               if prompt_annotated else
               "  ".join(f"pos {e['pos']} '{e['token']}'" for e in anchors))
    fig.text(0.5, y3, y3_text, ha="center", va="top", fontsize=8, color=ps.NAVY,
             transform=fig.transFigure)

    if out_path is None:
        out_path = BASE / concept / f"anchor_layer_grid_{template}_top{top_k}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{concept}] saved → {out_path}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--concept", help="Single concept name")
    group.add_argument("--all", action="store_true",
                       help="Run for every concept with emergence.npy")
    parser.add_argument("--template",    default="T0")
    parser.add_argument("--top_k",       type=int, default=6,
                        help="Grid: number of top-ranked anchors to include")
    parser.add_argument("--highlight_k", type=int, default=3,
                        help="Grid: highlight columns with rank <= this in red")
    parser.add_argument("--emergence_only", action="store_true",
                        help="Skip the anchor-layer grid plot")
    parser.add_argument("--grid_only",      action="store_true",
                        help="Skip the emergence per-anchor plot")
    args = parser.parse_args()

    if args.all:
        concepts = sorted(p.parent.name for p in BASE.glob("*/emergence.npy"))
        print(f"Found {len(concepts)} concepts with emergence.npy")
    else:
        concepts = [args.concept]

    for concept in concepts:
        if not args.grid_only:
            plot_emergence_per_anchor(concept)
        if not args.emergence_only:
            plot_anchor_layer_grid(concept, args.template, args.top_k, args.highlight_k)


if __name__ == "__main__":
    main()
