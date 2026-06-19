"""Anchor sensitivity analysis across all concepts.

Produces:
  runs/concept_localization/causal_vs_signal_scatter.png
      z-score (null separation) vs causal patching sum, with concept labels
      and group colouring — showing which concepts are both structurally
      localised and causally effective.

  runs/concept_localization/{concept}/anchor_sensitivity.png  (per concept)
      Per-concept anchor sensitivity figure with top-2 anchors, phase
      boundaries, act-normalised trajectory, and null band.

Usage
-----
    python -m experiments.concept_localization.plots.plot_anchor_analysis
"""

from __future__ import annotations

import json
import os
import re
import string as _string_module
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from experiments.plot_style import (
    GRAY, MAUVE, NAVY, TEAL, VIOLET, RED, apply,
)

apply()

BASE = Path("runs/concept_localization")
PHASES_JSON = BASE / "phases.json"
N_LAYERS = 36
LAYERS = np.arange(N_LAYERS)


def load_phases() -> dict:
    """Load hard-coded phase boundaries from phases.json."""
    if not PHASES_JSON.exists():
        return {}
    with open(PHASES_JSON) as f:
        return json.load(f)


def _template_string(concept: str, template_key: str | None) -> str | None:
    """Look up the actual prompt template string for a concept/template key."""
    if template_key is None:
        return None
    _module_map = {
        "carry":                  "experiments.concept_localization.concept_datasets.carry_dataset",
        "gcd":                    "experiments.concept_localization.concept_datasets.gcd_dataset",
        "residue_class":          "experiments.concept_localization.concept_datasets.residue_class_dataset",
        "perfect_square":         "experiments.concept_localization.concept_datasets.perfect_square_dataset",
        "decimal_termination":    "experiments.concept_localization.concept_datasets.decimal_termination_dataset",
        "dot_product_sign":       "experiments.concept_localization.concept_datasets.dot_product_sign_dataset",
        "conservation":           "experiments.concept_localization.concept_datasets.conservation_dataset",
        "momentum_conservation":  "experiments.concept_localization.concept_datasets.momentum_conservation_dataset",
        "doppler_shift":          "experiments.concept_localization.concept_datasets.doppler_shift_dataset",
        "wave_interference":      "experiments.concept_localization.concept_datasets.wave_interference_dataset",
        "geometric_series":       "experiments.concept_localization.concept_datasets.geometric_series_dataset",
        "triangle_inequality":    "experiments.concept_localization.concept_datasets.triangle_inequality_dataset",
        "balanced_parentheses":   "experiments.concept_localization.concept_datasets.balanced_parentheses_dataset",
        "negation_scope":         "experiments.concept_localization.concept_datasets.negation_scope_dataset",
        "syllogism":              "experiments.concept_localization.concept_datasets.syllogism_dataset",
        "transitive_ordering":    "experiments.concept_localization.concept_datasets.transitive_ordering_dataset",
        "causal_direction":       "experiments.concept_localization.concept_datasets.causal_direction_dataset",
    }
    mod_name = _module_map.get(concept)
    if mod_name is None:
        return None
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        templates = getattr(mod, "TEMPLATES", {})
        entry = templates.get(template_key)
        if entry is None:
            return None
        # TEMPLATES values are (pos_template_str, neg_template_str) tuples
        return entry[0] if isinstance(entry, (tuple, list)) else str(entry)
    except Exception:
        return None


def get_phases(concept: str, template: str | None, traj: np.ndarray) -> list[int]:
    """Return phase boundaries for concept/template from phases.json.

    Falls back to auto-detection if the entry is absent.
    """
    db = load_phases()
    t = template or "T0"
    if concept in db and t in db[concept]:
        return [int(x) for x in db[concept][t]]
    return detect_phases(traj)

# ── Concept registry ──────────────────────────────────────────────────────────
# (display_name, group, color)
CONCEPT_META = {
    "carry":                 ("Carry",                "modular",  VIOLET),
    "gcd":                   ("GCD",                  "modular",  "#D4823A"),
    "residue_class":         ("Residue class",        "modular",  RED),
    "perfect_square":        ("Perfect square",       "modular",  "#9B59B6"),
    "decimal_termination":   ("Decimal termination",  "modular",  "#E67E22"),
    "dot_product_sign":      ("Dot-product sign",     "modular",  "#7B5EA7"),
    "conservation":          ("Energy conservation",  "physical", TEAL),
    "momentum_conservation": ("Momentum cons.",       "physical", "#2E86AB"),
    "doppler_shift":         ("Doppler shift",        "physical", "#3B7A57"),
    "wave_interference":     ("Wave interference",    "physical", "#A0522D"),
    "geometric_series":      ("Geometric series",     "physical", "#8B6914"),
    "triangle_inequality":   ("Triangle inequality",  "physical", "#5C7A3E"),
    "balanced_parentheses":  ("Balanced parens",      "logical",  "#1A7A6E"),
    "negation_scope":        ("Negation scope",       "logical",  MAUVE),
    "syllogism":             ("Syllogism",            "logical",  "#6C5B8E"),
    "transitive_ordering":   ("Transitive ordering",  "logical",  NAVY),
    "causal_direction":      ("Causal direction",     "logical",  GRAY),
}

GROUP_COLORS = {"modular": VIOLET, "physical": TEAL, "logical": NAVY}
GROUP_LABELS = {"modular": "Modular / arithmetic", "physical": "Physical / linear", "logical": "Logical / linguistic"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_phases(normed: np.ndarray, min_frac: float = 0.28, merge_gap: int = 3) -> list[int]:
    n = len(normed)
    grad = np.diff(normed)
    if grad.max() < 1e-10:
        return []
    threshold = grad.max() * min_frac
    peaks = [
        i for i in range(1, len(grad) - 1)
        if grad[i] >= grad[i - 1] and grad[i] > grad[i + 1] and grad[i] > threshold
        and i < n - 4  # ignore last 3 layers — readout preparation, not concept encoding
    ]
    if not peaks:
        return []
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] <= merge_gap:
            merged[-1] = (merged[-1] + p) // 2
        else:
            merged.append(p)
    return merged


def load_emergence(concept: str) -> dict | None:
    p = BASE / concept / "emergence.npy"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True).item()


def load_null(concept: str) -> dict | None:
    p = BASE / concept / "null" / "null_permutation.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    # Expose maxnorm fields under canonical keys so callers can use them directly.
    # Falls back to activation-normalised fields for older JSON files without maxnorm.
    if "real_norms_maxnorm" not in d:
        d["real_norms_maxnorm"] = d["real_norms"]
        d["null_norms_maxnorm"] = d["null_norms"]
    return d


def load_causal(concept: str) -> dict | None:
    p = BASE / concept / "results.json"
    if not p.exists():
        return None
    with open(p) as f:
        r = json.load(f)
    if "causal" not in r or "all" not in r["causal"]:
        return None
    pm = r["causal"]["all"]["patching_mean"]
    arr = np.array([float(pm.get(str(l), 0.0)) for l in range(N_LAYERS)])
    return {"patching_mean": arr}


def top_k_anchors(
    emergence: dict,
    concept: str,
    k: int = 2,
) -> list[tuple[int, np.ndarray, str]]:
    """Return the top-k anchor entries (idx, normalised_trajectory, token_label).

    All anchors ranked by early-weighted abruptness: largest weighted single-step
    jump in the double-normalised trajectory (delta/act_norm then /max), with
    exponential decay lambda=3 so early transitions dominate.
    Each trajectory is normalised by its own max to live in [0, 1].
    """
    norms_raw = emergence["norms_raw"]
    tok_labels = emergence.get("token_labels_pos", [])
    active = [a for a in range(norms_raw.shape[0]) if norms_raw[a].max() > 1e-8]
    if not active:
        return []

    # Primary: early-weighted abruptness on the double-normalised trajectory
    # (delta / act_norm, then / its own max). Exponential decay with lambda=3
    # over normalised layer position weights early transitions ~20x over late ones.
    act_norms_raw = emergence.get("act_norms_raw")
    _DECAY = 3.0

    def _abruptness(a: int) -> float:
        if act_norms_raw is not None and act_norms_raw.shape[0] > a:
            row = act_norms_raw[a]
        else:
            row = norms_raw[a]
        m = row.max()
        if m < 1e-8:
            return 0.0
        traj = row / m
        w = 2
        diffs = traj[w:] - traj[:-w]
        weights = np.exp(-_DECAY * np.arange(len(diffs)) / len(diffs))
        return float((diffs * weights).max())

    pri_idx = max(active, key=_abruptness)
    pri_row = norms_raw[pri_idx]
    pri_traj = pri_row / pri_row.max()
    pri_label = tok_labels[pri_idx] if pri_idx < len(tok_labels) else str(pri_idx)
    result = [(pri_idx, pri_traj, pri_label)]

    if k == 1:
        return result

    # Remaining: also rank by abruptness (same criterion, no cosine similarity)
    rest = sorted(
        [a for a in active if a != pri_idx],
        key=_abruptness,
        reverse=True,
    )
    for a in rest[: k - 1]:
        row = norms_raw[a]
        traj = row / row.max()
        label = tok_labels[a] if a < len(tok_labels) else str(a)
        result.append((a, traj, label))
    return result



def plot_causal_vs_signal() -> None:
    """Single-panel scatter: null z-score (symlog) vs causal effect.

    Bubble size encodes phase count.  Labels are placed with an iterative
    repulsion algorithm in display coordinates and connected to their data
    point with a thin line when displaced beyond a threshold.
    """
    group_order = ["modular", "physical", "logical"]
    group_markers = {"modular": "o", "physical": "s", "logical": "^"}

    # ── Collect per-concept data ──────────────────────────────────────────────
    rows = []  # (z, causal_sum, n_phases, col, marker, name, group)
    for c, (name, group, col) in CONCEPT_META.items():
        null = load_null(c)
        if null is None:
            continue
        z = null["z_score"]
        em = load_emergence(c)
        top1 = top_k_anchors(em, c, k=1) if em is not None else []
        traj = top1[0][1] if top1 else np.array(null["real_norms_maxnorm"])
        template = em.get("template") if em else None
        n_phases = len(get_phases(c, template, traj)) + 1
        causal = load_causal(c)
        pm = float(causal["patching_mean"][causal["patching_mean"] > 0].sum()) if causal is not None else 0.0
        rows.append((z, pm, n_phases, col, group_markers[group], name, group))

    xs      = np.array([r[0] for r in rows])
    ys      = np.array([r[1] for r in rows])
    n_ph    = [r[2] for r in rows]
    cols    = [r[3] for r in rows]
    markers = [r[4] for r in rows]
    names   = [r[5] for r in rows]
    groups  = [r[6] for r in rows]

    MARKER_SIZE = 120   # uniform for all concepts

    # ── Figure ────────────────────────────────────────────────────────────────
    apply()
    fig, ax = plt.subplots(figsize=(11, 7))

    for x, y, c, m, grp in zip(xs, ys, cols, markers, groups):
        ax.scatter([x], [y], c=[c], s=[MARKER_SIZE], marker=m,
                   zorder=4, edgecolors="white", linewidths=0.7, alpha=0.92)

    # Phase count as white number stamped inside each marker
    for x, y, n in zip(xs, ys, n_ph):
        ax.text(x, y, str(n), fontsize=6.5, ha="center", va="center",
                fontweight="bold", color="white", zorder=5)

    ax.axvline(0, color=RED, lw=1.0, ls="--", alpha=0.55, zorder=2)
    ax.axhline(0, color=GRAY, lw=0.5, ls=":", alpha=0.35, zorder=1)

    ax.set_xscale("symlog", linthresh=5)
    ax.set_xlabel("Null separation (z-score, symlog)", fontsize=10)
    ax.set_ylabel("Causal patching effect  (sum of positive layers)", fontsize=10)
    ax.set_title("Null separation vs causal localisation", fontsize=12, pad=10)
    ax.tick_params(labelsize=8.5)

    # ── Label repulsion in display coords ────────────────────────────────────
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_ax  = ax.get_window_extent(renderer=renderer)
    W, H     = bbox_ax.width, bbox_ax.height

    pts_d = ax.transData.transform(np.column_stack([xs, ys]))  # (N,2) pixels

    def _to_n(p):
        return (p[0] - bbox_ax.x0) / W, (p[1] - bbox_ax.y0) / H

    def _from_n(n):
        return n[0] * W + bbox_ax.x0, n[1] * H + bbox_ax.y0

    pts_n = np.array([_to_n(p) for p in pts_d])  # (N,2) in [0,1]

    # Initial label positions: radial offset, equally spaced angles
    angles = np.linspace(0, 2 * np.pi, len(rows), endpoint=False)
    lbl_n  = pts_n.copy()
    lbl_n[:, 0] += 0.08 * np.cos(angles)
    lbl_n[:, 1] += 0.08 * np.sin(angles)

    # Iterative repulsion + label–point repulsion (800 steps)
    min_lbl_lbl = 0.090   # min label–label separation
    min_lbl_pt  = 0.048   # min label–any data point separation
    attract     = 0.016   # spring constant toward own data point
    bnd_margin  = 0.04    # keep labels inside this margin from axes edge
    step        = 0.32
    for _ in range(800):
        for i in range(len(lbl_n)):
            fx = fy = 0.0
            # Repel from every other label
            for j in range(len(lbl_n)):
                if i == j:
                    continue
                dx = lbl_n[i, 0] - lbl_n[j, 0]
                dy = lbl_n[i, 1] - lbl_n[j, 1]
                d  = max((dx ** 2 + dy ** 2) ** 0.5, 1e-9)
                if d < min_lbl_lbl:
                    f   = (min_lbl_lbl - d) / d
                    fx += dx * f * 0.55
                    fy += dy * f * 0.55
            # Repel from ALL data points (prevents labels sitting on markers)
            for j in range(len(pts_n)):
                dx = lbl_n[i, 0] - pts_n[j, 0]
                dy = lbl_n[i, 1] - pts_n[j, 1]
                d  = max((dx ** 2 + dy ** 2) ** 0.5, 1e-9)
                if d < min_lbl_pt:
                    f   = (min_lbl_pt - d) / d
                    fx += dx * f * 0.9
                    fy += dy * f * 0.9
            # Attract to own data point
            fx += attract * (pts_n[i, 0] - lbl_n[i, 0])
            fy += attract * (pts_n[i, 1] - lbl_n[i, 1])
            # Boundary repulsion
            for val, lo, hi, sign in [
                (lbl_n[i, 0], bnd_margin, 1 - bnd_margin, 0),
                (lbl_n[i, 1], bnd_margin, 1 - bnd_margin, 1),
            ]:
                if val < lo:
                    if sign == 0: fx += (lo - val) * 3.0
                    else:         fy += (lo - val) * 3.0
                elif val > hi:
                    if sign == 0: fx -= (val - hi) * 3.0
                    else:         fy -= (val - hi) * 3.0
            lbl_n[i, 0] += fx * step
            lbl_n[i, 1] += fy * step

    # Draw labels and connectors
    for i, (name, col) in enumerate(zip(names, cols)):
        lx_d, ly_d = _from_n(lbl_n[i])
        px_d, py_d = pts_d[i]
        lx, ly = ax.transData.inverted().transform([lx_d, ly_d])
        px, py = ax.transData.inverted().transform([px_d, py_d])
        disp   = ((lbl_n[i, 0] - pts_n[i, 0]) ** 2 + (lbl_n[i, 1] - pts_n[i, 1]) ** 2) ** 0.5
        if disp > 0.018:
            ax.annotate("", xy=(px, py), xytext=(lx, ly),
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=GRAY, alpha=0.55),
                        zorder=3)
        ax.text(lx, ly, name, fontsize=7.2, color=col,
                ha="center", va="center", zorder=5)

    # ── Legends ───────────────────────────────────────────────────────────────
    group_handles = [
        plt.scatter([], [], c=GROUP_COLORS[g], s=80, marker=group_markers[g],
                    edgecolors="white", linewidths=0.7, label=GROUP_LABELS[g])
        for g in group_order
    ]
    leg1 = ax.legend(handles=group_handles, fontsize=8, loc="upper left",
                     framealpha=0.88, title="Concept group", title_fontsize=8)
    ax.add_artist(leg1)
    # Explain the white number inside each marker
    ax.text(0.98, 0.02, "number inside marker = phase count",
            transform=ax.transAxes, fontsize=7.5, color=GRAY,
            ha="right", va="bottom", style="italic")

    fig.tight_layout()
    out = BASE / "causal_vs_signal_scatter.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")



def _anchor_display_name(pos: int, token_labels: list[str], template_str: str | None) -> str:
    """Return the template variable name (e.g. 'a', 'b') for anchor position pos.

    Joins token_labels into an approximate prompt string, builds a regex from
    template_str with one named group per field, then checks which group the
    token at pos falls inside.  Falls back to repr(token) if matching fails.
    Duplicate field names (e.g. {a}...{a}) get unique regex group names but
    map back to the same original name.
    """
    tok = token_labels[pos] if pos < len(token_labels) else str(pos)
    if not template_str:
        return repr(tok)

    approx = "".join(token_labels)

    pattern = ""
    field_map: list[tuple[str, str]] = []
    name_counts: dict[str, int] = {}
    for literal, field_name, _, _ in _string_module.Formatter().parse(template_str):
        pattern += re.escape(literal)
        if field_name:
            n = name_counts.get(field_name, 0)
            group = field_name if n == 0 else f"{field_name}_{n}"
            name_counts[field_name] = n + 1
            pattern += f"(?P<{group}>.*?)"
            field_map.append((group, field_name))

    m = re.match(pattern, approx, re.DOTALL)
    if not m:
        return repr(tok)

    char_starts: list[int] = []
    c = 0
    for t in token_labels:
        char_starts.append(c)
        c += len(t)

    tok_start = char_starts[pos]
    tok_end = tok_start + len(tok)
    for group, orig in field_map:
        fs, fe = m.span(group)
        if tok_start < fe and tok_end > fs:
            return orig

    return repr(tok)


def _phase_label(phase_bounds: list[int], n_layers: int) -> list[tuple[float, str]]:
    """Return (x_centre, label) for each phase region."""
    edges = [-0.5] + [b + 0.5 for b in phase_bounds] + [n_layers - 0.5]
    labels = [f"Phase {k+1}" for k in range(len(edges) - 1)]
    return [((edges[k] + edges[k+1]) / 2, labels[k]) for k in range(len(labels))]


_PHASE_LINE_COLORS = [NAVY, TEAL, RED, MAUVE, GRAY]


def plot_concept_anchor_sensitivity(concept: str) -> None:
    """Per-concept anchor sensitivity figure.

    All anchor trajectories shown faded; top-2 highlighted (primary in concept
    colour, secondary in gray).  Phase boundaries drawn as coloured dashed
    vertical lines with labels above.  Null band overlaid.
    Saved to runs/concept_localization/{concept}/anchor_sensitivity.png.
    """
    em = load_emergence(concept)
    null = load_null(concept)
    name, group, col = CONCEPT_META[concept]
    out_path = BASE / concept / "anchor_sensitivity.png"

    fig, ax = plt.subplots(figsize=(10, 4.8))

    top2 = top_k_anchors(em, concept, k=2) if em is not None else []
    has_emergence = len(top2) > 0
    top_idxs = {t[0] for t in top2}

    if has_emergence:
        norms_raw = em["norms_raw"]
        template = em.get("template", None)
        tok_labels = em.get("token_labels_pos", [])
        tmpl_str = _template_string(concept, template)

        # Faded background anchors
        _plotted_other = False
        for a in range(norms_raw.shape[0]):
            row = norms_raw[a]
            if row.max() < 1e-8 or a in top_idxs:
                continue
            ax.plot(LAYERS, row / row.max(), color=GRAY, lw=0.7, alpha=0.28, zorder=1,
                    label="other anchors" if not _plotted_other else "_nolegend_")
            _plotted_other = True

        # Secondary anchor — light gray with dots
        if len(top2) > 1:
            sec_idx, traj2, tok2 = top2[1]
            sec_name = _anchor_display_name(sec_idx, tok_labels, tmpl_str)
            ax.plot(LAYERS, traj2, color="#BBBBBB", lw=2.0, alpha=0.90, zorder=3,
                    marker="o", markersize=4.5, markerfacecolor="#BBBBBB", markeredgewidth=0,
                    label=f"Rank 2 · pos {sec_idx} ({sec_name})")

        # Primary anchor — concept colour with dots
        pri_idx, traj1, tok1 = top2[0]
        pri_name = _anchor_display_name(pri_idx, tok_labels, tmpl_str)
        ax.plot(LAYERS, traj1, color=col, lw=2.6, zorder=5,
                marker="o", markersize=5.0, markerfacecolor=col, markeredgewidth=0,
                label=f"Rank 1 · pos {pri_idx} ({pri_name})")

        # act_norms_raw[pos, l] = ||δ_l|| / E[||h_l||] (already act-normalised in make_gif.py).
        # Just rescale to [0, 1] by dividing by its own max.
        traj1_act_raw = em["act_norms_raw"][pri_idx]
        act_scale = traj1_act_raw.max()
        traj1_actn = traj1_act_raw / act_scale if act_scale > 1e-8 else traj1_act_raw
        ax.plot(LAYERS, traj1_actn, color=TEAL, lw=1.8, ls="--", zorder=4, alpha=0.85,
                marker="o", markersize=3.5, markerfacecolor=TEAL, markeredgewidth=0,
                label=r"$(|\delta_l|/\|h_l\|)\,/\,\max_l(|\delta_l|/\|h_l\|)$")

        phases = get_phases(concept, template, traj1)

    else:
        if null is None:
            plt.close(fig)
            return
        # Use maxnorm trajectory as the primary line
        traj1 = np.array(null["real_norms_maxnorm"])
        ax.plot(LAYERS, traj1, color=col, lw=2.2, zorder=5, label="delimiter anchor")
        # real_norms is activation-normalised — plot as dashed TEAL second line
        traj1_actn = np.array(null["real_norms"])
        act_scale = traj1_actn.max()
        traj1_actn = traj1_actn / act_scale if act_scale > 1e-8 else traj1_actn
        ax.plot(LAYERS, traj1_actn, color=TEAL, lw=1.8, ls="--", zorder=4, alpha=0.85,
                marker="o", markersize=3.5, markerfacecolor=TEAL, markeredgewidth=0,
                label=r"$(|\delta_l|/\|h_l\|)\,/\,\max_l(|\delta_l|/\|h_l\|)$")
        template = em.get("template") if em else None
        phases = get_phases(concept, template, traj1)

    # Use a blended transform: x in data coords, y in axes coords (0–1).
    # This keeps annotations inside the plot so ylim can stay at [0, 1].
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    # Phase boundary lines + annotations (no fill)
    for i, pb in enumerate(phases):
        lcol = _PHASE_LINE_COLORS[i % len(_PHASE_LINE_COLORS)]
        y_mid = float((traj1[pb] + traj1[min(pb + 1, N_LAYERS - 1)]) / 2)
        ax.axvline(pb, color=lcol, lw=1.2, ls="--", alpha=0.75, zorder=4)
        ax.text(pb + 0.4, 0.76,
                f"L{pb}→{pb+1}\ny={y_mid:.2f}",
                transform=blend, fontsize=7.5, color=lcol, va="bottom", ha="left", zorder=6)

    # Phase region labels near the top (axes-coord y so they never push ylim)
    if phases:
        phase_edges = [-0.5] + [b for b in phases] + [N_LAYERS - 0.5]
        for k in range(len(phase_edges) - 1):
            x_c = (phase_edges[k] + phase_edges[k + 1]) / 2
            ax.text(x_c, 0.97, f"Phase {k + 1}",
                    transform=blend, ha="center", va="top", fontsize=8,
                    color=GRAY, style="italic")

    # Null band
    if null is not None:
        null_arr = np.array(null["null_norms_maxnorm"])
        null_mean = null_arr.mean(0)
        null_lo = np.percentile(null_arr, 5, axis=0)
        null_hi = np.percentile(null_arr, 95, axis=0)
        ax.fill_between(LAYERS, null_lo, null_hi, color=GRAY, alpha=0.13, zorder=0,
                        label="null 5–95%")
        ax.plot(LAYERS, null_mean, color=GRAY, lw=1.0, ls="--", alpha=0.55, zorder=2,
                label="null mean")

        z = null["z_score"]
        zcol = RED if z < 0 else (GRAY if z < 10 else col)
        ax.text(0.02, 0.89, f"z = {z:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color=zcol)

    tmpl_display = _template_string(concept, template) or template
    tmpl_str = f"template: {tmpl_display!r}" if tmpl_display else "all templates"
    ax.set_title(
        f"{name} concept emergence — anchor sensitivity  [{tmpl_str}]",
        fontsize=11, pad=8,
    )
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel(r"$|\delta_l| / \max_l |\delta_l|$", fontsize=10)
    ax.set_xlim(-0.5, 35.5)
    ax.set_xticks(range(0, 36, 2))
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.88)
    ax.tick_params(labelsize=8)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating causal vs signal scatter…")
    plot_causal_vs_signal()

    print("Generating individual per-concept anchor sensitivity figures…")
    for concept in CONCEPT_META:
        em_exists = (BASE / concept / "emergence.npy").exists()
        null_exists = (BASE / concept / "null" / "null_permutation.json").exists()
        if em_exists or null_exists:
            plot_concept_anchor_sensitivity(concept)

    print("All figures saved to", BASE)
