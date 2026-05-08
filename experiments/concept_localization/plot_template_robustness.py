"""Template-robustness norm-trajectory plots.

Two outputs:
  1. concept_comparison_templates.pdf  — combined plot: one line per concept,
     mean across templates as the line, ±1 std as a shaded band.
  2. concept_per_template.pdf          — small-multiples: one subplot per
     concept, individual T0/T1/T2 lines + mean, so deviations are visible.

Loads deltas.pt which stores {key: {layer: tensor}} for keys
"all", "T0", "T1", "T2".
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.plot_style import (
    GRAY,
    MAUVE,
    NAVY,
    PHASE_BOUNDS,
    PHASE_LABELS,
    TEAL,
    VIOLET,
    apply,
    phase_vlines,
)

apply()

BASE = Path("runs/concept_localization")
N_LAYERS = 36
LAYERS = np.arange(N_LAYERS)
TMPL_KEYS = ["T0", "T1", "T2"]

# ── concept registry ──────────────────────────────────────────────────────────
CONCEPTS = [
    # (key,                  display label,              group,     color,     ls,   lw)
    ("carry", "Carry (mod 10)", "modular", VIOLET, "-", 2.4),
    ("residue_class", "Residue class (mod 7)", "modular", "#C0444A", "-", 2.0),
    ("gcd", "GCD divisibility", "modular", "#D4823A", "-", 1.8),
    ("transitive_ordering", "Transitive ordering", "linear", NAVY, "--", 2.0),
    ("conservation", "Energy conservation", "linear", TEAL, "--", 1.8),
    ("negation_scope", "Negation scope", "linear", MAUVE, "--", 1.8),
    ("causal_direction", "Causal direction", "linear", GRAY, "--", 1.6),
]


# ── helpers ───────────────────────────────────────────────────────────────────
def load_template_norms(subdir: str) -> dict[str, np.ndarray] | None:
    """Return {key: norm_array[N_LAYERS]} for 'all' + template keys."""
    p = BASE / subdir / "deltas.pt"
    if not p.exists():
        return None
    data = torch.load(p, map_location="cpu")
    out: dict[str, np.ndarray] = {}
    for key, layer_dict in data.items():
        arr = np.zeros(N_LAYERS)
        for layer, vec in layer_dict.items():
            arr[int(layer)] = float(vec.float().norm())
        out[key] = arr
    return out


def norm_array(arr: np.ndarray) -> np.ndarray:
    mx = arr.max()
    return arr / mx if mx > 1e-8 else arr


# ── load data ─────────────────────────────────────────────────────────────────
concept_data: dict[str, dict[str, np.ndarray]] = {}
for key, label, group, color, ls, lw in CONCEPTS:
    norms = load_template_norms(key)
    if norms is None:
        print(f"  [warn] {key}: no deltas.pt, skipping")
        continue
    concept_data[key] = norms

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: combined comparison — mean line + std band
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax_raw, ax_norm) = plt.subplots(
    1,
    2,
    figsize=(13, 4.8),
    gridspec_kw={"wspace": 0.18},
)

for key, label, group, color, ls, lw in CONCEPTS:
    if key not in concept_data:
        continue
    norms = concept_data[key]

    # compute mean/std across available templates
    tmpl_arrays = [norms[t] for t in TMPL_KEYS if t in norms]
    if not tmpl_arrays:
        continue
    stack = np.stack(tmpl_arrays)  # (n_templates, N_LAYERS)
    mean_raw = stack.mean(0)
    std_raw = stack.std(0)

    # normalise each template by its own max, then take mean/std
    tmpl_normed = [norm_array(a) for a in tmpl_arrays]
    stack_norm = np.stack(tmpl_normed)
    mean_norm = stack_norm.mean(0)
    std_norm = stack_norm.std(0)

    kw = dict(color=color, linestyle=ls, linewidth=lw, label=label)

    ax_raw.plot(LAYERS, mean_raw, **kw)
    ax_raw.fill_between(
        LAYERS, mean_raw - std_raw, mean_raw + std_raw, color=color, alpha=0.15, linewidth=0
    )

    ax_norm.plot(LAYERS, mean_norm, **kw)
    ax_norm.fill_between(
        LAYERS, mean_norm - std_norm, mean_norm + std_norm, color=color, alpha=0.15, linewidth=0
    )

# residue_class plateau annotation
if "residue_class" in concept_data:
    norms = concept_data["residue_class"]
    tmpl_normed = [norm_array(norms[t]) for t in TMPL_KEYS if t in norms]
    mean_norm = np.stack(tmpl_normed).mean(0)
    ax_norm.annotate(
        "residue class\nplateau L10–L30",
        xy=(16, mean_norm[16]),
        xytext=(22, 0.72),
        fontsize=7.5,
        color="#C0444A",
        ha="left",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#C0444A"),
    )

for ax, title, ylabel in [
    (ax_raw, "Raw δ norm", "‖δ_l‖  (residual-stream delta)"),
    (ax_norm, "Normalised δ norm", "‖δ_l‖ / max  (per concept)"),
]:
    phase_vlines(ax)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(-0.5, 35.5)
    ax.set_xticks(range(0, 36, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color=GRAY, linewidth=0.6, linestyle=":")
    for (lo, hi), name in zip(PHASE_BOUNDS, PHASE_LABELS, strict=False):
        ax.text(
            (lo + hi) / 2,
            1.02,
            name,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#666666",
            style="italic",
            transform=ax.get_xaxis_transform(),
        )

# legend
handles, labels = ax_norm.get_legend_handles_labels()
mod_idx = [
    i
    for i, lbl in enumerate(labels)
    if any(lbl == label and group == "modular" for _, label, group, *_ in CONCEPTS)
]
lin_idx = [
    i
    for i, lbl in enumerate(labels)
    if any(lbl == label and group == "linear" for _, label, group, *_ in CONCEPTS)
]

sep = Line2D([], [], linestyle="none", label="")
legend_handles = (
    [Line2D([], [], linestyle="none", label="Modular arithmetic:")]
    + [handles[i] for i in mod_idx]
    + [sep]
    + [Line2D([], [], linestyle="none", label="Logical / physical:")]
    + [handles[i] for i in lin_idx]
)
legend_labels = (
    ["Modular arithmetic:"]
    + [labels[i] for i in mod_idx]
    + [""]
    + ["Logical / physical:"]
    + [labels[i] for i in lin_idx]
)
ax_norm.legend(
    legend_handles,
    legend_labels,
    fontsize=7.8,
    loc="upper left",
    framealpha=0.92,
    edgecolor="#cccccc",
)

fig.suptitle(
    "Concept localisation: residual-stream delta norm (mean ± std across templates)",
    fontsize=11,
    y=1.02,
)

out1 = BASE / "concept_comparison_templates.pdf"
fig.savefig(out1, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: small multiples — individual templates + mean per concept
# ─────────────────────────────────────────────────────────────────────────────
available = [
    (key, label, group, color, ls, lw)
    for key, label, group, color, ls, lw in CONCEPTS
    if key in concept_data
]

ncols = 4
nrows = int(np.ceil(len(available) / ncols))
fig2, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(ncols * 3.6, nrows * 3.0),
    gridspec_kw={"hspace": 0.55, "wspace": 0.32},
)
axes_flat = axes.flatten() if nrows > 1 else np.atleast_1d(axes).flatten()

TMPL_STYLES = {"T0": ("-", 0.8), "T1": ("--", 0.8), "T2": (":", 0.8)}

for idx, (key, label, group, color, ls, lw) in enumerate(available):
    ax = axes_flat[idx]
    norms = concept_data[key]

    tmpl_arrays = [norms[t] for t in TMPL_KEYS if t in norms]
    tmpl_normed = [norm_array(a) for a in tmpl_arrays]

    # individual templates (faint)
    for t, a_norm in zip([t for t in TMPL_KEYS if t in norms], tmpl_normed, strict=False):
        tls, tlw = TMPL_STYLES[t]
        ax.plot(LAYERS, a_norm, color=color, linestyle=tls, linewidth=tlw, alpha=0.55, label=t)

    # mean (bold)
    mean_norm = np.stack(tmpl_normed).mean(0)
    std_norm = np.stack(tmpl_normed).std(0)
    ax.plot(LAYERS, mean_norm, color=color, linestyle="-", linewidth=2.0, label="mean", zorder=5)
    ax.fill_between(
        LAYERS, mean_norm - std_norm, mean_norm + std_norm, color=color, alpha=0.18, linewidth=0
    )

    ax.set_title(label, fontsize=8.5, pad=3)
    ax.set_xlabel("Layer", fontsize=7.5)
    ax.set_ylabel("‖δ‖ / max", fontsize=7.5)
    ax.set_xlim(-0.5, 35.5)
    ax.set_xticks(range(0, 36, 9))
    ax.set_ylim(-0.05, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    # legend only on first panel
    if idx == 0:
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85, edgecolor="#cccccc")

# hide unused axes
for ax in axes_flat[len(available) :]:
    ax.set_visible(False)

fig2.suptitle(
    "Template robustness: per-template norm trajectories (normalised)",
    fontsize=11,
    y=1.01,
)

out2 = BASE / "concept_per_template.pdf"
fig2.savefig(out2, bbox_inches="tight")
plt.close(fig2)
print(f"Saved {out2}")
