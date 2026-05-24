"""Multi-concept norm-trajectory comparison.

Loads deltas.pt from each concept's run directory and plots the residual-stream
delta norm across layers, normalised per concept so shapes are comparable.

Three groups:
  Modular          — carry, residue_class, gcd, perfect_square, decimal_termination
                     (solid lines, warm palette)
  Logical / state  — transitive_ordering, negation_scope, causal_direction,
                     balanced_parentheses, syllogism  (dashed lines, cool palette)
  Physical/linear  — conservation, momentum_conservation, doppler_shift,
                     wave_interference, geometric_series, triangle_inequality,
                     dot_product_sign  (dotted lines, teal palette)

Output: runs/concept_localization/concept_comparison.pdf
"""

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.plot_style import (
    GRAY,
    MAUVE,
    NAVY,
    TEAL,
    VIOLET,
    apply,
)

apply()

BASE = Path("runs/concept_localization")
OUT = BASE / "concept_comparison.pdf"

# ── concept registry ──────────────────────────────────────────────────────────
CONCEPTS = {
    # key: (display_name, subdir, group, color, linestyle, linewidth)
    # — modular arithmetic —
    "carry":               ("Carry (mod 10)",        "carry",               "modular",  VIOLET,    "-",  2.2),
    "residue_class":       ("Residue class",          "residue_class",       "modular",  "#C0444A", "-",  2.0),
    "gcd":                 ("GCD divisibility",       "gcd",                 "modular",  "#D4823A", "-",  2.0),
    "perfect_square":      ("Perfect square",         "perfect_square",      "modular",  "#9B59B6", "-",  2.0),
    "decimal_termination": ("Decimal termination",    "decimal_termination", "modular",  "#E67E22", "-",  2.0),
    # — logical / state-like —
    "transitive_ordering": ("Transitive ordering",    "transitive_ordering", "logical",  NAVY,      "--", 2.0),
    "negation_scope":      ("Negation scope",         "negation_scope",      "logical",  MAUVE,     "--", 2.0),
    "causal_direction":    ("Causal direction",       "causal_direction",    "logical",  GRAY,      "--", 2.0),
    "balanced_parentheses":("Balanced parentheses",   "balanced_parentheses","logical",  "#1A7A6E", "--", 2.0),
    "syllogism":           ("Syllogism",              "syllogism",           "logical",  "#6C5B8E", "--", 2.0),
    # — physical / linear —
    "conservation":        ("Energy conservation",    "conservation",        "physical", TEAL,      ":",  2.0),
    "momentum_conservation":("Momentum conservation","momentum_conservation","physical", "#2E86AB", ":",  2.0),
    "doppler_shift":       ("Doppler shift",          "doppler_shift",       "physical", "#3B7A57", ":",  2.0),
    "wave_interference":   ("Wave interference",      "wave_interference",   "physical", "#A0522D", ":",  2.0),
    "geometric_series":    ("Geometric series",       "geometric_series",    "physical", "#8B6914", ":",  2.0),
    "triangle_inequality": ("Triangle inequality",    "triangle_inequality", "physical", "#5C7A3E", ":",  2.0),
    "dot_product_sign":    ("Dot product sign",       "dot_product_sign",    "physical", "#7B5EA7", ":",  2.0),
}

N_LAYERS = 36


def load_norms(subdir: str) -> np.ndarray | None:
    p = BASE / subdir / "results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    norms_dict = d["sharpness"]["norm_by_layer"]
    norms = np.array([float(norms_dict.get(str(l), 0.0)) for l in range(N_LAYERS)])
    return norms


# ── load & normalise ──────────────────────────────────────────────────────────
all_norms: dict[str, np.ndarray] = {}
for key, (label, subdir, group, color, ls, lw) in CONCEPTS.items():
    norms = load_norms(subdir)
    if norms is not None:
        all_norms[key] = norms
    else:
        print(f"  [warn] {subdir}: no results.json, skipping")

layers = np.arange(N_LAYERS)

# ── figure ───────────────────────────────────────────────────────────────────
fig, (ax_raw, ax_norm) = plt.subplots(
    1,
    2,
    figsize=(13, 4.8),
    gridspec_kw={"wspace": 0.18},
)

for key, (label, subdir, group, color, ls, lw) in CONCEPTS.items():
    if key not in all_norms:
        continue
    norms = all_norms[key]
    mx = norms.max()
    if mx < 1e-6:
        continue
    norm_norms = norms / mx

    kw = dict(color=color, linestyle=ls, linewidth=lw, label=label)
    ax_raw.plot(layers, norms, **kw)
    ax_norm.plot(layers, norm_norms, **kw)

# highlight residue_class plateau on normalised panel
if "residue_class" in all_norms:
    n = all_norms["residue_class"]
    ax_norm.annotate(
        "residue class\nplateau L10–L30",
        xy=(16, n[16] / n.max()),
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
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(-0.5, 35.5)
    ax.set_xticks(range(0, 36, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color=GRAY, linewidth=0.6, linestyle=":")

from matplotlib.lines import Line2D

handles, labels = ax_norm.get_legend_handles_labels()

def _idx_for_group(g):
    return [i for i, lbl in enumerate(labels)
            if any(CONCEPTS[k][0] == lbl and CONCEPTS[k][2] == g for k in CONCEPTS)]

mod_idx = _idx_for_group("modular")
log_idx = _idx_for_group("logical")
phy_idx = _idx_for_group("physical")
sep = Line2D([], [], linestyle="none", label="")

legend_handles = (
    [Line2D([], [], linestyle="none", label="Modular arithmetic:")]
    + [handles[i] for i in mod_idx]
    + [sep]
    + [Line2D([], [], linestyle="none", label="Logical / state:")]
    + [handles[i] for i in log_idx]
    + [sep]
    + [Line2D([], [], linestyle="none", label="Physical / linear:")]
    + [handles[i] for i in phy_idx]
)
legend_labels = (
    ["Modular arithmetic:"] + [labels[i] for i in mod_idx]
    + [""]
    + ["Logical / state:"] + [labels[i] for i in log_idx]
    + [""]
    + ["Physical / linear:"] + [labels[i] for i in phy_idx]
)

ax_norm.legend(
    legend_handles,
    legend_labels,
    fontsize=7.0,
    loc="upper left",
    framealpha=0.92,
    edgecolor="#cccccc",
)

fig.suptitle(
    "Concept localisation: residual-stream delta norm across layers",
    fontsize=11,
    y=1.02,
)

fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT}")

# ── text summary ──────────────────────────────────────────────────────────────
print("\nPeak layers and normalised peak norm:")
for key, (label, subdir, group, color, ls, lw) in CONCEPTS.items():
    if key not in all_norms:
        continue
    norms = all_norms[key]
    peak_l = int(norms.argmax())
    print(f"  {label:30s}  peak=L{peak_l:>2}  group={group}")
