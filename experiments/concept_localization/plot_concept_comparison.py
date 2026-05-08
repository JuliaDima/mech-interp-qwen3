"""Multi-concept norm-trajectory comparison.

Loads deltas.pt from each concept's run directory and plots the residual-stream
delta norm across layers, normalised per concept so shapes are comparable.

Two groups:
  Modular  — carry, gcd, residue_class  (solid lines, warm palette)
  Linear   — transitive_ordering, conservation, causal_direction, negation_scope
             (dashed lines, cool palette)

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
    PHASE_BOUNDS,
    PHASE_LABELS,
    TEAL,
    VIOLET,
    apply,
    phase_vlines,
)

apply()

BASE = Path("runs/concept_localization")
OUT = BASE / "concept_comparison.pdf"

# ── concept registry ──────────────────────────────────────────────────────────
CONCEPTS = {
    # (display_name, subdir, group, color, linestyle, linewidth)
    "carry": ("Carry (mod 10)", "carry", "modular", VIOLET, "-", 2.4),
    "residue_class": (
        "Residue class (mod 7)",
        "residue_class",
        "modular",
        "#C0444A",
        "-",
        2.0,
    ),  # muted red
    "gcd": ("GCD divisibility", "gcd", "modular", "#D4823A", "-", 1.8),  # muted orange
    "transitive_ordering": (
        "Transitive ordering",
        "transitive_ordering",
        "linear",
        NAVY,
        "--",
        2.0,
    ),
    "conservation": ("Energy conservation", "conservation", "linear", TEAL, "--", 1.8),
    "negation_scope": ("Negation scope", "negation_scope", "linear", MAUVE, "--", 1.8),
    "causal_direction": ("Causal direction", "causal_direction", "linear", GRAY, "--", 1.6),
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
    phase_vlines(ax)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(-0.5, 35.5)
    ax.set_xticks(range(0, 36, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color=GRAY, linewidth=0.6, linestyle=":")

    # phase labels on x-axis transform
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

# shared legend in right panel
handles, labels = ax_norm.get_legend_handles_labels()
# split into modular / linear
mod_idx = [
    i
    for i, l in enumerate(labels)
    if any(CONCEPTS[k][0] == l and CONCEPTS[k][2] == "modular" for k in CONCEPTS)
]
lin_idx = [
    i
    for i, l in enumerate(labels)
    if any(CONCEPTS[k][0] == l and CONCEPTS[k][2] == "linear" for k in CONCEPTS)
]

from matplotlib.lines import Line2D

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
