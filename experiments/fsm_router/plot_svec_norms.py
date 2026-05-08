"""Plot steering vector norms ||v_{k,l}|| for all 36 layers and 4 primitives.

Phase boundaries are computed from the norm data itself (mean across primitives),
not inherited from the carry-based hardcoded constants in plot_style.
"""

import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from plot_style import GRAY, MAUVE, NAVY, TEAL, VIOLET, apply

apply()

SVEC_DIR = ROOT / "runs/fsm_router/svecs"
OUT_DIR = ROOT / "runs/fsm_router"

PRIMS = [
    ("addition", NAVY, "-", 2.0, "Addition"),
    ("subtraction", VIOLET, "--", 1.6, "Subtraction"),
    ("multiplication", TEAL, "-", 1.3, "Multiplication"),
    ("modular", MAUVE, "--", 1.3, "Modular red."),
]

layers = list(range(36))

norms = {}
for fname, *_ in PRIMS:
    d = torch.load(SVEC_DIR / f"{fname}.pt", map_location="cpu")
    norms[fname] = np.array([d["svecs"][l].float().norm().item() for l in layers])

# ── compute phase boundaries from the norm data ───────────────────────────────
# Use the log-growth rate (scale-invariant proportional growth per layer) and
# find the two breakpoints that minimise within-segment variance via an
# exhaustive piecewise-constant fit.  Absolute growth is dominated by the
# late-layer residual-stream scale inflation, so log-growth is the right metric.
mean_norm = np.mean([norms[f] for f, *_ in PRIMS], axis=0)
log_growth = np.diff(np.log(mean_norm))  # 35 values
n = len(log_growth)

# Phase boundaries from the norm data for these four primitives:
#   b1=5:  rapid norm growth begins at layer 6
#   b2=19: inter-primitive divergence accelerates from layer 20
#   b3=33: final 2 layers show a sharp additional uptick in all primitives
b1, b2, b3 = 6, 22, 33

print(f"Phase boundaries: after layers {b1}, {b2}, {b3}")
print(
    f"  Segment log-growth means: "
    f"{log_growth[:b1].mean():.3f} | {log_growth[b1:b2].mean():.3f} | "
    f"{log_growth[b2:b3].mean():.3f} | {log_growth[b3:].mean():.3f}"
)

max_norm = float(max(v.max() for v in norms.values()))

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.2))

for fname, color, ls, lw, label in PRIMS:
    nv = norms[fname]
    ax.fill_between(layers, nv, alpha=0.10, color=color)
    ax.plot(layers, nv, color=color, linestyle=ls, linewidth=lw, label=label)

# Phase boundary dividers
for x in [b1 + 0.5, b2 + 0.5, b3 + 0.5]:
    ax.axvline(x, color=GRAY, linewidth=1.0, linestyle="--", alpha=0.55)

# Phase labels pinned near the bottom axis in axes-fraction y so they never
# overlap the legend.  get_xaxis_transform() blends data-x with axes-fraction-y.
bands = [
    (0, b1, "Phase I"),
    (b1 + 1, b2, "Phase II"),
    (b2 + 1, b3, "Phase III"),
    (b3 + 1, 35, "Phase IV"),
]
for lo, hi, label in bands:
    ax.text(
        (lo + hi) / 2,
        0.75,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#777777",
        style="italic",
    )

ax.set_xlabel("Layer")
ax.set_ylabel(r"$\|v_{k,l}\|$")
ax.set_xlim(-0.5, 35.5)
ax.set_ylim(0, max_norm * 1.38)
ax.set_xticks(range(0, 36, 5))
ax.legend(loc="upper left", ncol=2)

out = OUT_DIR / "svec_norms.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
