"""Correlate carry concept localisation with steering vector effectiveness."""

import json
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import (
    GRAY,
    MAUVE,
    NAVY,
    PHASE_BOUNDS,
    PHASE_COLORS,
    PHASE_LABELS,
    TEAL,
    VIOLET,
    apply,
    phase_vlines,
    shade_phases,
)

apply()

CARRY_JSON = Path("runs/concept_localization/carry/results.json")
SVEC_DIR = Path("runs/fsm_router/svecs")
STEER_JSON = Path("runs/fsm_router/steer_results_in.json")
OUT_DIR = Path("runs/fsm_router")

# ── load ──────────────────────────────────────────────────────────────────────
with open(CARRY_JSON) as f:
    cr = json.load(f)
carry_norm = {int(k): v for k, v in cr["sharpness"]["norm_by_layer"].items()}
layers = sorted(carry_norm)
n_layers = len(layers)

sv_norms = {}
for prim in ["addition", "subtraction", "multiplication", "modular"]:
    d = torch.load(SVEC_DIR / f"{prim}.pt", map_location="cpu")
    sv_norms[prim] = {int(l): v.float().norm().item() for l, v in d["svecs"].items()}

with open(STEER_JSON) as f:
    sr = json.load(f)
best_alpha = max(r["alpha"] for r in sr)
delta_full = {r["layer"]: r["delta_full_acc"] for r in sr if r["alpha"] == best_alpha}
delta_first = {r["layer"]: r["delta_first_token_acc"] for r in sr if r["alpha"] == best_alpha}

# ── figure: three-panel ───────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(
    3,
    1,
    figsize=(12, 9.5),
    sharex=True,
    gridspec_kw={"hspace": 0.07},
)


def _decorate(ax):
    shade_phases(ax)
    phase_vlines(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── panel 1: carry delta norm ─────────────────────────────────────────────────
cn = [carry_norm[l] for l in layers]
ax1.fill_between(layers, cn, alpha=0.18, color=VIOLET)
ax1.plot(layers, cn, color=VIOLET, linewidth=2.2, label="Carry $\\|\\delta_l\\|$")
ax1.axvline(
    cr["sharpness"]["peak_layer"],
    color=VIOLET,
    linewidth=1.0,
    linestyle=":",
    alpha=0.8,
    label=f"Carry peak (L{cr['sharpness']['peak_layer']})",
)
_decorate(ax1)
ax1.set_ylabel("$\\|\\delta_l^{\\mathrm{carry}}\\|$", fontsize=10)
ax1.set_title(
    "Carry concept delta norm, steering vector norms, and steering improvement vs layer",
)
ax1.legend(fontsize=9, loc="upper left")

# phase labels on first panel
for (lo, hi), _, name in zip(PHASE_BOUNDS, PHASE_COLORS, PHASE_LABELS, strict=False):
    ax1.text(
        (lo + hi) / 2,
        max(cn) * 0.94,
        name,
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
        style="italic",
    )

# ── panel 2: sv norms per primitive ───────────────────────────────────────────
prim_styles = {
    "addition": (NAVY, "-", 2.0, "Addition (carry-relevant)"),
    "subtraction": (VIOLET, "--", 1.6, "Subtraction (carry-relevant)"),
    "multiplication": (TEAL, "-", 1.3, "Multiplication (no carry)"),
    "modular": (MAUVE, "--", 1.3, "Modular (no carry)"),
}
for prim, (color, ls, lw, label) in prim_styles.items():
    nv = [sv_norms[prim].get(l, np.nan) for l in layers]
    ax2.plot(layers, nv, color=color, linestyle=ls, linewidth=lw, label=label)
_decorate(ax2)
ax2.set_ylabel("$\\|v_{k,l}\\|$", fontsize=10)
ax2.legend(fontsize=8, loc="upper left", ncol=2)

# ── panel 3: steering improvement ────────────────────────────────────────────
df = [delta_full.get(l, 0) for l in layers]
di = [delta_first.get(l, 0) for l in layers]
best_l = max(delta_full, key=delta_full.get)

ax3.fill_between(layers, df, alpha=0.18, color=NAVY)
ax3.plot(layers, df, color=NAVY, linewidth=2.2, label=f"Δ full-answer (α={best_alpha})")
ax3.plot(
    layers, di, color=VIOLET, linewidth=1.6, linestyle="--", label=f"Δ first-token (α={best_alpha})"
)
ax3.axhline(0, color=GRAY, linewidth=0.8)
ax3.axvline(
    best_l,
    color=GRAY,
    linewidth=1.4,
    linestyle=":",
    label=f"Best layer (L={best_l}, +{delta_full[best_l]:.1f} pp)",
)
_decorate(ax3)
ax3.set_ylabel("Δ accuracy (pp)", fontsize=10)
ax3.set_xlabel("Layer", fontsize=10)
ax3.legend(fontsize=9, loc="upper right")
ax3.set_xticks(range(0, n_layers, 2))

fig.savefig(OUT_DIR / "carry_steer_correlation.pdf")
plt.close(fig)
print(f"Saved {OUT_DIR / 'carry_steer_correlation.pdf'}")


# ── stats ─────────────────────────────────────────────────────────────────────
def norm01(x):
    r = x - x.min()
    return r / r.max() if r.max() > 0 else r


carry_arr = np.array([carry_norm[l] for l in layers], dtype=float)
steer_arr = np.array([delta_full.get(l, 0) for l in layers], dtype=float)
r_p, p_p = pearsonr(norm01(carry_arr), norm01(steer_arr))
r_s, p_s = spearmanr(carry_arr, steer_arr)
print(f"Carry norm vs Δfull_acc  Pearson r={r_p:.3f} (p={p_p:.3g})")
print(f"Carry norm vs Δfull_acc  Spearman r={r_s:.3f} (p={p_s:.3g})")
