"""Visualisations for the carry concept localisation results.

Produces:
  cross_layer_sim.pdf   — 36×36 heatmap of cos(δ_i, δ_j)
  phase_annotated.pdf   — norm trajectory + template consistency with phase regions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import GRAY, MAUVE, NAVY, RED, TEAL, VIOLET, apply

apply()

OUT_DIR = Path("runs/concept_localization/carry")
RESULTS = OUT_DIR / "results.json"
DELTAS = OUT_DIR / "deltas.pt"

# ── load ──────────────────────────────────────────────────────────────────────
with open(RESULTS) as f:
    res = json.load(f)

deltas_raw = torch.load(DELTAS, map_location="cpu")
norms = np.array([res["sharpness"]["norm_by_layer"][str(l)] for l in range(36)])
layers = np.arange(36)

template_norms = {}
for t in ("T0", "T1", "T2"):
    vecs = torch.stack([deltas_raw[t][l].float() for l in range(36)])
    template_norms[t] = vecs.norm(dim=-1).numpy()

t_cons = res["template_consistency"]
tc_01 = np.array([t_cons[str(l)]["T0_vs_T1"] for l in range(36)])
tc_02 = np.array([t_cons[str(l)]["T0_vs_T2"] for l in range(36)])
tc_12 = np.array([t_cons[str(l)]["T1_vs_T2"] for l in range(36)])

all_vecs = torch.stack([deltas_raw["all"][l].float() for l in range(36)])
all_vecs_n = F.normalize(all_vecs, dim=-1)

# ── Figure 1: cross-layer similarity matrix ───────────────────────────────────
cos_mat = (all_vecs_n @ all_vecs_n.T).numpy()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cos_mat, vmin=0.7, vmax=1.0, cmap="Blues", aspect="auto", origin="upper")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label(r"$\cos(\delta_i,\, \delta_j)$", fontsize=10)
cb.ax.tick_params(labelsize=8)

for boundary in [4.5, 18.5]:
    ax.axhline(boundary, color="white", lw=1.5, ls="--", alpha=0.8)
    ax.axvline(boundary, color="white", lw=1.5, ls="--", alpha=0.8)

ax.set_xlabel("Layer $j$", fontsize=10)
ax.set_ylabel("Layer $i$", fontsize=10)
ax.set_title(r"Cross-layer cosine similarity of carry delta $\delta_l$", pad=8)
tick_pos = [0, 5, 10, 15, 20, 25, 30, 35]
ax.set_xticks(tick_pos)
ax.set_yticks(tick_pos)

for lo, hi, label in [(0, 4, "I"), (5, 18, "II"), (19, 35, "III")]:
    mid = (lo + hi) / 2
    ax.text(
        mid,
        mid,
        label,
        ha="center",
        va="center",
        fontsize=12,
        color="white",
        fontweight="bold",
        alpha=0.85,
    )

ax.grid(False)
fig.savefig(OUT_DIR / "cross_layer_sim.pdf")
plt.close(fig)
print("Saved cross_layer_sim.pdf")

# ── Figure 2: phase-annotated dual panel ──────────────────────────────────────
fig, (ax_norm, ax_tc) = plt.subplots(
    2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"hspace": 0.06, "height_ratios": [3, 2]}
)

for ax in (ax_norm, ax_tc):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# peak-normalised counterparts (shape-preserving, peak = 1)
norms_n = norms / norms.max()
template_norms_n = {t: v / v.max() for t, v in template_norms.items()}

# top panel: raw norm trajectory (left axis) + peak-normalised (right axis)
ax_norm.plot(layers, norms, color=VIOLET, lw=2.2, label="All templates (pooled)", zorder=3)
for t, col in zip(("T0", "T1", "T2"), (NAVY, TEAL, MAUVE)):
    ax_norm.plot(layers, template_norms[t], color=col, lw=1.1, ls="--", alpha=0.7, label=t, zorder=2)

peak_l = int(res["sharpness"]["peak_layer"])
ax_norm.scatter([peak_l], [norms[peak_l]], color=RED, zorder=5, s=55, label=f"Peak (L{peak_l})")
ax_norm.annotate(
    f"L{peak_l},  $\\psi={res['sharpness']['sharpness_index']:.3f}$",
    xy=(peak_l, norms[peak_l]),
    xytext=(peak_l - 9, norms[peak_l] - 6),
    arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9),
    fontsize=8.5,
    color=GRAY,
)

ax_norm.set_ylabel(r"$\|\delta_l\|$  (raw)", fontsize=11)
ax_norm.legend(fontsize=8, loc="upper left", ncol=2)
ax_norm.set_ylim(bottom=0)

# right axis: peak-normalised (same curves, [0,1] scale)
ax_n2 = ax_norm.twinx()
ax_n2.plot(layers, norms_n, color=VIOLET, lw=1.0, ls=":", alpha=0.4, zorder=1)
for t, col in zip(("T0", "T1", "T2"), (NAVY, TEAL, MAUVE)):
    ax_n2.plot(layers, template_norms_n[t], color=col, lw=0.7, ls=":", alpha=0.3, zorder=1)
ax_n2.set_ylabel(r"$\|\delta_l\| / \max$  (normalised)", fontsize=9, color=GRAY)
ax_n2.tick_params(axis="y", labelcolor=GRAY, labelsize=7)
ax_n2.set_ylim(bottom=0)
ax_n2.spines["top"].set_visible(False)

# bottom panel: template consistency
ax_tc.plot(layers, tc_01, color=NAVY, lw=1.8, label="T0 vs T1")
ax_tc.plot(layers, tc_02, color=TEAL, lw=1.8, label="T0 vs T2")
ax_tc.plot(layers, tc_12, color=MAUVE, lw=1.8, label="T1 vs T2")
ax_tc.axhline(1.0, color=GRAY, lw=0.7, ls="--", alpha=0.5)

ax_tc.text(30, 0.80, "late-layer\ndivergence", ha="center", fontsize=8, color=RED, style="italic")

ax_tc.set_ylabel("Template\nconsistency", fontsize=10)
ax_tc.set_xlabel("Layer", fontsize=10)
ax_tc.set_ylim(0.74, 1.03)
ax_tc.legend(fontsize=8, loc="lower left")

fig.suptitle("Carry concept localisation in Qwen3-4B", fontsize=13, y=1.01)
fig.savefig(OUT_DIR / "phase_annotated.pdf")
plt.close(fig)
print("Saved phase_annotated.pdf")
