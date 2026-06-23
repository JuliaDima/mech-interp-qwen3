"""
Two-panel figure for the thesis Anchor Selection section.

Each panel shows one anchor position, plotting both the raw delta norm D_l
(scaled to [0,1]) and the double-normalised trajectory D̃_l^act.

The two anchors are chosen to illustrate opposite regimes:
  (a) residue_class pos3 (answer position): D̃ is high from layer 0 and stays
      elevated broadly — the concept is encoded across the network.
  (b) GCD pos10 (operator position): D̃ starts near zero and spikes sharply at
      layer 17 — the concept is computed locally at a single processing stage.

In both cases the raw D_l grows monotonically and cannot distinguish between the
two regimes; only the activation-normalised and peak-rescaled curve reveals the
underlying structure.
"""
import pathlib, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif"})

REPO = pathlib.Path(__file__).parents[3]
RUN  = REPO / "runs/concept_localization"

def load_anchor(concept, template, anchor):
    adir = RUN / concept / template / anchor
    null_data = json.loads((adir / "null/null_permutation.json").read_text())
    # real_norms      = D̃_l^act = D_l^act / max_l D_l^act  (double-normalised)
    # real_norms_maxnorm = raw D_l / max_l(raw D_l)         (peak-scaled raw)
    real_dn   = np.array(null_data["real_norms"])
    raw_norms = np.array(null_data["real_norms_maxnorm"])
    null_mat  = np.array(null_data["null_norms_maxnorm"])
    null_mean = null_mat.mean(axis=0)
    null_lo   = np.percentile(null_mat, 5,  axis=0)
    null_hi   = np.percentile(null_mat, 95, axis=0)
    return raw_norms, real_dn, null_mean, null_lo, null_hi

CASES = [
    ("carry", "carry_T0", "anchor_rank1_pos5",
     r"(a) Carry — ones$_a$ position (pos 5)",
     "High from layer 0, broad encoding"),
    ("gcd",   "gcd_T0",   "anchor_rank6_pos10",
     r"(b) GCD — operator position (pos 10)",
     "Near-zero then sharp spike at layer 17"),
]

RAW_COLOR = "#aaaaaa"
DN_COLOR  = "#2c7bb6"
NULL_BAND = "#f4a08a"
NULL_LINE = "#d7191c"
LAYERS    = np.arange(36)

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(False)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4),
                         gridspec_kw={"wspace": 0.38})

for ax, (concept, template, anchor, title, subtitle) in zip(axes, CASES):
    raw, dn, null_mean, null_lo, null_hi = load_anchor(concept, template, anchor)

    # scale raw to [0,1] so both curves share the y-axis
    raw_scaled = raw / (raw.max() + 1e-8)

    ax.fill_between(LAYERS, null_lo, null_hi,
                    color=NULL_BAND, alpha=0.55, zorder=1, label="null 5–95%")
    ax.plot(LAYERS, null_mean, color=NULL_LINE, lw=0.9, ls="--",
            zorder=2, label="null mean")
    ax.plot(LAYERS, raw_scaled, color=RAW_COLOR, lw=1.5, ls="-",
            zorder=3, label=r"$D_l / \max D_l$ (raw)")
    ax.plot(LAYERS, dn, color=DN_COLOR, lw=2.0, ls="-",
            zorder=4, label=r"$\tilde{D}_l^{\mathrm{act}}$ (double-norm)")

    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.5, 35.5)
    ax.set_xticks(range(0, 36, 5))
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r"layer $l$", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel("normalised value", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes,
            fontsize=7.5, ha="center", va="bottom", color="#555555", style="italic")
    clean_axes(ax)

# shared legend below
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="lower center", ncol=4, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.10))

OUT = RUN / "norm_comparison.pdf"
fig.savefig(OUT, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT}")
