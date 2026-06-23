"""Two-panel causal overlay comparison for the thesis.

(a) A carry anchor (ones_b, pos9) where causal patching and grad·δ align with
    the double-normalised trajectory and the null excess.
(b) A carry anchor (operator '+', pos6) where the double-normalised curve has a
    clear peak but causal patching and grad·δ remain near zero — the positional
    signal is not causally relevant for the answer.

Saved to runs/concept_localization/causal_overlay_comparison.pdf.
"""
import json, pathlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif"})

_REPO = pathlib.Path(__file__).resolve().parents[3]
import sys; sys.path.insert(0, str(_REPO))
import experiments.plot_style as ps

RUN = _REPO / "runs/concept_localization"

CASES = [
    ("carry", "carry_T0", "anchor_rank2_pos9",
     r"(a) Carry — ones$_b$ position",
     "Patching and grad·$\delta$ track the double-norm"),
    ("carry", "carry_T0", "anchor_rank3_pos6",
     r"(b) Carry — operator position",
     "Double-norm has a peak; patching and grad·$\delta$ do not"),
]

LAYERS = np.arange(36)


def load_anchor(concept, template, anchor):
    adir = RUN / concept / template / anchor
    res  = json.loads((adir / "results.json").read_text())
    null = json.loads((adir / "null/null_permutation.json").read_text())

    causal      = res["causal"]["all"]
    patch_raw   = np.array(list(causal["patching_mean"].values()))
    grad_raw    = np.array(list(causal["grad_dot_delta_mean"].values()))
    real_dn     = np.array(null["real_norms"])          # double-normalised [0,1]
    null_mat    = np.array(null["null_norms"])
    null_mean   = null_mat.mean(0)
    null_std    = null_mat.std(0)
    return patch_raw, grad_raw, real_dn, null_mean, null_std


def peak_norm(v: np.ndarray) -> np.ndarray:
    peak = np.abs(v).max()
    return v / (peak + 1e-12)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(False)


fig, axes = plt.subplots(1, 2, figsize=(9, 3.4),
                         gridspec_kw={"wspace": 0.38})

for ax, (concept, template, anchor, title, subtitle) in zip(axes, CASES):
    patch_raw, grad_raw, real_dn, _null_mean, _null_std = load_anchor(concept, template, anchor)

    patch_n = peak_norm(patch_raw)
    grad_n  = peak_norm(grad_raw)

    ax.plot(LAYERS, real_dn,  color=ps.NAVY,   lw=2.0, zorder=4,
            label=r"$\tilde{D}_l^{\mathrm{act}}$")
    ax.plot(LAYERS, patch_n,  color=ps.VIOLET, lw=1.8, zorder=3,
            label="patch / peak")
    ax.plot(LAYERS, grad_n,   color=ps.TEAL,   lw=1.8, zorder=3,
            label=r"grad$\cdot\delta$ / peak")
    ax.axhline(0, color=ps.GRAY, lw=0.7, ls=":", zorder=0)

    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(-0.15, 1.12)
    ax.set_xticks(range(0, 36, 5))
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r"layer $l$", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel("normalised value", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes,
            fontsize=7.5, ha="center", va="bottom", color="#555555", style="italic")
    clean_axes(ax)

handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="lower center", ncol=4, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.10))

OUT = RUN / "causal_overlay_comparison.pdf"
fig.savefig(OUT, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT}")
