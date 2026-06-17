"""
Compose a 2-row × 3-column figure for the thesis:
  Row 1: null-permutation norm comparison (real vs null band)
  Row 2: inter-layer cosine similarity heatmap C_{l,l'}

Columns (a) carry/rank1_pos5, (b) gcd/rank3_pos6, (c) residue_class/rank6_pos8.
Saved to runs/concept_localization/cosim_patterns_abc.pdf
"""
import pathlib, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif"})
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

REPO = pathlib.Path(__file__).parents[2]
RUN = REPO / "runs/concept_localization"

CASES = [
    ("carry",        "carry_T0",        "anchor_rank1_pos5", "Carry",         r"ones$_b$ position"),
    ("gcd",          "gcd_T0",          "anchor_rank3_pos6", "GCD",           r"ones$_a$ position"),
    ("residue_class","residue_class_T0","anchor_rank6_pos8", "Residue class", "delimiter position"),
]

def load_cosim(anchor_dir: pathlib.Path) -> np.ndarray:
    d = torch.load(anchor_dir / "deltas.pt", map_location="cpu", weights_only=False)
    deltas = d["T0"]
    L = len(deltas)
    mat = torch.stack([deltas[l] for l in range(L)], dim=0).float()
    norms = mat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = mat / norms
    return (normed @ normed.T).numpy()

def load_null(anchor_dir: pathlib.Path):
    data = json.loads((anchor_dir / "null/null_permutation.json").read_text())
    real     = np.array(data["real_norms"])
    null_mat = np.array(data["null_norms"])
    null_mean = null_mat.mean(axis=0)
    null_std  = null_mat.std(axis=0)
    return real, null_mean, null_std

NULL_BAND  = "#aec7e8"
NULL_LINE  = "#4a90d9"
GREEN      = "#2ca02c"
RED_LINE   = "#d62728"

def _plot_null_comparison(ax, layers, real, null_mean, null_std,
                          show_legend: bool = False, ylabel: str | None = None):
    """Shared null-comparison plot used by the thesis figure and the emergence script."""
    from matplotlib.collections import LineCollection
    import matplotlib.patches as mpatches

    layers_arr = np.asarray(layers, dtype=float)
    real_arr   = np.asarray(real,   dtype=float)
    null_hi    = null_mean + null_std
    null_lo    = np.maximum(null_mean - null_std, 0.0)

    ax.fill_between(layers_arr, null_lo, null_hi,
                    color=NULL_BAND, alpha=0.65, zorder=1)
    ax.plot(layers_arr, null_mean, color=NULL_LINE, lw=0.9, ls="--", zorder=2)

    # green/red real line — one segment per pair of adjacent layers
    pts  = np.array([layers_arr, real_arr]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_colors = [
        GREEN if (real_arr[i] > null_hi[i] or real_arr[i + 1] > null_hi[i + 1])
        else RED_LINE
        for i in range(len(layers_arr) - 1)
    ]
    lc = LineCollection(segs, colors=seg_colors, linewidths=1.8, zorder=4)
    ax.add_collection(lc)
    ax.autoscale_view()

    ymax = max(real_arr.max(), null_hi.max()) * 1.08
    ax.set_ylim(0, ymax)
    ax.set_xlim(layers_arr[0] - 0.5, layers_arr[-1] + 0.5)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=9)

    if show_legend:
        handles = [
            mpatches.Patch(color=NULL_BAND, alpha=0.65, label=r"null $\mu\pm1\sigma$"),
            plt.Line2D([0], [0], color=NULL_LINE, lw=0.9, ls="--", label="null mean"),
            plt.Line2D([0], [0], color=GREEN,    lw=1.8, label=r"above null $+1\sigma$"),
            plt.Line2D([0], [0], color=RED_LINE, lw=1.8, label=r"below null $+1\sigma$"),
        ]
        ax.legend(handles=handles, fontsize=6.5, frameon=False,
                  loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

def clean_axes(ax):
    """Remove all spines except left and bottom; no box, no grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.grid(False)

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6.2))
gs = GridSpec(
    2, 4,
    figure=fig,
    width_ratios=[1, 1, 1, 0.045],
    height_ratios=[1, 1.35],
    hspace=0.42,
    wspace=0.28,
)

L = 36
layers = np.arange(L)

NULL_BAND  = "#aec7e8"   # light blue for null band
NULL_LINE  = "#4a90d9"   # medium blue for null mean dashed
REAL_COLOR = "#333333"   # dark grey: real norm always visible
ABOVE_FILL = "#f4a08a"   # light salmon: fill above-null region

for col, (concept, template, anchor, label, anchor_label) in enumerate(CASES):
    anchor_dir = RUN / concept / template / anchor
    cosim = load_cosim(anchor_dir)
    real, null_mean, null_std = load_null(anchor_dir)

    # ── row 0: null comparison ──────────────────────────────────────────────
    ax_null = fig.add_subplot(gs[0, col])
    _plot_null_comparison(
        ax_null, layers, real, null_mean, null_std,
        show_legend=(col == 2),
        ylabel=(r"$\tilde{\rho}_l$" if col == 0 else None),
    )
    ax_null.set_xticks(range(0, L, 5))
    ax_null.tick_params(labelsize=7)
    ax_null.set_title(label, fontsize=10, fontweight="bold", pad=14)
    ax_null.text(0.5, 1.01, anchor_label, transform=ax_null.transAxes,
                 fontsize=7.5, ha="center", va="bottom", color="#555555",
                 style="italic", clip_on=False)
    subfig_label = chr(ord("a") + col)
    ax_null.text(-0.18, 1.05, f"({subfig_label})", transform=ax_null.transAxes,
                 fontsize=11, fontweight="bold", va="top", ha="left")
    clean_axes(ax_null)

    # ── row 1: cosine similarity heatmap ────────────────────────────────────
    ax_cos = fig.add_subplot(gs[1, col])
    im = ax_cos.imshow(
        cosim, origin="lower", aspect="equal",
        cmap="RdBu_r", vmin=-1, vmax=1,
        interpolation="nearest",
    )
    ax_cos.set_xticks(range(0, L, 5))
    ax_cos.set_yticks(range(0, L, 5))
    ax_cos.tick_params(labelsize=7)
    ax_cos.set_xlabel(r"layer $l'$", fontsize=8)
    if col == 0:
        ax_cos.set_ylabel(r"layer $l$", fontsize=8)
    # remove all spines — heatmap has its own pixel border from imshow extent
    for sp in ax_cos.spines.values():
        sp.set_visible(False)

# shared colorbar
cbar_ax = fig.add_subplot(gs[1, 3])
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=mcolors.Normalize(vmin=-1, vmax=1))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(r"$C_{l,\,l'}$", fontsize=9)
cbar.ax.tick_params(labelsize=7)
cbar.outline.set_visible(False)

OUT = RUN / "cosim_patterns_abc.pdf"
fig.savefig(OUT, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT}")
