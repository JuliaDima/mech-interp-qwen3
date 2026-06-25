"""
Shared publication-quality matplotlib style for all thesis figures.

Colour palette (muted, Distill-inspired):
  VIOLET  — light violet   (primary concept / carry signal)
  NAVY    — navy blue      (primary metric / steering)
  TEAL    — muted teal     (secondary / accent)
  MAUVE   — warm mauve     (tertiary line)
  GRAY    — medium gray    (annotations, grid)
  RED     — muted red      (warnings / divergence zones)

Phase shading (very subtle pastels):
  PHASE_COLORS — list of [I, II, III] background fills
"""

import matplotlib.pyplot as plt

# ── Palette ───────────────────────────────────────────────────────────────────
VIOLET = "#8B7CB8"  # light violet  (replaces teal)
NAVY = "#2B4590"  # navy blue     (replaces dark green)
TEAL = "#4EA8A0"  # muted teal    (replaces orange)
MAUVE = "#B07898"  # warm mauve    (quaternary)
GRAY = "#6B7280"  # annotation gray
RED = "#C0444A"  # muted red

# Diverging colormaps adapted for the palette
# For cosine similarity / alignment heatmaps
CMAP_DIV = "RdBu_r"  # still best for signed similarity
CMAP_SEQ = "Blues"

# Phase background fills (very light)
PHASE_COLORS = ["#EDE9F6", "#EAF0FA", "#E6F4F1"]  # violet-tint / blue-tint / teal-tint
PHASE_LABELS = ["Phase I", "Phase II", "Phase III"]
PHASE_BOUNDS = [(0, 4), (5, 18), (19, 35)]

# ── rcParams ─────────────────────────────────────────────────────────────────
RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#DDDDDD",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "lines.linewidth": 1.8,
    "patch.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def apply():
    """Call once at the top of each plot script."""
    plt.rcParams.update(RC)


def shade_phases(ax, orientation="vertical", alpha=0.0, label_y=None, fontsize=8):
    """Mark the three processing phases on ax with vertical dashed lines only.

    Phase background fills are disabled (alpha=0.0) to keep a clean white
    background. Pass label_y to annotate phase names at a specific y position.
    """
    for (lo, hi), color, name in zip(PHASE_BOUNDS, PHASE_COLORS, PHASE_LABELS, strict=False):
        if label_y is not None:
            ax.text(
                (lo + hi) / 2,
                label_y,
                name,
                ha="center",
                va="top",
                fontsize=fontsize,
                color="#777777",
                style="italic",
            )


def phase_vlines(ax, color=None, lw=1.0, ls="--", alpha=0.55):
    """Draw vertical dashed lines at the two phase transition boundaries."""
    if color is None:
        color = GRAY
    for x in [4.5, 18.5]:
        ax.axvline(x, color=color, linewidth=lw, linestyle=ls, alpha=alpha)
