"""Modern overview figure of all concept datasets, grouped by domain.

Layout
------
  Top section (Mathematics, 3 × 3 grid of 9 concepts)
  Bottom-left  (Physics, 2 × 2 grid of 4 concepts)
  Bottom-right (Logic / Reasoning, 2 × 2 grid of 4 concepts)

Each card shows the concept name at the top and one representative
pos/neg pair, with the template type rotated across concepts so
symbolic, narrative, and question forms all appear in the figure.

Output: runs/concept_localization/concept_figure.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from experiments.plot_style import MAUVE, NAVY, TEAL, VIOLET, apply

apply()

# ── Palette ───────────────────────────────────────────────────────────────────

MATH_BG = "#F0EBF8"
MATH_ACC = VIOLET  # "#8B7CB8"
PHYS_BG = "#E4F4F2"
PHYS_ACC = TEAL  # "#4EA8A0"
LOGIC_BG = "#E7EEF9"
LOGIC_ACC = NAVY  # "#2B4590"

CARD_BG = "#FFFFFF"
CARD_EDGE = "#DEDEDE"
POS_CLR = "#2A6B52"  # deep teal-green
NEG_CLR = "#8B2020"  # deep burgundy
NAME_CLR = "#1A1A2E"
TMPL_CLR = "#6B7280"

# Monospace character width approximation (inches per char at fontsize 6)
# DejaVu Sans Mono: em ≈ fontsize × (1/72) inches; char_w ≈ 0.6 × em
_MONO_CW6 = 0.6 * 6 / 72  # ≈ 0.0500 inches

# ── Concept registry ──────────────────────────────────────────────────────────

MATH_CONCEPTS = [
    ("carry", "data.concept_datasets.carry_dataset", "generate_carry_pairs"),
    ("gcd", "data.concept_datasets.gcd_dataset", "generate_gcd_pairs"),
    ("residue class", "data.concept_datasets.residue_class_dataset", "generate_residue_pairs"),
    (
        "trans. order.",
        "data.concept_datasets.transitive_ordering_dataset",
        "generate_ordering_pairs",
    ),
    (
        "triangle ineq.",
        "data.concept_datasets.triangle_inequality_dataset",
        "generate_triangle_pairs",
    ),
    (
        "perf. square",
        "data.concept_datasets.perfect_square_dataset",
        "generate_perfect_square_pairs",
    ),
    ("dec. term.", "data.concept_datasets.decimal_termination_dataset", "generate_decimal_pairs"),
    ("geom. series", "data.concept_datasets.geometric_series_dataset", "generate_geometric_pairs"),
    ("dot product", "data.concept_datasets.dot_product_sign_dataset", "generate_dot_pairs"),
]

PHYS_CONCEPTS = [
    ("conservation", "data.concept_datasets.conservation_dataset", "generate_conservation_pairs"),
    ("momentum", "data.concept_datasets.momentum_conservation_dataset", "generate_momentum_pairs"),
    ("doppler", "data.concept_datasets.doppler_shift_dataset", "generate_doppler_pairs"),
    ("wave interf.", "data.concept_datasets.wave_interference_dataset", "generate_wave_pairs"),
]

LOGIC_CONCEPTS = [
    ("causal dir.", "data.concept_datasets.causal_direction_dataset", "generate_causal_pairs"),
    ("neg. scope", "data.concept_datasets.negation_scope_dataset", "generate_negation_pairs"),
    ("syllogism", "data.concept_datasets.syllogism_dataset", "generate_syllogism_pairs"),
    (
        "bal. paren.",
        "data.concept_datasets.balanced_parentheses_dataset",
        "generate_parentheses_pairs",
    ),
]

_TMPL_LABEL = {"T0": "symbolic", "T1": "narrative", "T2": "question"}

# ── Data loading ──────────────────────────────────────────────────────────────

_MAX_LEN = 44  # max characters per example line before truncation


def _pick_pair(concept_idx: int, mod: str, fn: str, seed: int = 42):
    """Return (pos, neg, tmpl_label) for concept at index, rotating template type."""
    m = __import__(mod, fromlist=[fn])
    pairs = getattr(m, fn)(seed=seed)
    templates = sorted(set(p.template for p in pairs))

    for k in range(len(templates)):
        t = templates[(concept_idx + k) % len(templates)]
        candidates = [p for p in pairs if p.template == t]
        candidates.sort(key=lambda p: len(p.prompt_pos))
        p = candidates[0]
        if len(p.prompt_pos.rstrip()) <= _MAX_LEN:
            return p.prompt_pos.rstrip(), p.prompt_neg.rstrip(), _TMPL_LABEL.get(t, t)

    # Fallback: globally shortest pair
    pairs.sort(key=lambda p: len(p.prompt_pos))
    p = pairs[0]
    return (
        p.prompt_pos.rstrip()[:_MAX_LEN],
        p.prompt_neg.rstrip()[:_MAX_LEN],
        _TMPL_LABEL.get(p.template, p.template),
    )


def _load(concepts: list, offset: int = 0, seed: int = 42):
    """Return list of (name, pos, neg, tmpl_label) for every concept."""
    out = []
    for i, (name, mod, fn) in enumerate(concepts):
        pos, neg, lbl = _pick_pair(offset + i, mod, fn, seed)
        out.append((name, pos, neg, lbl))
    return out


# ── Diff helpers ──────────────────────────────────────────────────────────────


def _diff_segments(pos: str, neg: str):
    """Return (pos_segs, neg_segs) where each segs is list of (text, is_diff)."""
    # Find the contiguous diff region
    lo = 0
    while lo < len(pos) and lo < len(neg) and pos[lo] == neg[lo]:
        lo += 1
    hi_p, hi_n = len(pos), len(neg)
    while hi_p > lo and hi_n > lo and pos[hi_p - 1] == neg[hi_n - 1]:
        hi_p -= 1
        hi_n -= 1

    def segs(s, lo, hi):
        result = []
        if lo > 0:
            result.append((s[:lo], False))
        if hi > lo:
            result.append((s[lo:hi], True))
        if hi < len(s):
            result.append((s[hi:], False))
        if not result:
            result.append((s, False))
        return result

    return segs(pos, lo, hi_p), segs(neg, lo, hi_n)


def _draw_seg_line(ax, x0, y, segments, base_color, diff_color, fontsize=6.2):
    """Draw a segmented line at (x0, y) in data units, advancing x per char."""
    cw = _MONO_CW6 * (fontsize / 6)
    x = x0
    for text, is_diff in segments:
        if not text:
            continue
        color = diff_color if is_diff else base_color
        fw = "bold" if is_diff else "normal"
        ax.text(
            x,
            y,
            text,
            fontsize=fontsize,
            color=color,
            fontweight=fw,
            fontfamily="monospace",
            ha="left",
            va="center",
            zorder=4,
            clip_on=True,
        )
        x += len(text) * cw
    return x


# ── Card drawing ──────────────────────────────────────────────────────────────


def _draw_card(ax, x, y, w, h, name, pos, neg, tmpl_label, accent):
    """Draw one concept card. x,y = bottom-left corner; all units = inches."""
    pad = 0.10
    r = 0.06  # corner radius

    # Card background
    bg = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={r}",
        facecolor=CARD_BG,
        edgecolor=CARD_EDGE,
        linewidth=0.7,
        zorder=2,
    )
    ax.add_patch(bg)

    # Concept name — bold, centered, near top
    name_y = y + h - 0.20
    ax.text(
        x + w / 2,
        name_y,
        name,
        fontsize=7.8,
        fontweight="bold",
        color=NAME_CLR,
        ha="center",
        va="center",
        zorder=4,
    )

    # Thin accent rule under name
    rule_y = y + h - 0.36
    ax.plot([x + pad, x + w - pad], [rule_y, rule_y], color=accent, lw=0.9, alpha=0.65, zorder=3)

    # Template type label (tiny, italic, right-aligned)
    ax.text(
        x + w - pad,
        rule_y - 0.05,
        tmpl_label,
        fontsize=4.8,
        color=TMPL_CLR,
        fontstyle="italic",
        ha="right",
        va="top",
        zorder=4,
    )

    # Diff segmentation
    pos_segs, neg_segs = _diff_segments(pos, neg)

    # Positive example
    pos_y = rule_y - 0.22
    _draw_seg_line(
        ax, x + pad, pos_y, pos_segs, base_color=POS_CLR, diff_color=VIOLET, fontsize=6.0
    )

    # Negative example
    neg_y = pos_y - 0.20
    _draw_seg_line(ax, x + pad, neg_y, neg_segs, base_color=NEG_CLR, diff_color=MAUVE, fontsize=6.0)


# ── Section drawing ───────────────────────────────────────────────────────────


def _draw_section(ax, sx, sy, sw, sh, title, data, accent, bg, ncols):
    """Draw a category section containing a grid of cards."""
    # Section background
    section_bg = mpatches.FancyBboxPatch(
        (sx, sy),
        sw,
        sh,
        boxstyle="round,pad=0.08",
        facecolor=bg,
        edgecolor="none",
        linewidth=0,
        zorder=0,
    )
    ax.add_patch(section_bg)

    # Section title
    ax.text(
        sx + 0.20,
        sy + sh - 0.13,
        title.upper(),
        fontsize=7.5,
        fontweight="bold",
        color=accent,
        ha="left",
        va="top",
        zorder=1,
        fontfamily="sans-serif",
        alpha=0.90,
    )

    nrows = (len(data) + ncols - 1) // ncols
    h_pad = 0.18
    v_pad = 0.18
    header = 0.33
    gap = 0.12

    avail_w = sw - 2 * h_pad
    avail_h = sh - header - 2 * v_pad

    cw = (avail_w - (ncols - 1) * gap) / ncols
    ch = (avail_h - (nrows - 1) * gap) / nrows

    for i, (name, pos, neg, lbl) in enumerate(data):
        row = i // ncols
        col = i % ncols
        cx = sx + h_pad + col * (cw + gap)
        # rows fill from top: row 0 is highest
        cy = sy + sh - header - v_pad - (row + 1) * ch - row * gap
        _draw_card(ax, cx, cy, cw, ch, name, pos, neg, lbl, accent)


# ── Main ──────────────────────────────────────────────────────────────────────


def generate_figure(out_path: Path, seed: int = 42) -> None:
    math_data = _load(MATH_CONCEPTS, offset=0, seed=seed)
    phys_data = _load(PHYS_CONCEPTS, offset=9, seed=seed)
    logic_data = _load(LOGIC_CONCEPTS, offset=13, seed=seed)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#F7F7F9")

    # Mathematics: full-width top strip
    _draw_section(
        ax,
        sx=0.20,
        sy=3.95,
        sw=15.60,
        sh=4.85,
        title="Mathematics",
        data=math_data,
        accent=MATH_ACC,
        bg=MATH_BG,
        ncols=3,
    )

    # Physics: bottom-left
    _draw_section(
        ax,
        sx=0.20,
        sy=0.20,
        sw=7.65,
        sh=3.55,
        title="Physics",
        data=phys_data,
        accent=PHYS_ACC,
        bg=PHYS_BG,
        ncols=2,
    )

    # Logic / Reasoning: bottom-right
    _draw_section(
        ax,
        sx=8.15,
        sy=0.20,
        sw=7.65,
        sh=3.55,
        title="Logic / Reasoning",
        data=logic_data,
        accent=LOGIC_ACC,
        bg=LOGIC_BG,
        ncols=2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="runs/concept_localization")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = _REPO_ROOT / args.out_dir / "concept_figure.pdf"
    generate_figure(out, seed=args.seed)
