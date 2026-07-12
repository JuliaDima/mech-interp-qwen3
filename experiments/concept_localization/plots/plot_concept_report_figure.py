"""One-page per-concept overview figure for the thesis report.

Recreates, in the report's own plotting style, the three-panel per-anchor
view from the hosted concept-emergence visualiser
(https://mechinterp-viz-94c364.uniofcam.dev/) across *all* anchors of a
concept at once: the prompt with every anchor highlighted, inter-layer
cosine similarity, delta trajectory vs. permutation null, the transcoder
feature constellation, and small activation-profile approximations for each
anchor's top-3 aligned features.

Data comes entirely from the committed lightweight export
(`data/{concept}_T0.concept.json`, the same file the visualiser loads), plus
a hand-curated JSON of feature interpretations (produced by inspecting the
per-feature activation images under `viz/data/images/`) — no RDS/GPU access
required.

Usage
-----
    python -m experiments.concept_localization.plots.plot_concept_report_figure \
        --concept gcd
    python -m experiments.concept_localization.plots.plot_concept_report_figure --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import experiments.plot_style as ps

DATA_DIR = _REPO_ROOT / "data"
INTERP_DIR = _REPO_ROOT / "experiments" / "concept_localization" / "plots" / "feature_interpretations"
VIZ_IMAGES_DIR = _REPO_ROOT / "viz" / "data" / "images"
OUT_DIR = _REPO_ROOT / "Report" / "Pages" / "Figures"

TOP_K = 3
GREEN = "#2E8B57"


def _diverging_color(score: float, vmax: float = 1.0) -> str:
    """Blue (opposing) -> light gray -> red (supporting), same formula the visualiser uses."""
    t = max(0.0, min(1.0, abs(score) / max(vmax, 1e-9)))
    base = np.array([241, 245, 249])
    target = np.array([37, 99, 235]) if score < 0 else np.array([220, 38, 38])
    rgb = base + (target - base) * t
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def load_concept(concept: str) -> dict:
    path = DATA_DIR / f"{concept}_T0.concept.json"
    data = json.loads(path.read_text())
    data["anchors"] = sorted(data["anchors"], key=lambda a: (a["position"], a["rank"]))
    return data


def load_interpretations(concept: str) -> dict[str, str]:
    path = INTERP_DIR / f"{concept}_interpretations.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def top_features(anchor: dict, k: int = TOP_K) -> list[dict]:
    return sorted(anchor["features"], key=lambda f: abs(f.get("score", 0.0)), reverse=True)[:k]


def _badge(ax, x: float, y: float, n: int, *, size: float = 380, fontsize: float = 15,
           color: str = GREEN, transform=None) -> None:
    """A small filled circle with a centered number — used everywhere a unicode
    circled digit would otherwise be needed (those glyphs aren't in the report's font).
    Uses a scatter marker (points-based) rather than a Circle patch so it stays
    perfectly round even inside axes with a non-square aspect ratio."""
    transform = transform or ax.transData
    ax.scatter([x], [y], s=size, color=color, edgecolors="white", linewidths=0.9,
                zorder=6, transform=transform)
    ax.text(x, y, str(n), ha="center", va="center", fontsize=fontsize, color="white",
             fontweight="bold", zorder=7, transform=transform)


# ── Row A: prompt tokens with anchors highlighted ──────────────────────────────

def _draw_token_row(fig, ax, data: dict, anchors: list[dict]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Each anchor's own "token" field (used in the column headers below) is read
    # off whichever example that anchor's statistics happen to be reported from,
    # which need not be the same example as the shared prompt_tokens_pos array —
    # so at anchor positions we display the anchor's own token, not the shared
    # array's, to guarantee this row always agrees with the column headers below.
    anchor_by_pos = {a["position"]: i + 1 for i, a in enumerate(anchors)}
    anchor_at_pos = {a["position"]: a for a in anchors}
    tokens = [anchor_at_pos[i]["token"] if i in anchor_at_pos else tok
              for i, tok in enumerate(data["prompt_tokens_pos"])]

    y = 0.62
    fontsize = 27
    canvas = fig.canvas
    canvas.draw()
    renderer = canvas.get_renderer()
    inv = ax.transAxes.inverted()

    # Pass 1: place tokens left-to-right from x=0, measuring each one's rendered
    # width via the canvas renderer so spacing matches natural text flow exactly.
    texts = []
    x = 0.0
    for i, tok in enumerate(tokens):
        is_anchor = i in anchor_by_pos
        txt = ax.text(x, y, tok, ha="left", va="center", fontsize=fontsize,
                       color=GREEN if is_anchor else "#334155",
                       fontweight="bold" if is_anchor else "normal",
                       family="monospace", transform=ax.transAxes)
        bbox = txt.get_window_extent(renderer=renderer)
        (x0, _), (x1, _) = inv.transform(bbox)
        texts.append(txt)
        x += (x1 - x0)

    # Pass 2: shift every token so the whole line is horizontally centred.
    total_w = x
    shift = (1 - total_w) / 2
    for txt in texts:
        txt.set_x(txt.get_position()[0] + shift)


# ── Column header shared by rows B/C/D/E ────────────────────────────────────────

def _column_header(ax, rank: int, anchor: dict) -> None:
    tok = anchor["token"].replace(" ", "·") or "·"
    _badge(ax, -0.19, 1.16, rank, size=480, fontsize=15, transform=ax.transAxes)
    ax.text(0.5, 1.16, f"'{tok}'", ha="center", va="center",
             fontsize=18, fontweight="bold", transform=ax.transAxes)


# ── Row B: inter-layer cosine similarity ────────────────────────────────────────

def _draw_cosine(ax, anchor: dict, show_axis: bool = False) -> None:
    cosine = np.asarray(anchor["cosine"], dtype=float)
    layers = anchor.get("cosine_layers") or list(range(cosine.shape[0]))
    ax.imshow(cosine, origin="lower", cmap=ps.CMAP_DIV, vmin=-1, vmax=1, aspect="equal")
    if show_axis:
        n = cosine.shape[0]
        ticks = [i for i in range(0, n, 5)] + ([n - 1] if (n - 1) % 5 else [])
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([layers[i] for i in ticks], fontsize=12)
        ax.set_yticklabels([layers[i] for i in ticks], fontsize=12)
        ax.set_xlabel("layer", fontsize=13)
        ax.set_ylabel("layer", fontsize=13)
        ax.tick_params(length=3.5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── Row C: delta trajectory + permutation null ──────────────────────────────────

def _draw_trajectory(ax, anchor: dict, show_axis: bool = False) -> None:
    layers = anchor["layers"]
    raw = np.asarray(anchor["raw_norm"], dtype=float)
    act = np.asarray(anchor["activation_norm"], dtype=float)
    raw_n = raw / max(raw.max(), 1e-12)
    act_n = act / max(act.max(), 1e-12)

    null = anchor.get("null", {})
    mean = np.asarray(null.get("mean", []), dtype=float)
    std = np.asarray(null.get("std", []), dtype=float)
    if mean.size and std.size:
        excess = np.maximum(0.0, act - (mean + std))
        excess_n = excess / max(act.max(), 1e-12)
        ax.fill_between(layers, 0, excess_n, color=ps.MAUVE, alpha=0.40, lw=0)

    ax.plot(layers, raw_n, color=ps.VIOLET, lw=2.2)
    ax.plot(layers, act_n, color=ps.TEAL, lw=1.8, ls=":")

    peak_layer = null.get("peak_layer")
    if peak_layer is not None:
        ax.axvline(peak_layer, color=ps.GRAY, lw=1.0, ls=":", alpha=0.7)

    span = max(layers) - min(layers)
    ax.set_xlim(min(layers) - 0.03 * span, max(layers) + 0.03 * span)
    ax.set_ylim(-0.04, 1.09)
    if show_axis:
        ax.set_xticks(range(0, max(layers) + 1, 5))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(labelsize=12, length=3.5)
        ax.set_xlabel("layer", fontsize=13)
        ax.set_ylabel("normalised signal", fontsize=13)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── Row D: transcoder feature constellation ─────────────────────────────────────

def _draw_constellation(ax, anchor: dict, top6: list[dict], show_axis: bool = False) -> None:
    features = anchor["features"]
    layers = anchor["layers"]
    lo, hi = min(layers), max(layers)

    def lx(layer: int) -> float:
        return lo + (layer - lo)

    feat_map = {f["feature"]: f for f in features}
    connections = [c for c in anchor.get("connections", [])
                    if c["source"] in feat_map and c["target"] in feat_map]

    vmax = max((abs(f.get("score", 0.0)) for f in features), default=1.0) or 1.0

    ax.axhline(0, color="#CBD5E1", lw=0.8, ls="--", zorder=1)

    for c in connections:
        s, t = feat_map[c["source"]], feat_map[c["target"]]
        ax.plot([lx(s["layer"]), lx(t["layer"])], [s["score"], t["score"]],
                 color=(ps.MAUVE if c.get("sign", 1) < 0 else "#4C9A6A"),
                 lw=0.6 + 2.2 * c.get("support_rate", 0.3),
                 alpha=0.30 + 0.55 * c.get("support_rate", 0.3), zorder=2)

    top6_keys = {f["feature"]: i + 1 for i, f in enumerate(top6)}
    for f in features:
        x, y = lx(f["layer"]), f["score"]
        is_top = f["feature"] in top6_keys
        size = 22 + 130 * min(abs(f["score"]) / vmax, 1.0)
        if is_top:
            size = size * 1.55 + 70
        ax.scatter([x], [y], s=size, color=_diverging_color(f["score"], vmax),
                    edgecolors="white", linewidths=1.0, zorder=3 + is_top)

    ax.set_xlim(lo - 2.5, hi + 2.5)
    ax.set_ylim(-1.15, 1.15)
    if show_axis:
        ax.set_xticks(range(0, hi + 1, 5))
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.tick_params(labelsize=12, length=3.5)
        ax.set_xlabel("layer", fontsize=13)
        ax.set_ylabel("combined cosine", fontsize=13)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── Row E: redrawn per-feature activation-profile plots for the top features ───
#
# The lightweight JSON export only carries mean_pos/mean_neg per feature, not
# the full per-bin profile, so the real per-bin shape is recovered by pixel
# analysis of the real activation-profile PNGs (the same images the hosted
# visualiser's click-to-inspect panel shows), then redrawn as a clean vector
# plot at report scale — sharper than embedding the (fairly low-resolution)
# source raster directly, and calibrated back to real units using mean_pos.

_BAR_BLUE = np.array([111, 142, 191])   # rendered (alpha-blended) "#4c72b0"
_BAR_ORANGE = np.array([227, 156, 116])  # rendered (alpha-blended) "#dd8452"
_HEAT_START = np.array([248, 248, 248])  # "#f8f8f8", shared low end of both heat cmaps
_HEAT_POS_END = np.array([43, 69, 144])  # rendered ps.NAVY high end
_HEAT_NEG_END = np.array([192, 57, 43])  # rendered "#c0392b" high end


def _feature_image_path(concept: str, anchor: dict, feature: dict) -> Path | None:
    v = feature.get("visualization")
    if not v or "panel_index" not in v:
        return None
    fname = f"anchor_rank{anchor['rank']}_pos{anchor['position']}_panel_{v['panel_index']}.png"
    return VIZ_IMAGES_DIR / f"{concept}_T0" / fname


def _cluster(values: list[int], gap: int = 5) -> list[list[int]]:
    values = sorted(values)
    groups = [[values[0]]]
    for v in values[1:]:
        if v - groups[-1][-1] <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def _contiguous_height(mask_col: np.ndarray, axis_row: int, max_gap: int = 3) -> int:
    """Bar height in pixels, scanning up from the axis and tolerating brief gaps
    (gridlines showing through the bars' alpha transparency)."""
    h = gap = 0
    row = axis_row - 1
    while row >= 0:
        if mask_col[row]:
            h += 1 + gap
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
        row -= 1
    return h


def _extract_1d_profile(img_path: Path, n_bins: int = 7) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (values, is_pos), each length n_bins, values in pixel-height units."""
    arr = np.asarray(Image.open(img_path).convert("RGB")).astype(int)
    black = np.all(arr < 60, axis=-1)
    axis_row = int(np.argmax(black.sum(axis=1)))
    tick_band = black[axis_row + 1:axis_row + 9, :]
    tick_cols = np.where(tick_band.sum(axis=0) >= 5)[0]
    if tick_cols.size == 0:
        return None
    tick_centers = np.array([np.mean(g) for g in _cluster(list(tick_cols))])
    if len(tick_centers) != n_bins:
        return None
    mask_b = np.linalg.norm(arr - _BAR_BLUE, axis=-1) < 40
    mask_o = np.linalg.norm(arr - _BAR_ORANGE, axis=-1) < 40
    half_w = max(2, int(np.median(np.diff(tick_centers)) * 0.35))
    values = np.zeros(n_bins)
    is_pos = np.zeros(n_bins, dtype=bool)
    for i, cx in enumerate(tick_centers):
        c0, c1 = int(cx - half_w), int(cx + half_w)
        col_b = mask_b[:, c0:c1].any(axis=1)
        col_o = mask_o[:, c0:c1].any(axis=1)
        h_b = _contiguous_height(col_b, axis_row)
        h_o = _contiguous_height(col_o, axis_row)
        values[i], is_pos[i] = (h_b, True) if h_b >= h_o else (h_o, False)
    return values, is_pos


def _extract_2d_profile(img_path: Path, n: int = 10) -> np.ndarray | None:
    """Returns an (n, n) array of rendered RGB colours, one per (b0, a0) grid cell."""
    arr = np.asarray(Image.open(img_path).convert("RGB")).astype(int)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    colorful = (np.abs(r - g) > 6) | (np.abs(g - b) > 6) | (np.abs(r - b) > 6)
    rows = np.where(colorful.any(axis=1))[0]
    cols = np.where(colorful.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    cell_h, cell_w = (r1 - r0 + 1) / n, (c1 - c0 + 1) / n
    grid = np.zeros((n, n, 3))
    for a0 in range(n):
        for b0 in range(n):
            cx = int(c0 + (a0 + 0.5) * cell_w)
            cy = int(r1 - (b0 + 0.5) * cell_h)
            patch = arr[max(cy - 3, 0):cy + 4, max(cx - 3, 0):cx + 4].reshape(-1, 3)
            grid[b0, a0] = np.median(patch, axis=0)
    return grid


def _project_intensity(sample: np.ndarray, end: np.ndarray, start: np.ndarray = _HEAT_START) -> float:
    v = end - start
    t = float(np.dot(sample - start, v) / np.dot(v, v))
    return max(0.0, min(1.0, t))



# The source chart always renders bin 0 ("a mod 7 == 0") in blue regardless of
# concept -- a fixed convention of the original visualiser, not a marker of
# which residue is the contrastive target. For GCD the target residue really
# is 0, so blue happens to line up with mean_pos; for residue_class the
# committed export's target residue is 1 (its example prompt is a=148,
# 148 mod 7 == 1), so calibrating off the blue bin silently used the wrong
# bin's pixel height as a stand-in for mean_pos. Calibrate off the bin that
# actually matches the concept's target residue instead of off bar colour.
_TARGET_RESIDUE_BIN = {"gcd": 0, "residue_class": 1}


def _draw_mini_1d(ax, feature: dict, concept: str, mod: int = 7) -> bool:
    result = None
    img_path = feature.get("_img_path")
    if img_path is not None and img_path.exists():
        result = _extract_1d_profile(img_path, n_bins=mod)
    if result is None:
        return False
    values, is_pos = result
    target_bin = _TARGET_RESIDUE_BIN.get(concept, 0)
    mean_pos = float(feature.get("mean_pos", 0.0))
    mean_neg = float(feature.get("mean_neg", 0.0))
    ref_px_pos = values[target_bin] if target_bin < len(values) else 0.0
    other_px = np.delete(values, target_bin) if target_bin < len(values) else values
    ref_px_neg = float(other_px.mean()) if other_px.size else 0.0
    # mean_pos is often ~0 for neg-polarity features (they barely fire on the
    # target-residue bin), which would otherwise collapse every bar to zero --
    # fall back to calibrating off mean_neg against the non-target bins instead.
    if ref_px_pos > 0 and mean_pos > 0:
        scale = mean_pos / ref_px_pos
    elif ref_px_neg > 0 and mean_neg > 0:
        scale = mean_neg / ref_px_neg
    else:
        scale = 1.0
    real_vals = values * scale
    colors = [_BAR_BLUE / 255 if p else _BAR_ORANGE / 255 for p in is_pos]
    x = np.arange(mod)
    ax.bar(x, real_vals, color=colors, width=0.65)
    ax.set_xticks(x)
    ax.tick_params(labelsize=11.5, length=3.5)
    ax.set_xlabel(f"a mod {mod}", fontsize=12.5)
    ax.grid(False)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    return True


def _draw_mini_2d(ax, feature: dict) -> bool:
    img_path = feature.get("_img_path")
    if img_path is None or not img_path.exists():
        return False
    grid = _extract_2d_profile(img_path)
    if grid is None:
        return False
    is_pos = feature.get("polarity") == "pos"
    end = _HEAT_POS_END if is_pos else _HEAT_NEG_END
    intensity = np.vectorize(lambda i, j: _project_intensity(grid[i, j], end))(
        *np.indices((grid.shape[0], grid.shape[1]))
    )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "heat", ["#f8f8f8", "#2B4590" if is_pos else "#C0392B"]
    )
    ax.imshow(intensity, origin="lower", cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ticks = list(range(0, 10, 2))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.tick_params(labelsize=11.5, length=3.5)
    ax.set_xlabel("$a_0$", fontsize=12.5)
    ax.set_ylabel("$b_0$", fontsize=12.5)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return True


def _draw_readout(ax, top: list[dict], concept: str, anchor: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = max(len(top), 1)
    slot = 1.0 / n
    for i, f in enumerate(top):
        y_top = 1.0 - i * slot
        polarity_color = "#DC2626" if f.get("polarity") == "pos" else "#2563EB"
        ax.text(0.03, y_top - slot * 0.015, f"{i + 1}. L{f['layer']}·F{f['feature_id']}",
                 fontsize=14, color=polarity_color, family="monospace", fontweight="bold",
                 ha="left", va="top", transform=ax.transAxes)

        # Leave a generous gap below each plot (x-tick labels + axis-label text
        # render outside the inset box's own bounds) before the next entry starts.
        f = dict(f, _img_path=_feature_image_path(concept, anchor, f))
        bax = ax.inset_axes([0.06, y_top - slot * 0.62, 0.90, slot * 0.50],
                             transform=ax.transAxes)
        plot_type = (f.get("visualization") or {}).get("plot_type")
        ok = _draw_mini_2d(bax, f) if plot_type == "2d" else _draw_mini_1d(bax, f, concept)
        if not ok:
            bax.axis("off")
            bax.text(0.5, 0.5, "(profile unavailable)", fontsize=12, color="#9CA3AF",
                       ha="center", va="center", transform=bax.transAxes)


# ── Legends (one per row, shown once as a strip below that row) ────────────────

def _row_title(ax, text: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=22, color="#1F2937",
             transform=ax.transAxes)


def _legend_cosine(ax) -> None:
    ax.axis("off")
    ax.text(0.5, 0.92, "layer $\\times$ layer cosine similarity of the delta direction",
             fontsize=15, color="#4B5563", ha="center", va="top", transform=ax.transAxes)
    sm = plt.cm.ScalarMappable(cmap=ps.CMAP_DIV, norm=plt.Normalize(-1, 1))
    cax = ax.inset_axes([0.32, 0.08, 0.36, 0.32])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("cosine", fontsize=13.5)
    cb.ax.tick_params(labelsize=11.5, length=0)
    cb.outline.set_visible(False)


def _legend_trajectory(ax) -> None:
    ax.axis("off")
    handles = [
        Line2D([0], [0], color=ps.VIOLET, lw=2.2, label="raw / peak"),
        Line2D([0], [0], color=ps.TEAL, lw=1.8, ls=":", label="double-normalised"),
        Patch(color=ps.MAUVE, alpha=0.40, label="excess above null $+1\\sigma$"),
        Line2D([0], [0], color=ps.GRAY, lw=1.0, ls=":", label="peak layer"),
    ]
    ax.legend(handles=handles, loc="center", ncol=len(handles), fontsize=14.5, frameon=False,
               bbox_to_anchor=(0.5, 0.5))


def _legend_constellation(ax) -> None:
    ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_diverging_color(1.0), markersize=14,
               label="supports (score $>0$)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_diverging_color(-1.0), markersize=14,
               label="opposes (score $<0$)"),
        Line2D([0], [0], color="#4C9A6A", lw=2.0, label="positive connection"),
        Line2D([0], [0], color=ps.MAUVE, lw=2.0, label="negative connection"),
    ]
    ax.legend(handles=handles, loc="center", ncol=len(handles), fontsize=14.5, frameon=False,
               bbox_to_anchor=(0.5, 0.78))
    ax.text(0.5, 0.24,
             "x: layer (0–35)   y: combined decoder+encoder cosine, $-1$ (opposes) to $+1$ (supports)   "
             "dot size $\\propto$ |score|   largest dots = top-3 features, plotted below",
             fontsize=13, color="#6B7280", ha="center", va="center", transform=ax.transAxes)


# ── Figure assembly ──────────────────────────────────────────────────────────────

CONCEPT_LETTER = {
    "carry": "a",
    "gcd": "b",
    "residue_class": "c",
}

CONCEPT_TITLE = {
    "carry": "Carry",
    "gcd": "GCD divisibility",
    "residue_class": "Residue class",
}


def plot_concept_overview(concept: str) -> Path:
    data = load_concept(concept)
    anchors = data["anchors"]
    K = len(anchors)

    ps.apply()
    col_w = 2.9
    row_h = {
        "tokens": 1.05,
        "cosine_title": 0.75, "cosine": 3.5, "cosine_leg": 0.68,
        "traj_title": 0.75, "traj": 3.6, "traj_leg": 0.55,
        "const_title": 0.75, "const": 3.7, "const_leg": 0.68,
        "readout": 10.6,
    }
    fig_w = col_w * K
    title_h = 0.85
    fig_h = sum(row_h.values()) + title_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        len(row_h), K, figure=fig,
        width_ratios=[col_w] * K,
        height_ratios=list(row_h.values()),
        hspace=0.34, wspace=0.06,
        top=1 - title_h / fig_h, bottom=0.02, left=0.02, right=0.98,
    )
    ROW = {name: i for i, name in enumerate(row_h)}

    ax_tok = fig.add_subplot(gs[ROW["tokens"], :])
    _draw_token_row(fig, ax_tok, data, anchors)

    _row_title(fig.add_subplot(gs[ROW["cosine_title"], :]),
                r"$C_{l,m} = \delta_l \cdot \delta_m \,/\, (\Vert\delta_l\Vert\,\Vert\delta_m\Vert)$"
                r"  —  inter-layer cosine similarity of the delta direction")
    _row_title(fig.add_subplot(gs[ROW["traj_title"], :]),
                r"$D_l = \Vert\delta_l\Vert_2$ (raw) and $\tilde D_l^{\mathrm{act}}$ (double-normalised)"
                r"  —  delta norm across layers vs. permutation null")
    _row_title(fig.add_subplot(gs[ROW["const_title"], :]),
                r"$\sigma_{l,f} = \cos^{\mathrm{dec}}_{l,f} + \cos^{\mathrm{enc}}_{l,f}$"
                r"  —  combined transcoder feature alignment")

    for col, anchor in enumerate(anchors):
        top = top_features(anchor, TOP_K)
        leftmost = col == 0

        ax_cos = fig.add_subplot(gs[ROW["cosine"], col])
        _column_header(ax_cos, col + 1, anchor)
        _draw_cosine(ax_cos, anchor, show_axis=leftmost)

        ax_traj = fig.add_subplot(gs[ROW["traj"], col])
        _draw_trajectory(ax_traj, anchor, show_axis=leftmost)

        ax_const = fig.add_subplot(gs[ROW["const"], col])
        _draw_constellation(ax_const, anchor, top, show_axis=leftmost)

        ax_read = fig.add_subplot(gs[ROW["readout"], col])
        _draw_readout(ax_read, top, concept, anchor)

    _legend_cosine(fig.add_subplot(gs[ROW["cosine_leg"], :]))
    _legend_trajectory(fig.add_subplot(gs[ROW["traj_leg"], :]))
    _legend_constellation(fig.add_subplot(gs[ROW["const_leg"], :]))

    letter = CONCEPT_LETTER.get(concept, "?")
    fig.text(0.5, 1 - 0.28 * title_h / fig_h,
              f"({letter}) {CONCEPT_TITLE.get(concept, concept)} — concept-localisation overview across multiple anchors",
              ha="center", va="center", fontsize=27, fontweight="bold")
    fig.text(0.5, 1 - 0.75 * title_h / fig_h,
              f"template: {data['template_text'].strip()}",
              ha="center", va="center", fontsize=17, color="#4B5563", style="italic")

    out_path = OUT_DIR / concept / "concept_overview_T0_thesis.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[{concept}] saved -> {out_path}  ({fig_w:.1f}in x {fig_h:.1f}in, aspect {fig_w/fig_h:.2f})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--concept")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    concepts = ["carry", "gcd", "residue_class"] if args.all else [args.concept]
    for concept in concepts:
        plot_concept_overview(concept)


if __name__ == "__main__":
    main()
