"""Plot top discriminative features at peak layers for each carry anchor.

Produces three figures per anchor:

  top_features_peak_layers.png
      Top features at peak layers, ranked by jac×|score|.

  top_features_peak_layer_clusters.png
      Top 1000 peak-layer features clustered by activation shape into 10
      clusters, plotting the top 5 ranked features from each cluster.

  top_features_peak_layer_clusters_by_activation.png
      Same clusters, but plotting the top 5 features per cluster by mean
      absolute activation.

  top_features_fourier_modes.png
      Features from the full ranked list whose activation grid is well-explained
      by a Fourier mode (R² ≥ --fourier_threshold), sorted by R².  Covers 1D
      iso-sum, 1D iso-diff, and 2D product modes.

Usage
-----
    python scripts/sweeps/plot_sweep_peak_features.py
    python scripts/sweeps/plot_sweep_peak_features.py --concept carry --max_features 30
    python scripts/sweeps/plot_sweep_peak_features.py --fourier_threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import experiments.plot_style as ps
from experiments.concept_localization.attr_survival import load_survival_set

_BASE = _REPO_ROOT / "runs" / "concept_localization"
_DEFAULT_PEAK_LAYERS_JSON = Path(__file__).with_name("peak_layers.json")
_BDY_XS = np.linspace(9.5, 0.5, 200)
_BDY_YS = 9.5 - _BDY_XS


# ── Peak-layer selection ──────────────────────────────────────────────────────

def expand_layer_window(
    layers: list[int],
    window: tuple[int, int] = (-1, 2),
    n_layers: int = 36,
) -> list[int]:
    result: set[int] = set()
    for peak in layers:
        for offset in range(window[0], window[1] + 1):
            layer = peak + offset
            if 0 <= layer < n_layers:
                result.add(layer)
    return sorted(result)


def load_peak_layer_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def peak_layers_from_config(
    config: dict | list,
    concept: str,
    anchor_name: str,
    anchor_rank: int,
    anchor_pos: int,
) -> list[int] | None:
    """Read manual peak layers from JSON.

    The per-anchor form is intentionally simple: [9, 13, 20].
    Global JSON files may also map by concept/anchor for batch runs.
    """
    if isinstance(config, list):
        if all(isinstance(layer, int) for layer in config):
            return [int(layer) for layer in config]
        for entry in config:
            if not isinstance(entry, dict):
                continue
            if (
                entry.get("concept", concept) == concept
                and int(entry.get("anchor_rank", -1)) == anchor_rank
                and int(entry.get("anchor_pos", -1)) == anchor_pos
            ):
                return [int(layer) for layer in entry["layers"]]
        return None

    if not isinstance(config, dict):
        raise ValueError("peak layer config must be a dict or list")

    concept_config = config.get(concept, {})
    for candidate in (concept_config, config):
        if isinstance(candidate, dict) and anchor_name in candidate:
            layers = candidate[anchor_name]
            if isinstance(layers, dict):
                layers = layers["layers"]
            return [int(layer) for layer in layers]
    return None


def peak_layers_from_results(
    results: dict,
    second_peak_threshold: float = 0.75,
    window: tuple[int, int] = (-1, 2),
    n_layers: int = 36,
) -> list[int]:
    norms = {int(k): v for k, v in results["sharpness"]["norm_by_layer"].items()}
    layers = sorted(norms)
    vals = np.array([norms[l] for l in layers], dtype=float)
    vals_n = vals / (vals.max() + 1e-12)

    maxima: list[tuple[int, float]] = []
    for i, l in enumerate(layers):
        left  = vals_n[i - 1] if i > 0             else -1.0
        right = vals_n[i + 1] if i < len(layers)-1 else -1.0
        if vals_n[i] > left and vals_n[i] > right:
            maxima.append((l, float(vals_n[i])))
    maxima.sort(key=lambda x: -x[1])

    selected: list[int] = []
    if maxima:
        selected.append(maxima[0][0])
        if len(maxima) > 1 and maxima[1][1] >= second_peak_threshold:
            selected.append(maxima[1][0])

    return expand_layer_window(selected, window=window, n_layers=n_layers)


# ── Carry grid builder ────────────────────────────────────────────────────────

def build_carry_grid(
    feat_acts: np.ndarray,
    examples: list[dict],
    use_pos: bool = True,
) -> np.ndarray:
    """Mean activation per (ones(a), ones(b)) cell; NaN for empty cells."""
    grid_sum = np.zeros((10, 10), dtype=np.float32)
    grid_cnt = np.zeros((10, 10), dtype=np.int32)
    for pair_i, ex in enumerate(examples):
        meta = ex["meta"]
        if use_pos:
            a, b    = meta["a_pos"], meta["b_pos"]
            act_idx = 2 * pair_i
        else:
            a, b    = meta["a_neg"], meta["b_neg"]
            act_idx = 2 * pair_i + 1
        if act_idx >= len(feat_acts):
            continue
        d_a, d_b = a % 10, b % 10
        grid_sum[d_a, d_b] += feat_acts[act_idx]
        grid_cnt[d_a, d_b] += 1
    with np.errstate(invalid="ignore"):
        grid = grid_sum / np.where(grid_cnt > 0, grid_cnt, np.nan)
    grid[grid_cnt == 0] = np.nan
    return grid


# ── Fourier basis (precomputed once at import) ────────────────────────────────

def _make_fourier_bases() -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Build a (n_bases, 100) matrix of centred, unit-norm Fourier basis vectors
    on the 10×10 (ones-a, ones-b) grid.

    Families:
      iso-sum  k=1..5  — cos(2πk(a+b)/10 + φ), 8 phases in [0,π)
      iso-diff k=1..5  — cos(2πk(b-a)/10 + φ), 8 phases
      2D       ka×kb   — cos(2πka·a/10+φa)·cos(2πkb·b/10+φb), 4×4 phases
    """
    da, db = np.meshgrid(np.arange(10), np.arange(10), indexing='ij')
    da_f = da.flatten().astype(float)
    db_f = db.flatten().astype(float)

    rows: list[np.ndarray] = []
    meta: list[tuple[str, int]] = []

    for k in range(1, 6):
        for phi in np.linspace(0, np.pi, 8, endpoint=False):
            v = np.cos(2 * np.pi * k * (da_f + db_f) / 10 + phi)
            v -= v.mean()
            rows.append(v)
            meta.append(('sum', k))
        for phi in np.linspace(0, np.pi, 8, endpoint=False):
            v = np.cos(2 * np.pi * k * (db_f - da_f) / 10 + phi)
            v -= v.mean()
            rows.append(v)
            meta.append(('diff', k))

    for ka in range(1, 6):
        for kb in range(1, 6):
            for pa in np.linspace(0, np.pi, 4, endpoint=False):
                for pb in np.linspace(0, np.pi, 4, endpoint=False):
                    v = (np.cos(2 * np.pi * ka * da_f / 10 + pa) *
                         np.cos(2 * np.pi * kb * db_f / 10 + pb))
                    v -= v.mean()
                    rows.append(v)
                    meta.append(('2d', ka * 10 + kb))

    B = np.array(rows, dtype=np.float32)
    norms = np.linalg.norm(B, axis=1, keepdims=True)
    B /= np.where(norms > 1e-12, norms, 1.0)
    return B, meta


_FOURIER_B, _FOURIER_META = _make_fourier_bases()


def fourier_mode_scores(grid: np.ndarray) -> dict:
    """Score a (10,10) activation grid for Fourier mode content.

    Computes R² = (dot(g_c, b))² / ||g_c||² for every precomputed basis b,
    where g_c is the centred activation grid.  Returns the mode with the
    highest R² across all bases and phase shifts.

    Returns:
        best   float  max R² ∈ [0,1]
        type   str    'sum' | 'diff' | '2d' | 'none'
        k      int    harmonic (1–5) or ka*10+kb for 2D
        label  str    e.g. 'iso-sum k=1', '2D 3×2'
    """
    g = np.nan_to_num(grid, nan=0.0).flatten().astype(np.float32)
    g -= g.mean()
    g_norm_sq = float(np.dot(g, g))
    if g_norm_sq < 1e-12:
        return {'best': 0.0, 'type': 'none', 'k': 0, 'label': 'none'}

    r2s = (_FOURIER_B @ g) ** 2 / g_norm_sq   # (n_bases,)

    best_idx          = int(r2s.argmax())
    best_r2           = float(r2s[best_idx])
    btype, bk         = _FOURIER_META[best_idx]

    if btype == 'sum':
        label = f'iso-sum k={bk}'
    elif btype == 'diff':
        label = f'iso-diff k={bk}'
    else:
        label = f'2D {bk//10}×{bk%10}'

    return {'best': best_r2, 'type': btype, 'k': bk, 'label': label}


# ── Shared grid-cell drawing ──────────────────────────────────────────────────

def _draw_cell(ax, grid, cmap, vmin, vmax, title, fig):
    im = ax.imshow(
        grid.T,
        origin="lower", aspect="equal", cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[-0.5, 9.5, -0.5, 9.5], interpolation="nearest",
    )
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(-0.5, 9.5)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticks(np.arange(-0.5, 10, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1.0), minor=True)
    ax.tick_params(which='both', length=0, labelsize=5)   # no tick marks on either major or minor
    ax.grid(which='minor', color='#DDDDDD', linewidth=0.3)
    ax.grid(which='major', visible=False)
    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_color(ps.GRAY)
        spine.set_visible(True)
    ax.set_xlabel("ones(a)", fontsize=6, labelpad=1)
    ax.set_ylabel("ones(b)", fontsize=6, labelpad=1)
    ax.set_title(title, fontsize=7, pad=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5, length=0)
    cbar.outline.set_edgecolor(ps.GRAY)


# ── Figure 1: top features at peak layers (jac×|score|) ──────────────────────

def plot_peak_features(
    sweep_dir: Path,
    anchor_name: str,
    anchor_pos: int,
    anchor_rank: int,
    peak_layers: list[int],
    max_features: int = 30,
    top_pct: float = 0.30,
    ncols: int = 5,
    survival_set: set[tuple[int, int]] | None = None,
) -> list[tuple]:
    """Plot carry grids for top-jac×|score| features; return items for Figure 2."""
    ranked_path = sweep_dir / "sweep_ranked.json"
    acts_path   = sweep_dir / "sweep_activations.npz"
    ex_path     = sweep_dir / "sweep_examples.pkl"
    if not (ranked_path.exists() and acts_path.exists() and ex_path.exists()):
        print(f"  [skip] missing sweep files in {sweep_dir}")
        return []

    ranked_all = json.loads(ranked_path.read_text())
    npz        = np.load(acts_path)
    with open(ex_path, "rb") as f:
        examples = pickle.load(f)

    is_carry = len(examples) > 0 and "b_pos" in examples[0]["meta"]

    peak_set   = set(peak_layers)
    from_peak  = [r for r in ranked_all if r["layer"] in peak_set]
    if survival_set is not None:
        n_before = len(from_peak)
        from_peak = [r for r in from_peak if (r["layer"], r["feat_id"]) in survival_set]
        print(f"  [attr-survival] {anchor_name}: {n_before} → {len(from_peak)} features after graph-survival filter")
    if not from_peak:
        print(f"  [skip] no ranked features at peak layers {peak_layers}")
        return []

    n_select = min(max_features, max(1, math.ceil(len(from_peak) * top_pct)))
    selected = from_peak[:n_select]
    print(
        f"  {anchor_name}: {len(from_peak)} features at peak layers "
        f"{peak_layers[:6]}{'…' if len(peak_layers)>6 else ''}  →  "
        f"plotting top {n_select} ({top_pct*100:.0f}%)"
    )

    items: list[tuple] = []
    for r in selected:
        key = f"L{r['layer']}_F{r['feat_id']}"
        if key not in npz:
            continue
        feat_acts  = npz[key].astype(np.float32)
        if is_carry:
            grid_pos   = build_carry_grid(feat_acts, examples, use_pos=True)
            grid_neg   = build_carry_grid(feat_acts, examples, use_pos=False)
            grid_comb  = np.where(np.isnan(grid_pos), grid_neg, grid_pos)
            fourier    = fourier_mode_scores(grid_comb)
            items.append((r["layer"], r["feat_id"], r["score"], r["jaccard"],
                          grid_comb, fourier))
        else:
            items.append((r["layer"], r["feat_id"], r["score"], r["jaccard"],
                          feat_acts))

    if not items:
        print(f"  [skip] no features found in npz for {anchor_name}")
        return []

    n     = len(items)
    nrows = math.ceil(n / ncols)
    cell  = 3.0
    ps.apply()
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * cell, nrows * (cell + 0.5) + 1.5),
                             squeeze=False)

    if is_carry:
        for idx, (layer, feat_id, score, jac, grid, fourier) in enumerate(items):
            ax   = axes[idx // ncols][idx % ncols]
            vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0
            vmax = max(vmax, 1e-8)
            cmap = "Blues" if score >= 0 else "Reds"
            sign = "+" if score >= 0 else ""
            title = (f"L{layer:02d} F{feat_id}\n"
                     f"cs={sign}{score:.2f}  jac={jac:.2f}\n"
                     f"R²={fourier['best']:.2f} ({fourier['label']})")
            _draw_cell(ax, grid, cmap, 0, vmax, title, fig)
    else:
        from matplotlib.patches import Patch
        pos_mask = npz["pos_mask"] if "pos_mask" in npz else None
        if pos_mask is not None:
            bar_colors = ["#2196F3" if m else "#E53935" for m in pos_mask]
        else:
            bar_colors = ["#2196F3"] * len(items[0][4])
        for idx, (layer, feat_id, score, jac, feat_acts) in enumerate(items):
            ax   = axes[idx // ncols][idx % ncols]
            x = np.arange(len(feat_acts))
            ax.bar(x, feat_acts, color=bar_colors, width=0.8)
            ax.tick_params(labelsize=5)
            ax.set_xticks([])
            sign = "+" if score >= 0 else ""
            title = (f"L{layer:02d} F{feat_id}\n"
                     f"score={sign}{score:.2f}  jac={jac:.2f}")
            ax.set_title(title, fontsize=7, pad=2)
            ax.set_xlabel("examples", fontsize=6, labelpad=1)
            ax.set_ylabel("activation", fontsize=6, labelpad=1)

        fig.legend(
            handles=[Patch(facecolor="#2196F3", label="pos"), Patch(facecolor="#E53935", label="neg")],
            loc="upper right",
            fontsize=7,
            framealpha=0.8,
        )

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    layer_str = ",".join(str(l) for l in sorted(peak_layers)[:8])
    if len(peak_layers) > 8:
        layer_str += "…"
    
    if is_carry:
        fig.suptitle(
            f"carry — anchor rank {anchor_rank}, pos {anchor_pos}\n"
            f"Top {n_select} features at peak layers [{layer_str}]  "
            f"(top {top_pct*100:.0f}% by jac×|score|)",
            fontsize=9, y=1.005,
        )
    else:
        concept_name = anchor_name.split("_")[0] if "_" in anchor_name else "concept"
        fig.suptitle(
            f"{concept_name} — anchor rank {anchor_rank}, pos {anchor_pos}\n"
            f"Top {n_select} features at peak layers [{layer_str}]  "
            f"(top {top_pct*100:.0f}% by jac×|score|)",
            fontsize=9, y=1.005,
        )
        
    fig.tight_layout()
    out_path = sweep_dir / "top_features_peak_layers.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")
    return items


# ── Clustered peak-layer features ─────────────────────────────────────────────

def _normalise_rows(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 1e-8, norms, 1.0)


def _farthest_first_labels(x: np.ndarray, n_clusters: int) -> np.ndarray:
    """Assign examples to deterministic farthest-first cosine medoids.

    K-means can spend multiple centroids splitting one very dense visual motif.
    For this plot we want diverse activation shapes, so keep farthest-first
    medoids fixed and assign each feature to its nearest medoid.
    """
    n = x.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    n_clusters = min(n_clusters, n)

    centers = [0]
    min_dist = np.sum((x - x[0]) ** 2, axis=1)
    for _ in range(1, n_clusters):
        idx = int(np.argmax(min_dist))
        centers.append(idx)
        min_dist = np.minimum(min_dist, np.sum((x - x[idx]) ** 2, axis=1))

    medoids = x[np.array(centers)]
    dist = np.sum((x[:, None, :] - medoids[None, :, :]) ** 2, axis=2)
    return dist.argmin(axis=1)


def plot_peak_feature_clusters(
    sweep_dir: Path,
    anchor_name: str,
    anchor_pos: int,
    anchor_rank: int,
    peak_layers: list[int],
    cluster_pool_size: int = 1000,
    n_clusters: int = 10,
    top_per_cluster: int = 5,
    sort_by: str = "rank",
    out_name: str = "top_features_peak_layer_clusters.png",
    survival_set: set[tuple[int, int]] | None = None,
) -> None:
    """Cluster top peak-layer features and plot top features per cluster.

    sort_by='rank' preserves sweep_ranked.json order within each cluster;
    sort_by='activation' uses mean absolute activation.
    """
    ranked_path = sweep_dir / "sweep_ranked.json"
    acts_path   = sweep_dir / "sweep_activations.npz"
    ex_path     = sweep_dir / "sweep_examples.pkl"
    if not (ranked_path.exists() and acts_path.exists() and ex_path.exists()):
        return

    ranked_all = json.loads(ranked_path.read_text())
    npz        = np.load(acts_path)
    with open(ex_path, "rb") as f:
        examples = pickle.load(f)

    is_carry = len(examples) > 0 and "b_pos" in examples[0]["meta"]
    peak_set = set(peak_layers)
    candidates = []
    vectors = []

    pool = [row for row in ranked_all if row["layer"] in peak_set]
    if survival_set is not None:
        n_before = len(pool)
        pool = [r for r in pool if (r["layer"], r["feat_id"]) in survival_set]
        print(f"  [attr-survival] {anchor_name} clusters: {n_before} → {len(pool)} after graph-survival filter")

    for rank_idx, r in enumerate(pool):
        if len(candidates) >= cluster_pool_size:
            break
        key = f"L{r['layer']}_F{r['feat_id']}"
        if key not in npz:
            continue
        feat_acts = npz[key].astype(np.float32)
        if is_carry:
            grid_pos  = build_carry_grid(feat_acts, examples, use_pos=True)
            grid_neg  = build_carry_grid(feat_acts, examples, use_pos=False)
            grid      = np.where(np.isnan(grid_pos), grid_neg, grid_pos)
            fourier   = fourier_mode_scores(grid)
            plot_obj  = grid
            vector    = grid.flatten()
            act_score = float(np.nanmean(np.abs(grid)))
            candidates.append((rank_idx, r["layer"], r["feat_id"], r["score"], r["jaccard"], act_score, plot_obj, fourier))
        else:
            plot_obj  = feat_acts
            vector    = feat_acts
            act_score = float(np.nanmean(np.abs(feat_acts)))
            candidates.append((rank_idx, r["layer"], r["feat_id"], r["score"], r["jaccard"], act_score, plot_obj))
        vectors.append(vector)

    if not candidates:
        print(f"  [clusters] no candidate features found for {anchor_name}")
        return

    x = _normalise_rows(np.stack(vectors))
    labels = _farthest_first_labels(x, n_clusters=n_clusters)

    groups: list[tuple[int, list[int]]] = []
    if sort_by not in {"rank", "activation"}:
        raise ValueError("sort_by must be 'rank' or 'activation'")

    def member_sort_key(idx: int) -> tuple[float, int]:
        if sort_by == "activation":
            return (-candidates[idx][5], candidates[idx][0])
        return (candidates[idx][0], 0)

    for cluster_id in sorted(set(int(label) for label in labels)):
        member_idxs = np.flatnonzero(labels == cluster_id).tolist()
        member_idxs.sort(key=member_sort_key)
        groups.append((cluster_id, member_idxs))
    groups.sort(key=lambda item: member_sort_key(item[1][0]))

    plotted: list[tuple[int, tuple]] = []
    for cluster_id, member_idxs in groups:
        for idx in member_idxs[:top_per_cluster]:
            plotted.append((cluster_id, candidates[idx]))

    nrows = len(groups)
    ncols = top_per_cluster
    cell = 2.6 if is_carry else 2.2
    ps.apply()
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * cell, nrows * (cell + 0.35) + 1.6),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.set_visible(False)

    if is_carry:
        for row_idx, (cluster_id, member_idxs) in enumerate(groups):
            for col_idx, idx in enumerate(member_idxs[:top_per_cluster]):
                rank_idx, layer, feat_id, score, jac, act_score, grid, fourier = candidates[idx]
                ax = axes[row_idx][col_idx]
                ax.set_visible(True)
                vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0
                vmax = max(vmax, 1e-8)
                cmap = "Blues" if score >= 0 else "Reds"
                sign = "+" if score >= 0 else ""
                title = (f"C{cluster_id} #{rank_idx + 1}  L{layer:02d} F{feat_id}\n"
                         f"cs={sign}{score:.2f} jac={jac:.2f} act={act_score:.2f}\n"
                         f"R²={fourier['best']:.2f} ({fourier['label']})")
                _draw_cell(ax, grid, cmap, 0, vmax, title, fig)
    else:
        from matplotlib.patches import Patch
        pos_mask = npz["pos_mask"] if "pos_mask" in npz else None
        for row_idx, (cluster_id, member_idxs) in enumerate(groups):
            for col_idx, idx in enumerate(member_idxs[:top_per_cluster]):
                rank_idx, layer, feat_id, score, jac, act_score, feat_acts = candidates[idx]
                ax = axes[row_idx][col_idx]
                ax.set_visible(True)
                colors = ["#2196F3" if m else "#E53935" for m in pos_mask] if pos_mask is not None else "#2196F3"
                ax.bar(np.arange(len(feat_acts)), feat_acts, color=colors, width=0.8)
                ax.set_xticks([])
                ax.tick_params(labelsize=5)
                sign = "+" if score >= 0 else ""
                ax.set_title(
                    f"C{cluster_id} #{rank_idx + 1}  L{layer:02d} F{feat_id}\n"
                    f"score={sign}{score:.2f} jac={jac:.2f} act={act_score:.2f}",
                    fontsize=7, pad=2,
                )
                ax.set_xlabel("examples", fontsize=6, labelpad=1)
                ax.set_ylabel("activation", fontsize=6, labelpad=1)
        if pos_mask is not None:
            fig.legend(
                handles=[Patch(facecolor="#2196F3", label="pos"), Patch(facecolor="#E53935", label="neg")],
                loc="upper right", fontsize=7, framealpha=0.8,
            )

    layer_str = ",".join(str(l) for l in sorted(peak_layers)[:10])
    if len(peak_layers) > 10:
        layer_str += "…"
    fig.suptitle(
        f"{anchor_name} — clustered peak-layer features\n"
        f"top {min(cluster_pool_size, len(candidates))} at layers [{layer_str}], "
        f"{len(groups)} clusters, top {top_per_cluster}/cluster by {sort_by}",
        fontsize=9, y=1.002,
    )
    fig.tight_layout()
    out_path = sweep_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  [clusters:{sort_by}] {anchor_name}: clustered {len(candidates)} features into "
        f"{len(groups)} clusters; plotted {len(plotted)} → {out_path}"
    )


# ── Figure 2: Fourier-mode features (all ranked, threshold-filtered) ──────────

_FOURIER_CMAPS = {'sum': 'Blues', 'diff': 'Greens', '2d': 'Purples', 'none': 'Greys'}


def plot_fourier_features(
    sweep_dir: Path,
    anchor_name: str,
    anchor_pos: int,
    anchor_rank: int,
    threshold: float = 0.25,
    max_features: int = 50,
    ncols: int = 5,
    survival_set: set[tuple[int, int]] | None = None,
) -> None:
    """Scan all ranked features, filter by Fourier R² threshold, plot as carry grids."""
    ranked_path = sweep_dir / "sweep_ranked.json"
    acts_path   = sweep_dir / "sweep_activations.npz"
    ex_path     = sweep_dir / "sweep_examples.pkl"
    if not (ranked_path.exists() and acts_path.exists() and ex_path.exists()):
        return

    ranked_all = json.loads(ranked_path.read_text())
    npz        = np.load(acts_path)
    with open(ex_path, "rb") as f:
        examples = pickle.load(f)

    is_carry = len(examples) > 0 and "b_pos" in examples[0]["meta"]
    if not is_carry:
        print(f"  [fourier] skip Fourier analysis for non-carry concept {anchor_name}")
        return

    # Score all features (pre-filter by attr survival if provided)
    pool = ranked_all
    if survival_set is not None:
        n_before = len(pool)
        pool = [r for r in pool if (r["layer"], r["feat_id"]) in survival_set]
        print(f"  [attr-survival] {anchor_name} fourier: {n_before} → {len(pool)} after graph-survival filter")
    scored: list[tuple] = []
    for r in pool:
        key = f"L{r['layer']}_F{r['feat_id']}"
        if key not in npz:
            continue
        feat_acts = npz[key].astype(np.float32)
        grid_pos  = build_carry_grid(feat_acts, examples, use_pos=True)
        grid_neg  = build_carry_grid(feat_acts, examples, use_pos=False)
        grid_comb = np.where(np.isnan(grid_pos), grid_neg, grid_pos)
        fourier   = fourier_mode_scores(grid_comb)
        if fourier['best'] >= threshold:
            scored.append((r["layer"], r["feat_id"], r["score"], r["jaccard"],
                           grid_comb, fourier))

    if not scored:
        print(f"  [fourier] no features above R²≥{threshold} for {anchor_name}")
        return

    # Sort by Fourier R² descending, cap at max_features
    scored.sort(key=lambda x: -x[5]['best'])
    scored = scored[:max_features]

    print(f"  [fourier] {anchor_name}: {len(scored)} features with R²≥{threshold}")

    n     = len(scored)
    nrows = math.ceil(n / ncols)
    cell  = 3.0
    ps.apply()
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * cell, nrows * (cell + 0.5) + 1.5),
                             squeeze=False)

    for idx, (layer, feat_id, score, jac, grid, fourier) in enumerate(scored):
        ax   = axes[idx // ncols][idx % ncols]
        vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0
        vmax = max(vmax, 1e-8)
        cmap = _FOURIER_CMAPS.get(fourier['type'], 'Blues')
        sign = "+" if score >= 0 else ""
        title = (f"L{layer:02d} F{feat_id}\n"
                 f"{fourier['label']}  R²={fourier['best']:.2f}\n"
                 f"cs={sign}{score:.2f}  jac={jac:.2f}")
        _draw_cell(ax, grid, cmap, 0, vmax, title, fig)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"carry — anchor rank {anchor_rank}, pos {anchor_pos}  |  "
        f"Fourier-mode features  (R² ≥ {threshold}, sorted by R²)\n"
        "Blue = iso-sum  ·  Green = iso-diff  ·  Purple = 2D product",
        fontsize=9, y=1.005,
    )
    fig.tight_layout()
    out_path = sweep_dir / "top_features_fourier_modes.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept",              default="carry")
    parser.add_argument("--max_features",         type=int,   default=30)
    parser.add_argument("--top_pct",              type=float, default=0.30)
    parser.add_argument("--second_peak_thresh",   type=float, default=0.75)
    parser.add_argument("--window",               type=str,   default="-1,2",
                        help="Inclusive offsets around each configured/manual peak layer")
    parser.add_argument("--peak_layers_json",     type=Path, default=_DEFAULT_PEAK_LAYERS_JSON,
                        help="JSON file with manual peak layers by concept and anchor")
    parser.add_argument("--ncols",                type=int,   default=5)
    parser.add_argument("--cluster_pool_size",    type=int,   default=1000,
                        help="Top ranked peak-layer features to cluster")
    parser.add_argument("--n_clusters",           type=int,   default=10,
                        help="Number of peak-feature activation clusters")
    parser.add_argument("--top_per_cluster",      type=int,   default=5,
                        help="Features plotted from each activation cluster")
    parser.add_argument("--fourier_threshold",    type=float, default=0.25,
                        help="Min R² for a feature to appear in the Fourier figure")
    parser.add_argument("--fourier_max_features", type=int,   default=50)
    parser.add_argument("--no_attr_filter",       action="store_true",
                        help="Disable attribution-graph survival pre-filter (not recommended)")
    parser.add_argument("--attr_min_survival",    type=float, default=0.05,
                        help="Min fraction of graphs a feature must survive to pass the filter")
    parser.add_argument("--attr_survival_file",   type=Path,  default=None,
                        help="Explicit path to survival_stats.json (overrides default location)")
    args = parser.parse_args()

    concept_dir = _BASE / args.concept
    if not concept_dir.exists():
        print(f"No run directory for concept '{args.concept}'")
        sys.exit(1)

    anchor_dirs = sorted(concept_dir.glob("anchor_rank*_pos*"))
    if not anchor_dirs:
        print(f"No anchor subdirectories found in {concept_dir}")
        sys.exit(1)

    win = tuple(int(x) for x in args.window.split(","))
    if len(win) != 2:
        raise ValueError("--window must be two comma-separated integers, e.g. -1,2")
    global_peak_layer_config = load_peak_layer_config(args.peak_layers_json)

    if args.no_attr_filter:
        survival_set = None
        print("  [attr-survival] filter disabled via --no_attr_filter")
    else:
        survival_set = load_survival_set(
            concept=args.concept,
            min_survival=args.attr_min_survival,
            survival_file=args.attr_survival_file,
            required=True,
        )

    for anchor_dir in anchor_dirs:
        m = re.match(r"anchor_rank(\d+)_pos(\d+)", anchor_dir.name)
        if not m:
            continue
        rank = int(m.group(1))
        pos  = int(m.group(2))

        res_path = anchor_dir / "results.json"
        if not res_path.exists():
            continue
        results = json.loads(res_path.read_text())

        anchor_peak_layer_path = anchor_dir / "peak_layers.json"
        peak_layer_config = (
            load_peak_layer_config(anchor_peak_layer_path)
            if anchor_peak_layer_path.exists()
            else global_peak_layer_config
        )
        manual_peak_layers = peak_layers_from_config(
            peak_layer_config,
            concept=args.concept,
            anchor_name=anchor_dir.name,
            anchor_rank=rank,
            anchor_pos=pos,
        )
        if manual_peak_layers is not None:
            peak_layers = expand_layer_window(manual_peak_layers, window=win)
            print(f"\n{anchor_dir.name}  manual_peak_layers={manual_peak_layers}  "
                  f"window={win}  peak_layers={peak_layers}")
        else:
            peak_layers = peak_layers_from_results(
                results,
                second_peak_threshold=args.second_peak_thresh,
                window=win,
            )
            print(f"\n{anchor_dir.name}  peak_layer={results['sharpness']['peak_layer']}  "
                  f"peak_layers={peak_layers}")

        plot_peak_features(
            sweep_dir    = anchor_dir / "sweep",
            anchor_name  = anchor_dir.name,
            anchor_pos   = pos,
            anchor_rank  = rank,
            peak_layers  = peak_layers,
            max_features = args.max_features,
            top_pct      = args.top_pct,
            ncols        = args.ncols,
            survival_set = survival_set,
        )
        plot_peak_feature_clusters(
            sweep_dir         = anchor_dir / "sweep",
            anchor_name       = anchor_dir.name,
            anchor_pos        = pos,
            anchor_rank       = rank,
            peak_layers       = peak_layers,
            cluster_pool_size = args.cluster_pool_size,
            n_clusters        = args.n_clusters,
            top_per_cluster   = args.top_per_cluster,
            sort_by           = "rank",
            out_name          = "top_features_peak_layer_clusters.png",
            survival_set      = survival_set,
        )
        plot_peak_feature_clusters(
            sweep_dir         = anchor_dir / "sweep",
            anchor_name       = anchor_dir.name,
            anchor_pos        = pos,
            anchor_rank       = rank,
            peak_layers       = peak_layers,
            cluster_pool_size = args.cluster_pool_size,
            n_clusters        = args.n_clusters,
            top_per_cluster   = args.top_per_cluster,
            sort_by           = "activation",
            out_name          = "top_features_peak_layer_clusters_by_activation.png",
            survival_set      = survival_set,
        )
        plot_fourier_features(
            sweep_dir    = anchor_dir / "sweep",
            anchor_name  = anchor_dir.name,
            anchor_pos   = pos,
            anchor_rank  = rank,
            threshold    = args.fourier_threshold,
            max_features = args.fourier_max_features,
            ncols        = args.ncols,
            survival_set = survival_set,
        )


if __name__ == "__main__":
    main()
