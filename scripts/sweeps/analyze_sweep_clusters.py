"""Cluster-based analysis of transcoder feature sweep results.

For a given sweep directory (sweep_ranked.json + sweep_activations.npz +
sweep_examples.pkl) this script:

  1. Filters examples to a single template (default T0).
  2. Builds the delta matrix D[i] = pos_act[i] - neg_act[i] for the top-K
     ranked features at peak layers.
  3. Clusters features by cosine similarity.
  4. For each cluster:
       a. Plots top-3 features as activation bar charts (pos blue / neg red).
       b. Runs PCA and reports explained variance + class separation.

Outputs land in <sweep_dir>/cluster_analysis_T0/  (or the specified out_dir).

Usage
-----
    # standalone
    python scripts/sweeps/analyze_sweep_clusters.py \\
        --sweep_dir runs/concept_localization/gcd/anchor_rank1_pos6/sweep

    # all concepts, all rank-1 anchors
    python scripts/sweeps/analyze_sweep_clusters.py --all

    # specific concept(s)
    python scripts/sweeps/analyze_sweep_clusters.py --concept gcd carry
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import experiments.plot_style as ps

_BASE = _REPO_ROOT / "runs" / "concept_localization"

THRESHOLDS = {
    "max_fourier_r2": 0.60,
    "sv_ratio_12":    0.50,
    "abs_pearson_r":  0.40,
    "n_feats":        3,
    "var_pc12":       0.30,
}
EXTRA_PC_PAIRS = [(1, 3), (1, 4), (2, 3), (3, 4)]


def passes_thresholds(summary: dict) -> tuple[bool, dict]:
    """Return (passes, per-metric check dict). passes is True iff ALL conditions met."""
    n   = summary.get("n_features", 0)
    r2  = summary.get("max_fourier_r2", float("nan"))
    sv  = summary.get("sv_ratio_12", 0.0)
    pr  = summary.get("pearson_r", float("nan"))
    vp  = summary.get("var_pc12", 0.0)
    checks = {
        "max_fourier_r2_ok": (not math.isnan(r2)) and r2  > THRESHOLDS["max_fourier_r2"],
        "sv_ratio_12_ok":    sv  > THRESHOLDS["sv_ratio_12"],
        "abs_pearson_r_ok":  (not math.isnan(pr)) and abs(pr) > THRESHOLDS["abs_pearson_r"],
        "n_feats_ok":        n   >= THRESHOLDS["n_feats"],
        "var_pc12_ok":       vp  > THRESHOLDS["var_pc12"],
    }
    return all(checks.values()), checks


# ── Data loading ──────────────────────────────────────────────────────────────

def _peak_layers(results: dict, n_layers: int = 36) -> set[int]:
    norms = {int(k): v for k, v in results["sharpness"]["norm_by_layer"].items()}
    ls = sorted(norms)
    vs = np.array([norms[l] for l in ls], dtype=float)
    vs_n = vs / (vs.max() + 1e-12)
    maxima = [(ls[i], float(vs_n[i]))
              for i in range(len(ls))
              if vs_n[i] > (vs_n[i-1] if i > 0 else -1.)
              and vs_n[i] > (vs_n[i+1] if i < len(ls)-1 else -1.)]
    maxima.sort(key=lambda x: -x[1])
    sel = [maxima[0][0]] if maxima else []
    if len(maxima) > 1 and maxima[1][1] >= 0.75:
        sel.append(maxima[1][0])
    out: set[int] = set()
    for p in sel:
        for o in range(-1, 3):
            l = p + o
            if 0 <= l < n_layers:
                out.add(l)
    return out


def load_sweep(sweep_dir: Path, template: str | None = "T0"):
    """Return (ranked, npz, examples, pair_indices) filtered to template."""
    ranked  = json.loads((sweep_dir / "sweep_ranked.json").read_text())
    npz     = np.load(sweep_dir / "sweep_activations.npz")
    with open(sweep_dir / "sweep_examples.pkl", "rb") as f:
        examples = pickle.load(f)

    if template:
        pair_indices = [i for i, e in enumerate(examples) if e.get("template") == template]
    else:
        pair_indices = list(range(len(examples)))

    return ranked, npz, examples, pair_indices


def build_delta_matrix(
    ranked: list[dict],
    npz,
    pair_indices: list[int],
    top_k: int,
    peak_layers: set[int] | None,
) -> tuple[np.ndarray, list[str]]:
    """(n_pairs × n_feats) delta matrix restricted to pair_indices."""
    pool = [r for r in ranked if peak_layers is None or r["layer"] in peak_layers]
    if not pool:
        pool = ranked
    selected = pool[:top_k]

    cols, feat_labels = [], []
    for r in selected:
        key = f"L{r['layer']}_F{r['feat_id']}"
        if key not in npz:
            continue
        arr = npz[key].astype(np.float32)        # shape (2 * n_all_pairs,)
        pos = arr[0::2]                           # even = pos
        neg = arr[1::2]                           # odd  = neg
        if pair_indices:
            max_i = min(len(pos), len(neg))
            idx = [i for i in pair_indices if i < max_i]
            if not idx:
                continue
            delta = pos[idx] - neg[idx]
        else:
            n = min(len(pos), len(neg))
            delta = pos[:n] - neg[:n]
        cols.append(delta)
        feat_labels.append(key)

    if not cols:
        return np.empty((0, 0)), []
    lengths = [len(c) for c in cols]
    min_len = min(lengths)
    D = np.column_stack([c[:min_len] for c in cols])
    return D, feat_labels


# ── Label extraction ──────────────────────────────────────────────────────────

def extract_labels(
    examples: list[dict], pair_indices: list[int]
) -> tuple[np.ndarray, str]:
    """Extract the most informative per-pair label from metadata."""
    metas = [examples[i]["meta"] for i in pair_indices]

    # known modular / class fields
    for key in ["offset", "remainder", "class", "carry", "result"]:
        if key in metas[0]:
            try:
                vals = np.array([int(m[key]) if isinstance(m[key], bool)
                                  else m[key] for m in metas])
                n_u = len(np.unique(vals))
                if 2 <= n_u <= 12:
                    return vals.astype(int), key
            except Exception:
                pass

    # any integer/bool field with 2-10 unique values
    for key in list(metas[0].keys())[:8]:
        try:
            vals = np.array([m[key] for m in metas], dtype=float)
            n_u = len(np.unique(vals))
            if 2 <= n_u <= 10:
                return vals.astype(int), key
        except Exception:
            pass

    # fallback: pair index mod 4
    idx_arr = np.array(pair_indices)
    return (idx_arr % 4).astype(int), "pair_idx mod 4"


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_features(D: np.ndarray, n_clusters: int, min_size: int = 2):
    norms = np.linalg.norm(D, axis=0, keepdims=True).clip(min=1e-10)
    D_n   = D / norms
    cos   = np.clip(D_n.T @ D_n, -1.0, 1.0)
    dist  = np.clip(1.0 - cos, 0.0, None)
    clust = AgglomerativeClustering(n_clusters=n_clusters,
                                    metric="precomputed", linkage="average")
    labels = clust.fit_predict(dist)
    groups = [np.where(labels == c)[0].tolist()
              for c in range(n_clusters) if (labels == c).sum() >= min_size]
    return cos, labels, groups


# ── Per-cluster plots ─────────────────────────────────────────────────────────

def plot_cluster_top3(
    cluster_id: int,
    feat_indices: list[int],
    feat_labels: list[str],
    ranked: list[dict],
    npz,
    pair_indices: list[int],
    out_path: Path,
    concept: str,
    anchor_name: str,
) -> None:
    """Bar chart of top-3 features: sorted pos (blue) then neg (red) activations."""
    # rank within cluster by jaccard×|score|
    rank_map = {f"L{r['layer']}_F{r['feat_id']}": r["jaccard"] * abs(r["score"])
                for r in ranked}
    cluster_feats = sorted(feat_indices,
                           key=lambda i: rank_map.get(feat_labels[i], 0.0),
                           reverse=True)[:3]
    if not cluster_feats:
        return

    n = len(cluster_feats)
    ps.apply()
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), squeeze=False)

    for col, fi in enumerate(cluster_feats):
        key = feat_labels[fi]
        ax  = axes[0][col]
        if key not in npz:
            ax.set_visible(False)
            continue

        arr = npz[key].astype(np.float32)
        pos_all = arr[0::2]
        neg_all = arr[1::2]
        max_i = min(len(pos_all), len(neg_all))
        idx = [i for i in pair_indices if i < max_i]
        if not idx:
            ax.set_visible(False)
            continue

        pos_acts = pos_all[idx]
        neg_acts = neg_all[idx]
        xs = np.arange(len(idx))

        ax.bar(xs, pos_acts, color=ps.NAVY, alpha=0.75, label="pos", width=1.0)
        ax.bar(xs, neg_acts, color=ps.RED,  alpha=0.55, label="neg", width=1.0)
        ax.axhline(pos_acts.mean(), color=ps.NAVY, lw=1.2, ls="--", alpha=0.8)
        ax.axhline(neg_acts.mean(), color=ps.RED,  lw=1.2, ls="--", alpha=0.8)
        ax.axhline(0, color=ps.GRAY, lw=0.6)

        layer = key.split("_")[0][1:]
        feat  = key.split("_")[1][1:]
        jac   = rank_map.get(key, 0.0)
        ax.set_title(f"L{layer} F{feat}\njac×|sc|={jac:.3f}", fontsize=9)
        ax.set_xlabel("example pair (original order)", fontsize=8)
        ax.set_ylabel("activation", fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Cluster {cluster_id} — top-3 features  |  {concept} {anchor_name}",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_pca(
    cluster_id: int,
    D_cluster: np.ndarray,
    P_cluster: np.ndarray,
    N_cluster: np.ndarray,
    labels: np.ndarray,
    label_name: str,
    concept: str,
    anchor_name: str,
    out_path: Path,
    n_pcs: int = 6,
) -> dict:
    """PCA scatter + polar centroid ring + Fourier fits for both Deltas and Raw Activations. Returns summary dict."""
    # ── 1. DELTA PCA ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    D_std  = scaler.fit_transform(D_cluster)
    n_pcs_actual = min(n_pcs, D_cluster.shape[1], D_cluster.shape[0])
    if n_pcs_actual < 2:
        return None, None, None

    if D_std.var(axis=0).sum() < 1e-12:
        return None, None, None   # degenerate cluster
    pca = PCA(n_components=n_pcs_actual, random_state=42)
    Z   = pca.fit_transform(D_std)
    var = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0)
    svs = np.sqrt(np.maximum(pca.explained_variance_, 0.0))
    sv_ratio = float(svs[1] / svs[0]) if svs[0] > 1e-10 else 0.0

    unique_labels = np.sort(np.unique(labels))
    n_cls = len(unique_labels)
    cmap  = plt.get_cmap("tab10", n_cls)

    cents = np.array([[Z[labels == u, 0].mean(), Z[labels == u, 1].mean()]
                      for u in unique_labels])
    centre   = cents.mean(axis=0)
    c_cent   = cents - centre
    radii    = np.linalg.norm(c_cent, axis=1)
    angles   = np.arctan2(c_cent[:, 1], c_cent[:, 0])
    radius_cv = float(np.std(radii) / (np.mean(radii) + 1e-9))
    label_rank  = np.searchsorted(np.sort(unique_labels), unique_labels)
    if n_cls > 2 and np.std(angles) > 1e-10 and np.std(label_rank) > 1e-10:
        pearson_r = float(np.corrcoef(label_rank, angles)[0, 1])
    else:
        pearson_r = float('nan')

    fourier_valid = n_cls > 3
    g = int(unique_labels.max()) + 1
    k_vals = unique_labels.astype(float)
    X_f = np.column_stack([np.cos(2*np.pi*k_vals/g), np.sin(2*np.pi*k_vals/g)])
    n_show = min(n_pcs_actual, 2)
    pc_means = [np.array([Z[labels == u, i].mean() for u in unique_labels])
                for i in range(n_show)]
    pc_r2, pc_coefs = [], []
    for m in pc_means:
        c, *_ = np.linalg.lstsq(X_f, m, rcond=None)
        fit = X_f @ c
        r2 = float(1 - np.var(m - fit) / (np.var(m) + 1e-9)) if fourier_valid else float('nan')
        pc_r2.append(r2)
        pc_coefs.append(c)

    valid_r2 = [r for r in pc_r2 if not math.isnan(r)]
    max_r2 = max(valid_r2) if valid_r2 else float('nan')

    # ── 2. RAW ACTIVATIONS PCA ────────────────────────────────────────────────
    # vstack pos and neg activations
    X_raw = np.vstack([P_cluster, N_cluster])
    labels_raw = np.concatenate([np.zeros(len(labels)), labels])  # 0 for pos, offset 1-6 for neg
    
    scaler_raw = StandardScaler()
    X_raw_std  = scaler_raw.fit_transform(X_raw)
    
    pca_raw = PCA(n_components=n_pcs_actual, random_state=42)
    Z_raw   = pca_raw.fit_transform(X_raw_std)
    var_raw = np.nan_to_num(pca_raw.explained_variance_ratio_, nan=0.0)
    svs_raw = np.sqrt(np.maximum(pca_raw.explained_variance_, 0.0))
    sv_ratio_raw = float(svs_raw[1] / svs_raw[0]) if svs_raw[0] > 1e-10 else 0.0

    unique_labels_raw = np.sort(np.unique(labels_raw))
    n_cls_raw = len(unique_labels_raw)
    cmap_raw  = plt.get_cmap("tab10", n_cls_raw)

    cents_raw = np.array([[Z_raw[labels_raw == u, 0].mean(), Z_raw[labels_raw == u, 1].mean()]
                          for u in unique_labels_raw])
    
    # Polar metrics on the ring (offsets 1-6, excluding 0)
    ring_mask = unique_labels_raw != 0
    if ring_mask.sum() > 2:
        ring_cents = cents_raw[ring_mask]
        ring_labels = unique_labels_raw[ring_mask]
        centre_raw = ring_cents.mean(axis=0)
        c_cent_raw = ring_cents - centre_raw
        radii_raw  = np.linalg.norm(c_cent_raw, axis=1)
        angles_raw = np.arctan2(c_cent_raw[:, 1], c_cent_raw[:, 0])
        radius_cv_raw = float(np.std(radii_raw) / (np.mean(radii_raw) + 1e-9))
        label_rank_raw = np.searchsorted(np.sort(ring_labels), ring_labels)
        if np.std(angles_raw) > 1e-10 and np.std(label_rank_raw) > 1e-10:
            pearson_r_raw = float(np.corrcoef(label_rank_raw, angles_raw)[0, 1])
        else:
            pearson_r_raw = float('nan')
    else:
        centre_raw = cents_raw.mean(axis=0)
        radii_raw  = np.linalg.norm(cents_raw - centre_raw, axis=1)
        angles_raw = np.arctan2(cents_raw[:, 1] - centre_raw[1], cents_raw[:, 0] - centre_raw[0])
        radius_cv_raw = 0.0
        pearson_r_raw = float('nan')

    # Fourier fits for all 7 classes (including 0)
    fourier_valid_raw = n_cls_raw > 3
    g_raw = int(unique_labels_raw.max()) + 1
    k_vals_raw = unique_labels_raw.astype(float)
    X_f_raw = np.column_stack([np.cos(2*np.pi*k_vals_raw/g_raw), np.sin(2*np.pi*k_vals_raw/g_raw)])
    pc_means_raw = [np.array([Z_raw[labels_raw == u, i].mean() for u in unique_labels_raw])
                    for i in range(n_show)]
    pc_r2_raw, pc_coefs_raw = [], []
    for m in pc_means_raw:
        c, *_ = np.linalg.lstsq(X_f_raw, m, rcond=None)
        fit = X_f_raw @ c
        r2 = float(1 - np.var(m - fit) / (np.var(m) + 1e-9)) if fourier_valid_raw else float('nan')
        pc_r2_raw.append(r2)
        pc_coefs_raw.append(c)

    valid_r2_raw = [r for r in pc_r2_raw if not math.isnan(r)]
    max_r2_raw = max(valid_r2_raw) if valid_r2_raw else float('nan')

    # ── 3. PLOTTING (2 rows × 3 columns) ──────────────────────────────────────
    ps.apply()
    fig = plt.figure(figsize=(15, 9.6))
    
    # Row 1: Delta
    ax0 = fig.add_subplot(2, 3, 1)
    ax1 = fig.add_subplot(2, 3, 2, projection="polar")
    ax2 = fig.add_subplot(2, 3, 3)
    
    # Row 2: Raw
    ax3 = fig.add_subplot(2, 3, 4)
    ax4 = fig.add_subplot(2, 3, 5, projection="polar")
    ax5 = fig.add_subplot(2, 3, 6)

    # Panel 1: Delta Scatter
    for k, u in enumerate(unique_labels):
        mask = labels == u
        ax0.scatter(Z[mask, 0], Z[mask, 1], color=cmap(k), alpha=0.35, s=18, zorder=3)
    for k, (cx, cy) in enumerate(cents):
        ax0.scatter(cx, cy, color=cmap(k), s=100, marker="D", zorder=5, edgecolors="white", linewidths=0.9)
        ax0.annotate(str(unique_labels[k]), (cx, cy), textcoords="offset points", xytext=(4, 3),
                     fontsize=7, color=cmap(k), fontweight="bold")
    ax0.plot(np.append(cents[:, 0], cents[0, 0]), np.append(cents[:, 1], cents[0, 1]), color="#555", lw=1.0, ls="--", alpha=0.6)
    ax0.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=9)
    ax0.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=9)
    ax0.set_title(f"Delta PC1 vs PC2  SV2/SV1={sv_ratio:.3f}", fontsize=10)
    from matplotlib.lines import Line2D as _L2D
    ax0.legend(handles=[
        _L2D([0],[0], marker="o", color="#555", linestyle="None", markersize=6, alpha=0.7,
             label=f"Δ activation\n(by {label_name})"),
        _L2D([0],[0], marker="D", color="#555", linestyle="None", markersize=7,
             markeredgecolor="white", markeredgewidth=0.8, label="centroid"),
    ], fontsize=6, loc="best", framealpha=0.85, edgecolor="#ddd")

    # Panel 2: Delta Polar Centroids
    mean_r = float(radii.mean()) if radii.mean() > 0 else 1.0
    for k, (angle, radius, lbl) in enumerate(zip(angles, radii, unique_labels)):
        ax1.scatter(angle, radius, color=cmap(k), s=90, zorder=5)
        ax1.annotate(str(lbl), xy=(angle, radius), xytext=(5, 3), textcoords="offset points",
                     fontsize=8, color=cmap(k), fontweight="bold")
    ax1.plot(np.append(angles, angles[0]), np.append(radii, radii[0]), color="#555", lw=1.0, ls="--", alpha=0.6)
    ideal_th = np.linspace(0, 2*np.pi, n_cls, endpoint=False)
    ax1.plot(np.append(ideal_th, ideal_th[0]), np.full(n_cls + 1, mean_r), color=ps.RED, lw=0.9, ls=":", alpha=0.55, label="ideal")
    r_title = f"Pearson r={pearson_r:.2f}" if not math.isnan(pearson_r) else ""
    ax1.set_title(f"Delta Polar centroids\nR_cv={radius_cv:.3f}  {r_title}", fontsize=9, pad=12)
    ax1.set_rticks([])
    ax1.legend(fontsize=7, loc="upper right")

    # Panel 3: Delta Fourier Fits
    line_colors = [ps.NAVY, ps.TEAL, ps.MAUVE, ps.RED]
    k_fine = np.linspace(float(unique_labels.min()), float(unique_labels.max()), 100)
    for i in range(n_show):
        ax2.plot(unique_labels, pc_means[i], "o", color=line_colors[i], ms=6, zorder=4)
        ax2.plot(unique_labels, pc_means[i], color=line_colors[i], lw=1.0, alpha=0.4)
        if fourier_valid:
            c = pc_coefs[i]
            fit_fine = c[0]*np.cos(2*np.pi*k_fine/g) + c[1]*np.sin(2*np.pi*k_fine/g)
            ax2.plot(k_fine, fit_fine, color=line_colors[i], lw=1.5, ls="--", label=f"PC{i+1} R²={pc_r2[i]:.2f}")
        else:
            ax2.plot([], [], color=line_colors[i], lw=1.5, ls="--", label=f"PC{i+1}")
    ax2.axhline(0, color=ps.GRAY, lw=0.7, ls="--", alpha=0.6)
    ax2.set_xlabel(label_name, fontsize=9)
    ax2.set_ylabel("mean PC score", fontsize=9)
    ax2.set_title(f"Delta Fourier fits (g={g})", fontsize=9)
    ax2.set_xticks(unique_labels)
    ax2.legend(fontsize=7, ncol=2)

    # Panel 4: Raw Scatter
    for k, u in enumerate(unique_labels_raw):
        mask = labels_raw == u
        # mark 0 (pos) as X, 1-6 as circles
        marker = "x" if u == 0 else "o"
        size = 28 if u == 0 else 18
        ax3.scatter(Z_raw[mask, 0], Z_raw[mask, 1], color=cmap_raw(k), alpha=0.35, s=size, marker=marker, zorder=3)
    for k, (cx, cy) in enumerate(cents_raw):
        marker = "X" if unique_labels_raw[k] == 0 else "D"
        ax3.scatter(cx, cy, color=cmap_raw(k), s=100, marker=marker, zorder=5, edgecolors="white", linewidths=0.9)
        ax3.annotate(str(int(unique_labels_raw[k])), (cx, cy), textcoords="offset points", xytext=(4, 3),
                     fontsize=7, color=cmap_raw(k), fontweight="bold")
    # Draw ring line connecting 1 -> 2 -> ... -> 6 -> 1 (skipping 0)
    if ring_mask.sum() > 2:
        ring_cents = cents_raw[ring_mask]
        ax3.plot(np.append(ring_cents[:, 0], ring_cents[0, 0]), np.append(ring_cents[:, 1], ring_cents[0, 1]), color="#555", lw=1.0, ls="--", alpha=0.6)
    ax3.set_xlabel(f"PC1 ({var_raw[0]*100:.1f}%)", fontsize=9)
    ax3.set_ylabel(f"PC2 ({var_raw[1]*100:.1f}%)", fontsize=9)
    ax3.set_title(f"Raw PC1 vs PC2  SV2/SV1={sv_ratio_raw:.3f}", fontsize=10)
    ax3.legend(handles=[
        _L2D([0],[0], marker="x", color="#555", linestyle="None", markersize=7,
             markeredgewidth=1.4, label="pos example\n(concept present)"),
        _L2D([0],[0], marker="o", color="#555", linestyle="None", markersize=6,
             alpha=0.7, label=f"neg example\n(by {label_name})"),
        _L2D([0],[0], marker="D", color="#555", linestyle="None", markersize=7,
             markeredgecolor="white", markeredgewidth=0.8, label="centroid"),
    ], fontsize=6, loc="best", framealpha=0.85, edgecolor="#ddd")

    # Panel 5: Raw Polar Centroids
    mean_r_raw = float(radii_raw.mean()) if radii_raw.mean() > 0 else 1.0
    for k, u in enumerate(unique_labels_raw):
        # find angle/radius from cents_raw relative to centre_raw
        diff = cents_raw[k] - centre_raw
        r = np.linalg.norm(diff)
        angle = np.arctan2(diff[1], diff[0])
        marker = "X" if u == 0 else "o"
        ax4.scatter(angle, r, color=cmap_raw(k), s=120 if u == 0 else 90, marker=marker, zorder=5)
        ax4.annotate(str(int(u)), xy=(angle, r), xytext=(5, 3), textcoords="offset points",
                     fontsize=8, color=cmap_raw(k), fontweight="bold")
    if ring_mask.sum() > 2:
        ax4.plot(np.append(angles_raw, angles_raw[0]), np.append(radii_raw, radii_raw[0]), color="#555", lw=1.0, ls="--", alpha=0.6)
    ideal_th_raw = np.linspace(0, 2*np.pi, ring_mask.sum(), endpoint=False)
    ax4.plot(np.append(ideal_th_raw, ideal_th_raw[0]), np.full(ring_mask.sum() + 1, mean_r_raw), color=ps.RED, lw=0.9, ls=":", alpha=0.55, label="ideal")
    r_title_raw = f"Pearson r={pearson_r_raw:.2f}" if not math.isnan(pearson_r_raw) else ""
    ax4.set_title(f"Raw Polar centroids (excl 0)\nR_cv={radius_cv_raw:.3f}  {r_title_raw}", fontsize=9, pad=12)
    ax4.set_rticks([])
    ax4.legend(fontsize=7, loc="upper right")

    # Panel 6: Raw Fourier Fits
    k_fine_raw = np.linspace(float(unique_labels_raw.min()), float(unique_labels_raw.max()), 100)
    for i in range(n_show):
        ax5.plot(unique_labels_raw, pc_means_raw[i], "o", color=line_colors[i], ms=6, zorder=4)
        ax5.plot(unique_labels_raw, pc_means_raw[i], color=line_colors[i], lw=1.0, alpha=0.4)
        if fourier_valid_raw:
            c = pc_coefs_raw[i]
            fit_fine = c[0]*np.cos(2*np.pi*k_fine_raw/g_raw) + c[1]*np.sin(2*np.pi*k_fine_raw/g_raw)
            ax5.plot(k_fine_raw, fit_fine, color=line_colors[i], lw=1.5, ls="--", label=f"PC{i+1} R²={pc_r2_raw[i]:.2f}")
        else:
            ax5.plot([], [], color=line_colors[i], lw=1.5, ls="--", label=f"PC{i+1}")
    ax5.axhline(0, color=ps.GRAY, lw=0.7, ls="--", alpha=0.6)
    ax5.set_xlabel(f"{label_name} (0=pos)", fontsize=9)
    ax5.set_ylabel("mean PC score", fontsize=9)
    ax5.set_title(f"Raw Fourier fits (g={g_raw})", fontsize=9)
    ax5.set_xticks(unique_labels_raw)
    ax5.legend(fontsize=7, ncol=2)

    r2_str = f"{max_r2:.3f}" if not math.isnan(max_r2) else "n/a"
    fig.suptitle(
        f"Cluster {cluster_id}  ({D_cluster.shape[1]} feats)  —  {concept} {anchor_name}\n"
        f"Delta var(PC1+PC2)={sum(var[:2])*100:.1f}%  |  Raw var(PC1+PC2)={sum(var_raw[:2])*100:.1f}%\n"
        f"Max Delta Fourier R²={r2_str}  |  Max Raw Fourier R²={max_r2_raw:.3f}",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "cluster_id": cluster_id,
        "n_features": D_cluster.shape[1],
        "sv_ratio_12": round(sv_ratio, 4),
        "var_pc1": round(float(var[0]), 4),
        "var_pc12": round(float(sum(var[:2])), 4),
        "max_fourier_r2": round(max_r2, 4) if not math.isnan(max_r2) else float('nan'),
        "best_pc": int(np.argmax([r for r in pc_r2 if not math.isnan(r)] or [0])) + 1,
        "fourier_valid": fourier_valid,
        "pearson_r": round(pearson_r, 4) if not math.isnan(pearson_r) else float('nan'),
    }
    pca_state = {
        "Z": Z,
        "var": var,
        "labels": labels,
        "unique_labels": unique_labels,
        "fourier_valid": fourier_valid,
        "g": int(g),
    }
    pca_state_raw = {
        "Z": Z_raw,
        "var": var_raw,
        "labels": labels_raw,
        "unique_labels": unique_labels_raw,
        "fourier_valid": fourier_valid_raw,
        "g": int(g_raw),
    }
    return summary, pca_state, pca_state_raw


# ── Extra PC-pair plots ───────────────────────────────────────────────────────

def _pca_pair_panels(
    ax_scatter, ax_polar, ax_fourier,
    pci: int, pcj: int,
    Z: np.ndarray, var: np.ndarray,
    labels: np.ndarray, unique_labels: np.ndarray,
    fourier_valid: bool, g: int,
    label_name: str,
    row_label: str,
    is_raw: bool = False,
) -> None:
    """Fill one row of 3 panels for a given PC pair. Shared by delta and raw rows."""
    i0, j0 = pci - 1, pcj - 1
    n_cls  = len(unique_labels)
    cmap   = plt.get_cmap("tab10", n_cls)
    line_colors = [ps.NAVY, ps.TEAL]

    # centroids
    cents  = np.array([[Z[labels == u, i0].mean(), Z[labels == u, j0].mean()]
                       for u in unique_labels])

    # for the polar ring, exclude class 0 (pos) on the raw row
    ring_mask  = (unique_labels != 0) if is_raw else np.ones(n_cls, dtype=bool)
    ring_cents = cents[ring_mask]
    ring_ulbls = unique_labels[ring_mask]

    if ring_cents.shape[0] > 0:
        centre = ring_cents.mean(axis=0)
    else:
        centre = cents.mean(axis=0)
    c_cent    = ring_cents - centre
    radii     = np.linalg.norm(c_cent, axis=1)
    angles    = np.arctan2(c_cent[:, 1], c_cent[:, 0])
    radius_cv = float(np.std(radii) / (np.mean(radii) + 1e-9))
    lr        = np.searchsorted(np.sort(ring_ulbls), ring_ulbls)
    if len(ring_ulbls) > 2 and np.std(angles) > 1e-10 and np.std(lr) > 1e-10:
        pearson_r = float(np.corrcoef(lr, angles)[0, 1])
    else:
        pearson_r = float("nan")

    # Fourier
    k_vals = unique_labels.astype(float)
    X_f    = np.column_stack([np.cos(2 * np.pi * k_vals / g),
                               np.sin(2 * np.pi * k_vals / g)])
    pc_means, pc_r2, pc_coefs = [], [], []
    for idx in (i0, j0):
        m = np.array([Z[labels == u, idx].mean() for u in unique_labels])
        c, *_ = np.linalg.lstsq(X_f, m, rcond=None)
        fit   = X_f @ c
        r2    = float(1 - np.var(m - fit) / (np.var(m) + 1e-9)) if fourier_valid else float("nan")
        pc_means.append(m); pc_r2.append(r2); pc_coefs.append(c)

    # ── scatter ──────────────────────────────────────────────────────────────
    for k, u in enumerate(unique_labels):
        mask   = labels == u
        marker = "x" if (is_raw and u == 0) else "o"
        size   = 28  if (is_raw and u == 0) else 18
        ax_scatter.scatter(Z[mask, i0], Z[mask, j0], color=cmap(k),
                           alpha=0.35, s=size, marker=marker, zorder=3)
    for k, (cx, cy) in enumerate(cents):
        mk = "X" if (is_raw and unique_labels[k] == 0) else "D"
        ax_scatter.scatter(cx, cy, color=cmap(k), s=100, marker=mk,
                           zorder=5, edgecolors="white", linewidths=0.9)
        ax_scatter.annotate(str(unique_labels[k]), (cx, cy),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=7, color=cmap(k), fontweight="bold")
    if ring_cents.shape[0] > 1:
        ax_scatter.plot(np.append(ring_cents[:, 0], ring_cents[0, 0]),
                        np.append(ring_cents[:, 1], ring_cents[0, 1]),
                        color="#555", lw=1.0, ls="--", alpha=0.6)
    sv_ij = float(np.sqrt(max(var[j0], 0.0)) / (np.sqrt(max(var[i0], 0.0)) + 1e-10))
    ax_scatter.set_xlabel(f"PC{pci} ({var[i0]*100:.1f}%)", fontsize=9)
    ax_scatter.set_ylabel(f"PC{pcj} ({var[j0]*100:.1f}%)", fontsize=9)
    ax_scatter.set_title(f"{row_label} PC{pci} vs PC{pcj}  SV{pcj}/SV{pci}={sv_ij:.3f}",
                         fontsize=10)

    # ── scatter legend ────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    if is_raw:
        legend_handles = [
            Line2D([0], [0], marker="x", color="#555", linestyle="None",
                   markersize=7, markeredgewidth=1.4,
                   label="pos example\n(concept present)"),
            Line2D([0], [0], marker="o", color="#555", linestyle="None",
                   markersize=6, alpha=0.7,
                   label=f"neg example\n(by {label_name})"),
            Line2D([0], [0], marker="D", color="#555", linestyle="None",
                   markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                   label="centroid"),
        ]
    else:
        legend_handles = [
            Line2D([0], [0], marker="o", color="#555", linestyle="None",
                   markersize=6, alpha=0.7,
                   label=f"Δ activation\n(by {label_name})"),
            Line2D([0], [0], marker="D", color="#555", linestyle="None",
                   markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                   label="centroid"),
        ]
    ax_scatter.legend(handles=legend_handles, fontsize=6, loc="best",
                      framealpha=0.85, edgecolor="#ddd")

    # ── polar ─────────────────────────────────────────────────────────────────
    mean_r = float(radii.mean()) if radii.mean() > 0 else 1.0
    # plot all centroids; ring centroids get the polar metrics treatment
    for k, u in enumerate(unique_labels):
        diff   = cents[k] - centre
        r      = float(np.linalg.norm(diff))
        angle  = float(np.arctan2(diff[1], diff[0]))
        mk     = "X" if (is_raw and u == 0) else "o"
        sz     = 120 if (is_raw and u == 0) else 90
        ax_polar.scatter(angle, r, color=cmap(k), s=sz, marker=mk, zorder=5)
        ax_polar.annotate(str(u), xy=(angle, r), xytext=(5, 3),
                          textcoords="offset points", fontsize=8,
                          color=cmap(k), fontweight="bold")
    if len(radii) > 1:
        ax_polar.plot(np.append(angles, angles[0]), np.append(radii, radii[0]),
                      color="#555", lw=1.0, ls="--", alpha=0.6)
    n_ring = ring_mask.sum()
    ideal_th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    ax_polar.plot(np.append(ideal_th, ideal_th[0]), np.full(n_ring + 1, mean_r),
                  color=ps.RED, lw=0.9, ls=":", alpha=0.55, label="ideal")
    r_title = f"Pearson r={pearson_r:.2f}" if not math.isnan(pearson_r) else ""
    excl    = " (excl 0)" if is_raw else ""
    ax_polar.set_title(
        f"{row_label} Polar centroids{excl}\nR_cv={radius_cv:.3f}  {r_title}",
        fontsize=9, pad=12,
    )
    ax_polar.set_rticks([])
    ax_polar.legend(fontsize=7, loc="upper right")

    # ── Fourier ───────────────────────────────────────────────────────────────
    k_fine = np.linspace(float(unique_labels.min()), float(unique_labels.max()), 100)
    for pc_idx, m, c, r2, color in zip((pci, pcj), pc_means, pc_coefs, pc_r2, line_colors):
        ax_fourier.plot(unique_labels, m, "o", color=color, ms=6, zorder=4)
        ax_fourier.plot(unique_labels, m, color=color, lw=1.0, alpha=0.4)
        if fourier_valid:
            fit_fine = c[0]*np.cos(2*np.pi*k_fine/g) + c[1]*np.sin(2*np.pi*k_fine/g)
            ax_fourier.plot(k_fine, fit_fine, color=color, lw=1.5, ls="--",
                            label=f"PC{pc_idx} R²={r2:.2f}")
        else:
            ax_fourier.plot([], [], color=color, lw=1.5, ls="--", label=f"PC{pc_idx}")
    ax_fourier.axhline(0, color=ps.GRAY, lw=0.7, ls="--", alpha=0.6)
    xlabel = f"{label_name} (0=pos)" if is_raw else label_name
    ax_fourier.set_xlabel(xlabel, fontsize=9)
    ax_fourier.set_ylabel("mean PC score", fontsize=9)
    ax_fourier.set_title(f"{row_label} Fourier fits PC{pci} & PC{pcj} (g={g})", fontsize=9)
    ax_fourier.set_xticks(unique_labels)
    ax_fourier.legend(fontsize=7, ncol=2)


def plot_pca_pair_extra(
    pci: int,
    pcj: int,
    pca_state: dict,
    pca_state_raw: dict,
    label_name: str,
    concept: str,
    anchor_name: str,
    cluster_id: int,
    out_path: Path,
) -> None:
    """2-row × 3-col PCA plot (delta row + raw row) for an arbitrary (pci, pcj) pair."""
    n_pcs     = pca_state["Z"].shape[1]
    n_pcs_raw = pca_state_raw["Z"].shape[1]
    if pci > n_pcs or pcj > n_pcs or pci > n_pcs_raw or pcj > n_pcs_raw:
        return

    ps.apply()
    fig = plt.figure(figsize=(15, 9.6))
    ax0 = fig.add_subplot(2, 3, 1)
    ax1 = fig.add_subplot(2, 3, 2, projection="polar")
    ax2 = fig.add_subplot(2, 3, 3)
    ax3 = fig.add_subplot(2, 3, 4)
    ax4 = fig.add_subplot(2, 3, 5, projection="polar")
    ax5 = fig.add_subplot(2, 3, 6)

    _pca_pair_panels(ax0, ax1, ax2, pci, pcj,
                     pca_state["Z"], pca_state["var"],
                     pca_state["labels"], pca_state["unique_labels"],
                     pca_state["fourier_valid"], pca_state["g"],
                     label_name, row_label="Delta", is_raw=False)

    _pca_pair_panels(ax3, ax4, ax5, pci, pcj,
                     pca_state_raw["Z"], pca_state_raw["var"],
                     pca_state_raw["labels"], pca_state_raw["unique_labels"],
                     pca_state_raw["fourier_valid"], pca_state_raw["g"],
                     label_name, row_label="Raw", is_raw=True)

    fig.suptitle(
        f"Cluster {cluster_id}  —  {concept} {anchor_name}  (PC{pci} vs PC{pcj})",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Cosine heatmap ────────────────────────────────────────────────────────────

def plot_cosine_heatmap(cos_sim, cluster_labels, feat_labels, out_path, title):
    sort_idx = np.argsort(cluster_labels)
    C = cos_sim[np.ix_(sort_idx, sort_idx)]
    ps.apply()
    fig, ax = plt.subplots(figsize=(min(12, 0.12*len(feat_labels)+4),
                                    min(10, 0.12*len(feat_labels)+3)))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.75, label="cosine similarity")
    for b in np.where(np.diff(cluster_labels[sort_idx]))[0] + 0.5:
        ax.axhline(b, color="black", lw=0.6, alpha=0.7)
        ax.axvline(b, color="black", lw=0.6, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("feature index (sorted by cluster)", fontsize=8)
    ax.set_ylabel("feature index (sorted by cluster)", fontsize=8)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main analysis entry ───────────────────────────────────────────────────────

def run_analysis(
    sweep_dir: Path,
    results_json: Path | None = None,
    template: str | None = "T0",
    top_k: int = 100,
    n_clusters: int = 6,
    n_pcs: int = 6,
    out_dir: Path | None = None,
) -> list[dict]:
    if out_dir is None:
        tag = template if template else "all"
        out_dir = sweep_dir / f"cluster_analysis_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if results_json is None:
        results_json = sweep_dir.parent / "results.json"

    ranked, npz, examples, pair_indices = load_sweep(sweep_dir, template)
    if not pair_indices:
        print(f"  [skip] no examples for template={template} in {sweep_dir}")
        return []

    peak_layers = None
    if results_json.exists():
        res = json.loads(results_json.read_text())
        peak_layers = _peak_layers(res)

    D, feat_labels = build_delta_matrix(ranked, npz, pair_indices, top_k, peak_layers)
    if D.size == 0:
        print(f"  [skip] empty delta matrix in {sweep_dir}")
        return []

    labels, label_name = extract_labels(examples, pair_indices)
    # trim to same length as D rows
    labels = labels[:D.shape[0]]

    n_clusters = min(n_clusters, D.shape[1])
    concept    = sweep_dir.parent.parent.name
    anchor     = sweep_dir.parent.name

    print(f"  {concept}/{anchor}  D={D.shape}  "
          f"peak_layers={sorted(peak_layers) if peak_layers else 'all'}  "
          f"label='{label_name}'  n_cls={len(np.unique(labels))}")

    cos_sim, cluster_labels, groups = cluster_features(D, n_clusters)

    plot_cosine_heatmap(cos_sim, cluster_labels, feat_labels,
                        out_dir / "cosine_similarity.png",
                        f"{concept} {anchor} — feature cosine similarity")
    print(f"    cosine heatmap saved")

    rank_map = {f"L{r['layer']}_F{r['feat_id']}": r["jaccard"] * abs(r["score"]) for r in ranked}
    cluster_features_json: dict[str, list[str]] = {}

    summaries = []
    for group in groups:
        c_id   = int(cluster_labels[group[0]])
        layers = sorted(set(int(feat_labels[i].split("_")[0][1:]) for i in group))
        print(f"    cluster {c_id}: {len(group)} features  layers {layers}")

        top3_indices = sorted(group, key=lambda i: rank_map.get(feat_labels[i], 0.0), reverse=True)[:3]
        cluster_features_json[f"cluster_{c_id:02d}"] = [feat_labels[i] for i in top3_indices]

        # top-3 bar chart
        plot_cluster_top3(
            c_id, group, feat_labels, ranked, npz, pair_indices,
            out_dir / f"cluster_{c_id:02d}_top3.png",
            concept, anchor,
        )

        # PCA
        D_c = D[:, group]
        
        # build P_c and N_c
        cols_pos, cols_neg = [], []
        for i in group:
            key = feat_labels[i]
            arr = npz[key].astype(np.float32)
            pos_all = arr[0::2]
            neg_all = arr[1::2]
            max_i = min(len(pos_all), len(neg_all))
            idx = [p for p in pair_indices if p < max_i]
            cols_pos.append(pos_all[idx])
            cols_neg.append(neg_all[idx])
        P_c = np.column_stack(cols_pos)
        N_c = np.column_stack(cols_neg)

        s, pca_state, pca_state_raw = plot_cluster_pca(
            c_id, D_c, P_c, N_c, labels, label_name,
            concept, anchor,
            out_dir / f"cluster_{c_id:02d}_pca.png",
            n_pcs=n_pcs,
        )
        if s is None:
            continue
        summaries.append(s)

        passes, checks = passes_thresholds(s)
        if not passes:
            metrics_path = out_dir / f"cluster_{c_id:02d}_pca_extra_metrics.json"
            metrics_path.write_text(json.dumps(
                {**s, "threshold_checks": checks, "thresholds": THRESHOLDS},
                indent=2,
            ))
            print(f"      cluster {c_id}: extra PCA skipped (thresholds not met) → {metrics_path.name}")
        else:
            for pci, pcj in EXTRA_PC_PAIRS:
                out_p = out_dir / f"cluster_{c_id:02d}_pca_pc{pci}_pc{pcj}.png"
                plot_pca_pair_extra(pci, pcj, pca_state, pca_state_raw,
                                    label_name, concept, anchor, c_id, out_p)
            print(f"      cluster {c_id}: extra PCA plots saved for pairs {EXTRA_PC_PAIRS}")

    cf_path = out_dir / "cluster_features.json"
    with cf_path.open("w") as f:
        json.dump(cluster_features_json, f, indent=2)
    print(f"    cluster_features.json saved → {cf_path}")

    if summaries:
        print(f"\n    {'C':>4}  {'N':>4}  {'SV2/1':>6}  "
              f"{'var12%':>7}  {'maxR²':>6}  {'bestPC':>7}")
        for s in sorted(summaries, key=lambda x: x["max_fourier_r2"], reverse=True):
            print(f"    {s['cluster_id']:>4}  {s['n_features']:>4}  "
                  f"{s['sv_ratio_12']:>6.3f}  {s['var_pc12']*100:>6.1f}%  "
                  f"{s['max_fourier_r2']:>6.3f}  PC{s['best_pc']:>2}")

    return summaries


# ── CLI ───────────────────────────────────────────────────────────────────────

def _anchor_dirs(concept_dir: Path) -> list[Path]:
    return sorted(concept_dir.glob("anchor_rank*_pos*"))


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sweep_dir", default=None,
                        help="Single sweep directory to analyse")
    parser.add_argument("--concept", nargs="*", default=None,
                        help="Concept name(s) — analyses rank-1 anchor for each")
    parser.add_argument("--all", action="store_true",
                        help="Run for all concepts, all anchors")
    parser.add_argument("--template", default="T0",
                        help="Template filter (T0/T1/T2/None for all)")
    parser.add_argument("--top_k",     type=int, default=100)
    parser.add_argument("--n_clusters",type=int, default=6)
    parser.add_argument("--n_pcs",     type=int, default=6)
    args = parser.parse_args()

    tmpl = None if args.template.lower() == "none" else args.template

    if args.sweep_dir:
        run_analysis(Path(args.sweep_dir), template=tmpl,
                     top_k=args.top_k, n_clusters=args.n_clusters, n_pcs=args.n_pcs)
        return

    if args.all:
        concept_dirs = sorted(d for d in _BASE.iterdir()
                              if d.is_dir() and any(d.glob("anchor_rank*_pos*")))
    elif args.concept:
        concept_dirs = [_BASE / c for c in args.concept if (_BASE / c).is_dir()]
    else:
        parser.print_help()
        return

    for cdir in concept_dirs:
        for anchor_dir in _anchor_dirs(cdir):
            sweep_dir = anchor_dir / "sweep"
            if not (sweep_dir / "sweep_ranked.json").exists():
                continue
            print(f"\n{'─'*60}")
            run_analysis(sweep_dir, template=tmpl,
                         top_k=args.top_k, n_clusters=args.n_clusters, n_pcs=args.n_pcs)


if __name__ == "__main__":
    main()
