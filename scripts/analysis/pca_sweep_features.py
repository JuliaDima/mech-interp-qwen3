"""Geometric analysis of sweep features: PCA, circular structure, modular manifolds.

Loads transcoder feature activations from a concept sweep, builds the delta matrix
D[i] = pos_act[i] - neg_act[i], and probes for low-dimensional and ring-like structure
via PCA.  Designed for concepts whose labels form a cyclic group (e.g. gcd with mod-g
classes).

Usage
-----
    python scripts/analysis/pca_sweep_features.py \\
        --sweep_dir runs/concept_localization/gcd/anchor_rank1_pos6/sweep \\
        --top_k 100 --n_pcs 6

    # use peak layers from results.json (recommended)
    python scripts/analysis/pca_sweep_features.py \\
        --sweep_dir runs/concept_localization/gcd/anchor_rank1_pos6/sweep \\
        --results_json runs/concept_localization/gcd/anchor_rank1_pos6/results.json \\
        --top_k 80
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import experiments.plot_style as ps

# ── Peak-layer helpers ────────────────────────────────────────────────────────

def _peak_layers_from_results(results: dict, thresh: float = 0.75, n_layers: int = 36) -> set[int]:
    # norm_by_layer already stores delta_norm / mean_act_norm (activation-normalised)
    # when normalised=True, so use it directly as the double-normalised trajectory.
    norms = {int(k): v for k, v in results["sharpness"]["norm_by_layer"].items()}
    ls = sorted(norms)
    vs = np.array([norms[l] for l in ls], dtype=float)
    vs_n = vs / (vs.max() + 1e-12)
    maxima: list[tuple[int, float]] = []
    for i, l in enumerate(ls):
        left  = vs_n[i - 1] if i > 0 else -1.0
        right = vs_n[i + 1] if i < len(ls) - 1 else -1.0
        if vs_n[i] > left and vs_n[i] > right:
            maxima.append((l, float(vs_n[i])))
    maxima.sort(key=lambda x: -x[1])
    sel = [maxima[0][0]] if maxima else []
    if len(maxima) > 1 and maxima[1][1] >= thresh:
        sel.append(maxima[1][0])
    out: set[int] = set()
    for p in sel:
        for o in range(-1, 3):
            l = p + o
            if 0 <= l < n_layers:
                out.add(l)
    return out


# ── Data loading ──────────────────────────────────────────────────────────────

def load_delta_matrix(
    sweep_dir: Path,
    ranked: list[dict],
    npz,
    top_k: int,
    peak_layers: set[int] | None,
) -> tuple[np.ndarray, list[str]]:
    """Build (n_examples, n_features) delta matrix from top_k ranked features.

    Returns (D, feat_labels) where D[i] = pos_act[i] - neg_act[i].
    """
    pool = [r for r in ranked if (peak_layers is None or r["layer"] in peak_layers)]
    if not pool:
        pool = ranked  # fallback: use all layers
    selected = pool[:top_k]

    cols = []
    labels = []
    for r in selected:
        key = f"L{r['layer']}_F{r['feat_id']}"
        if key not in npz:
            continue
        arr = npz[key].astype(np.float32)  # shape (2*n_examples,)
        n = len(arr) // 2
        pos = arr[0::2][:n]
        neg = arr[1::2][:n]
        cols.append(pos - neg)
        labels.append(key)

    if not cols:
        raise RuntimeError("No features loaded — check sweep_dir and top_k")

    D = np.column_stack(cols)  # (n_examples, n_features)
    return D, labels


def extract_labels(examples: list[dict]) -> tuple[np.ndarray, str]:
    """Return (label_array, label_name) from example metadata."""
    meta0 = examples[0]["meta"]
    if "offset" in meta0:
        arr = np.array([e["meta"]["offset"] for e in examples], dtype=int)
        return arr, "offset (a_neg mod g)"
    if "carry" in meta0:
        arr = np.array([int(e["meta"].get("carry", 0)) for e in examples], dtype=int)
        return arr, "carry"
    # fallback: a_pos mod some small number
    vals = np.array([e["meta"].get("a_pos", 0) for e in examples], dtype=int)
    return vals % 10, "a_pos mod 10"


# ── Circular structure detection ──────────────────────────────────────────────

def circular_diagnostics(
    pcs: np.ndarray,     # (n, 2) — first two PCs
    labels: np.ndarray,  # (n,) integer label
    ax_scatter: plt.Axes,
    ax_angle: plt.Axes,
    title: str,
    label_name: str,
    cmap_name: str = "tab10",
) -> dict:
    """Plot scatter with centroids + angular correlation diagnostics.

    Returns dict with radius_cv, angle_spacing_std, angle_pearson_r.
    """
    unique_labels = np.sort(np.unique(labels))
    n_cls = len(unique_labels)
    cmap = plt.get_cmap(cmap_name, n_cls)

    # ── Scatter ──────────────────────────────────────────────────────────────
    for i, u in enumerate(unique_labels):
        mask = labels == u
        ax_scatter.scatter(
            pcs[mask, 0], pcs[mask, 1],
            color=cmap(i), alpha=0.45, s=22, label=f"{label_name.split()[0]}={u}",
            zorder=3,
        )

    # Centroids
    centroids = np.array([pcs[labels == u].mean(axis=0) for u in unique_labels])
    for i, (cx, cy) in enumerate(centroids):
        ax_scatter.scatter(cx, cy, color=cmap(i), s=90, marker="D", zorder=5,
                           edgecolors="white", linewidths=0.8)

    # Ring: connect centroids in label order
    ring_x = np.append(centroids[:, 0], centroids[0, 0])
    ring_y = np.append(centroids[:, 1], centroids[0, 1])
    ax_scatter.plot(ring_x, ring_y, color="#555555", lw=1.0, ls="--",
                    alpha=0.6, zorder=4)

    ax_scatter.set_title(title, fontsize=10)
    ax_scatter.set_xlabel("PC1", fontsize=9)
    ax_scatter.set_ylabel("PC2", fontsize=9)
    ax_scatter.legend(fontsize=7, ncol=2, loc="best")

    # ── Angular diagnostics ──────────────────────────────────────────────────
    # centre centroids
    centre = centroids.mean(axis=0)
    c_centred = centroids - centre
    radii = np.linalg.norm(c_centred, axis=1)
    angles = np.arctan2(c_centred[:, 1], c_centred[:, 0])  # in [−π, π]

    # sort by label to correlate with residue class
    sort_order = np.argsort(angles)
    sorted_labels = unique_labels[sort_order]
    sorted_angles = np.degrees(angles[sort_order])

    # Ideal equally-spaced angles for a regular n-gon
    ideal_spacing = 360.0 / n_cls
    angle_diffs = np.diff(np.sort(np.degrees(angles) % 360.0))
    spacing_std = float(np.std(angle_diffs))

    radius_cv = float(np.std(radii) / (np.mean(radii) + 1e-9))  # coeff of variation

    # Pearson between label rank and angle (phase correlation)
    label_rank = np.searchsorted(np.sort(unique_labels), unique_labels)
    # unwrap angles to be monotone if possible
    pearson_r = float(np.corrcoef(label_rank, angles)[0, 1])

    # Plot centroids in polar style on ax_angle
    theta_rad = np.append(np.sort(angles % (2 * math.pi)),
                           (np.sort(angles % (2 * math.pi)))[0])
    ax_angle.plot(
        sorted(angles % (2 * math.pi)),
        radii[sort_order],
        "o-", color=ps.NAVY, lw=1.2, ms=6,
    )
    for i, (angle, radius, lbl) in enumerate(
        zip(angles % (2 * math.pi), radii, unique_labels)
    ):
        ax_angle.annotate(
            str(lbl), (angle, radius),
            textcoords="offset points", xytext=(4, 4), fontsize=7,
        )
    ax_angle.set_xlabel("centroid angle (rad)", fontsize=9)
    ax_angle.set_ylabel("radius from centroid mean", fontsize=9)
    ax_angle.set_title("centroid ring diagnostics", fontsize=9)

    return {
        "radius_cv": radius_cv,
        "angle_spacing_std": spacing_std,
        "ideal_spacing": ideal_spacing,
        "angle_pearson_r": pearson_r,
        "centroid_radii": radii.tolist(),
        "centroid_angles_deg": np.degrees(angles).tolist(),
        "sorted_labels": sorted_labels.tolist(),
        "sorted_angles_deg": sorted_angles.tolist(),
    }


# ── Feature clustering by cosine similarity ───────────────────────────────────

def cluster_features_by_cosine(
    D: np.ndarray,
    n_clusters: int,
    min_cluster_size: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Cluster features (columns of D) by cosine similarity.

    Returns
    -------
    cos_sim  : (n_features, n_features) cosine similarity matrix
    labels   : (n_features,) cluster assignment per feature
    groups   : list of feature-index lists, one per cluster with >= min_cluster_size members
    """
    from sklearn.cluster import AgglomerativeClustering

    # L2-normalise each feature vector (column) so dot-product = cosine similarity
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    D_norm = D / norms                          # (n_examples, n_features)
    cos_sim = D_norm.T @ D_norm                 # (n_features, n_features)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    dist = 1.0 - cos_sim                        # cosine distance in [0, 2]
    dist = np.clip(dist, 0.0, None)             # numerical safety

    clust = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = clust.fit_predict(dist)

    groups = []
    for c in range(n_clusters):
        idx = np.where(labels == c)[0].tolist()
        if len(idx) >= min_cluster_size:
            groups.append(idx)

    return cos_sim, labels, groups


def plot_cosine_heatmap(
    cos_sim: np.ndarray,
    cluster_labels: np.ndarray,
    feat_labels: list[str],
    out_path: Path,
    anchor_name: str,
) -> None:
    """Heatmap of pairwise cosine similarity, rows/cols sorted by cluster."""
    n = len(cluster_labels)
    sort_idx = np.argsort(cluster_labels)
    C_sorted = cos_sim[np.ix_(sort_idx, sort_idx)]

    ps.apply()
    fig, ax = plt.subplots(figsize=(min(14, 0.12 * n + 4), min(12, 0.12 * n + 3)))
    im = ax.imshow(C_sorted, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")

    # mark cluster boundaries
    boundaries = np.where(np.diff(cluster_labels[sort_idx]))[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="black", lw=0.6, alpha=0.7)
        ax.axvline(b, color="black", lw=0.6, alpha=0.7)

    ax.set_title(f"Feature cosine similarity — {anchor_name}\n(sorted by cluster)", fontsize=10)
    ax.set_xlabel("feature index (sorted)", fontsize=9)
    ax.set_ylabel("feature index (sorted)", fontsize=9)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _pca_cluster(
    D_cluster: np.ndarray,
    labels: np.ndarray,
    label_name: str,
    n_pcs: int,
    cluster_id: int,
    feat_indices: list[int],
    out_dir: Path,
    anchor_name: str,
    no_plots: bool = False,
) -> dict:
    """Run PCA + ring/Fourier diagnostics for one cluster. Returns diagnostic dict."""
    from sklearn.preprocessing import StandardScaler

    n_feats = D_cluster.shape[1]
    scaler = StandardScaler()
    D_std = scaler.fit_transform(D_cluster)

    n_pcs_actual = min(n_pcs, n_feats, D_std.shape[0])
    if n_pcs_actual < 2:
        print(f"    cluster {cluster_id}: too few features ({n_feats}) for PCA — skip")
        return {}

    pca = PCA(n_components=n_pcs_actual, random_state=42)
    Z = pca.fit_transform(D_std)
    var_ratio = pca.explained_variance_ratio_
    sing_vals = np.sqrt(pca.explained_variance_)
    ratio_12 = sing_vals[1] / sing_vals[0] if len(sing_vals) > 1 and sing_vals[0] > 0 else 0.0

    unique_labels = np.sort(np.unique(labels))
    n_cls = len(unique_labels)
    cmap = plt.get_cmap("tab10", n_cls)

    # ── Centroids and ring diagnostics ───────────────────────────────────────
    means_pc1 = np.array([Z[labels == u, 0].mean() for u in unique_labels])
    means_pc2 = np.array([Z[labels == u, 1].mean() for u in unique_labels])
    centroids = np.column_stack([means_pc1, means_pc2])
    centre = centroids.mean(axis=0)
    c_centred = centroids - centre
    radii  = np.linalg.norm(c_centred, axis=1)
    angles = np.arctan2(c_centred[:, 1], c_centred[:, 0])

    radius_cv    = float(np.std(radii) / (np.mean(radii) + 1e-9))
    angle_diffs  = np.diff(np.sort(np.degrees(angles) % 360.0))
    spacing_std  = float(np.std(angle_diffs))
    ideal_spacing = 360.0 / n_cls
    label_rank   = np.searchsorted(np.sort(unique_labels), unique_labels)
    pearson_r    = float(np.corrcoef(label_rank, angles)[0, 1])

    # ── Fourier fit for all PCs: y = a·cos(2πk/g) + b·sin(2πk/g) ────────────
    g      = int(unique_labels.max()) + 1
    k_vals = unique_labels.astype(float)
    cos_b  = np.cos(2 * np.pi * k_vals / g)
    sin_b  = np.sin(2 * np.pi * k_vals / g)
    X_f    = np.column_stack([cos_b, sin_b])

    n_show = min(n_pcs_actual, 4)
    pc_means  = [np.array([Z[labels == u, i].mean() for u in unique_labels])
                 for i in range(n_show)]
    pc_coefs  = [np.linalg.lstsq(X_f, m, rcond=None)[0] for m in pc_means]
    pc_fits   = [X_f @ c for c in pc_coefs]
    pc_r2     = [float(1 - np.var(m - f) / (np.var(m) + 1e-9))
                 for m, f in zip(pc_means, pc_fits)]
    pc_phases = [float(np.degrees(np.arctan2(c[1], c[0]))) for c in pc_coefs]

    # find best quadrature pair: highest min(R²) with phase diff closest to 90°
    best_pair, best_score = (0, 1), -1.0
    pair_stats = {}
    for i in range(n_show):
        for j in range(i + 1, n_show):
            pd = float(abs((pc_phases[i] - pc_phases[j] + 180) % 360 - 180))
            score = min(pc_r2[i], pc_r2[j]) * (1 - abs(pd - 90) / 90)
            pair_stats[(i, j)] = (pd, score)
            if score > best_score:
                best_score, best_pair = score, (i, j)
    bi, bj = best_pair
    best_phase_diff = pair_stats[best_pair][0]

    # r2 for PC1/PC2 for backwards compat with summary table
    r2_pc1, r2_pc2 = pc_r2[0], pc_r2[1]
    phase_diff = float(abs((pc_phases[0] - pc_phases[1] + 180) % 360 - 180))

    print(f"\n  Cluster {cluster_id}  ({n_feats} features):")
    print(f"    SV2/SV1={ratio_12:.4f}  Pearson_r={pearson_r:.3f}")
    for i in range(n_show):
        print(f"    PC{i+1}: R²={pc_r2[i]:.3f}  phase={pc_phases[i]:.1f}°")
    print(f"    Best quadrature pair: PC{bi+1}–PC{bj+1}  "
          f"phase_diff={best_phase_diff:.1f}°  score={best_score:.3f}")

    if not no_plots:
        # ── Plot: 4-panel layout ──────────────────────────────────────────────────
        ps.apply()
        line_colors = [ps.NAVY, ps.TEAL, ps.MAUVE, ps.RED]
        markers     = ["o", "s", "^", "D"]

        fig = plt.figure(figsize=(20, 5.2))
        ax0 = fig.add_subplot(1, 4, 1)
        ax1 = fig.add_subplot(1, 4, 2, projection="polar")
        ax2 = fig.add_subplot(1, 4, 3)
        ax3 = fig.add_subplot(1, 4, 4)

        # Panel 1: scatter PC1 vs PC2 with centroids
        for k, u in enumerate(unique_labels):
            mask = labels == u
            ax0.scatter(Z[mask, 0], Z[mask, 1], color=cmap(k), alpha=0.35, s=18, zorder=3)
        for k, (cx, cy) in enumerate(centroids):
            ax0.scatter(cx, cy, color=cmap(k), s=110, marker="D", zorder=5,
                        edgecolors="white", linewidths=0.9)
            ax0.annotate(str(unique_labels[k]), (cx, cy),
                         textcoords="offset points", xytext=(4, 3),
                         fontsize=7, color=cmap(k), fontweight="bold")
        cx_ring = np.append(centroids[:, 0], centroids[0, 0])
        cy_ring = np.append(centroids[:, 1], centroids[0, 1])
        ax0.plot(cx_ring, cy_ring, color="#555555", lw=1.0, ls="--", alpha=0.6)
        ax0.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=9)
        ax0.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=9)
        ax0.set_title(f"PC1 vs PC2  (SV2/SV1={ratio_12:.3f})", fontsize=10)

        # Panel 2: polar centroid plot
        mean_r = float(radii.mean()) if radii.mean() > 0 else 1.0
        for k, (angle, radius, lbl) in enumerate(zip(angles, radii, unique_labels)):
            ax1.scatter(angle, radius, color=cmap(k), s=90, zorder=5)
            ax1.annotate(str(lbl), xy=(angle, radius), xytext=(5, 3),
                         textcoords="offset points", fontsize=8,
                         color=cmap(k), fontweight="bold")
        ax1.plot(np.append(angles, angles[0]), np.append(radii, radii[0]),
                 color="#555555", lw=1.0, ls="--", alpha=0.6)
        ideal_th = np.linspace(0, 2 * np.pi, n_cls, endpoint=False)
        ax1.plot(np.append(ideal_th, ideal_th[0]), np.full(n_cls + 1, mean_r),
                 color=ps.RED, lw=0.9, ls=":", alpha=0.55, label="ideal")
        ax1.set_title(f"Polar centroids\nR_cv={radius_cv:.3f}  Δθ_std={spacing_std:.1f}°",
                      fontsize=9, pad=12)
        ax1.set_rticks([])
        ax1.legend(fontsize=7, loc="upper right")

        # Panel 3: Fourier fits for PC1–PC4
        k_fine = np.linspace(float(unique_labels.min()), float(unique_labels.max()), 120)
        for i in range(n_show):
            fit_fine = (pc_coefs[i][0] * np.cos(2*np.pi*k_fine/g)
                        + pc_coefs[i][1] * np.sin(2*np.pi*k_fine/g))
            ax2.plot(unique_labels, pc_means[i], markers[i],
                     color=line_colors[i], ms=6, zorder=4)
            ax2.plot(k_fine, fit_fine, color=line_colors[i], lw=1.6, ls="--",
                     label=f"PC{i+1}  R²={pc_r2[i]:.2f}")
        ax2.axhline(0, color=ps.GRAY, lw=0.7, ls="--", alpha=0.6)
        ax2.set_xlabel(label_name, fontsize=9)
        ax2.set_ylabel("mean PC score", fontsize=9)
        ax2.set_title(f"Fourier fits  PC1–PC{n_show}  (freq 1/g={g})", fontsize=9)
        ax2.set_xticks(unique_labels)
        ax2.legend(fontsize=7, ncol=2)

        # Panel 4: best quadrature pair scatter
        best_cents = np.column_stack([pc_means[bi], pc_means[bj]])
        for k, u in enumerate(unique_labels):
            mask = labels == u
            ax3.scatter(Z[mask, bi], Z[mask, bj], color=cmap(k), alpha=0.35, s=18, zorder=3)
        for k, (cx, cy) in enumerate(best_cents):
            ax3.scatter(cx, cy, color=cmap(k), s=110, marker="D", zorder=5,
                        edgecolors="white", linewidths=0.9)
            ax3.annotate(str(unique_labels[k]), (cx, cy),
                         textcoords="offset points", xytext=(4, 3),
                         fontsize=7, color=cmap(k), fontweight="bold")
        cx_b = np.append(best_cents[:, 0], best_cents[0, 0])
        cy_b = np.append(best_cents[:, 1], best_cents[0, 1])
        ax3.plot(cx_b, cy_b, color="#555555", lw=1.0, ls="--", alpha=0.6)
        ax3.set_xlabel(f"PC{bi+1} ({var_ratio[bi]*100:.1f}%)", fontsize=9)
        ax3.set_ylabel(f"PC{bj+1} ({var_ratio[bj]*100:.1f}%)", fontsize=9)
        ax3.set_title(
            f"Best quadrature pair: PC{bi+1}–PC{bj+1}\n"
            f"Δphase={best_phase_diff:.1f}°  "
            f"R²=({pc_r2[bi]:.2f}, {pc_r2[bj]:.2f})",
            fontsize=9,
        )

        fig.suptitle(
            f"Cluster {cluster_id}  ({n_feats} feats)  —  {anchor_name}",
            fontsize=10, y=1.01,
        )
        fig.tight_layout()
        out_path = out_dir / f"cluster_{cluster_id:02d}_pca.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    → {out_path}")

    return {
        "cluster_id": int(cluster_id),
        "n_features": n_feats,
        "sv_ratio_12": round(ratio_12, 4),
        "var_pc1": round(float(var_ratio[0]), 4),
        "var_pc2": round(float(var_ratio[1]), 4),
        "radius_cv": round(radius_cv, 4),
        "angle_spacing_std": round(spacing_std, 4),
        "ideal_spacing": round(ideal_spacing, 4),
        "angle_pearson_r": round(pearson_r, 4),
        "fourier_r2_pc1": round(r2_pc1, 4),
        "fourier_r2_pc2": round(r2_pc2, 4),
        "fourier_phase_diff": round(phase_diff, 2),
        "best_pair": f"PC{bi+1}-PC{bj+1}",
        "best_pair_phase_diff": round(best_phase_diff, 2),
        "best_pair_score": round(best_score, 4),
    }


def run_clustered_pca(
    D: np.ndarray,
    feat_labels: list[str],
    labels: np.ndarray,
    label_name: str,
    n_clusters: int,
    n_pcs: int,
    out_dir: Path,
    anchor_name: str,
    min_cluster_size: int = 3,
    no_plots: bool = False,
) -> None:
    """Cluster features by cosine similarity, run PCA per cluster, save diagnostics."""
    print(f"\n{'─'*60}")
    print(f"Clustering {D.shape[1]} features into {n_clusters} groups by cosine similarity")

    cos_sim, cluster_labels, groups = cluster_features_by_cosine(
        D, n_clusters=n_clusters, min_cluster_size=min_cluster_size
    )

    clust_dir = out_dir / f"clusters_k{n_clusters}"
    clust_dir.mkdir(parents=True, exist_ok=True)

    if not no_plots:
        plot_cosine_heatmap(cos_sim, cluster_labels, feat_labels,
                            clust_dir / "cosine_similarity.png", anchor_name)

    # Print cluster summary
    print(f"\nCluster sizes:")
    for c in range(n_clusters):
        idx = np.where(cluster_labels == c)[0]
        layers = [int(feat_labels[i].split("_")[0][1:]) for i in idx]
        print(f"  cluster {c}: {len(idx)} features  layers {sorted(set(layers))}")

    # PCA per cluster
    all_diags = []
    for group_idx, feat_idx in enumerate(groups):
        c_id = cluster_labels[feat_idx[0]]  # original cluster id
        D_cluster = D[:, feat_idx]
        diag = _pca_cluster(D_cluster, labels, label_name, n_pcs,
                             c_id, feat_idx, clust_dir, anchor_name, no_plots=no_plots)
        if diag:
            all_diags.append(diag)

    if not all_diags:
        print("No clusters large enough for PCA.")
        return

    # Summary table
    print(f"\n{'─'*75}")
    print(f"{'Cluster':>8}  {'N':>4}  {'SV2/SV1':>8}  {'R²_PC1':>7}  {'R²_PC2':>7}  "
          f"{'ΔPhase':>7}  {'R_cv':>6}  {'Pearson_r':>10}")
    for d in sorted(all_diags, key=lambda x: x["fourier_r2_pc1"] + x["fourier_r2_pc2"], reverse=True):
        print(f"  {d['cluster_id']:>6}  {d['n_features']:>4}  {d['sv_ratio_12']:>8.4f}  "
              f"  {d['fourier_r2_pc1']:>6.3f}  {d['fourier_r2_pc2']:>6.3f}  "
              f"  {d['fourier_phase_diff']:>6.1f}°  {d['radius_cv']:>5.3f}  "
              f"  {d['angle_pearson_r']:>9.4f}")
    print(f"{'─'*75}")
    print("Best clock basis: R²_PC1≈R²_PC2≈1  ΔPhase≈90°  SV2/SV1≈1")


# ── Main PCA plotting routine ─────────────────────────────────────────────────

def run_pca_analysis(
    sweep_dir: Path,
    results_json: Path | None,
    top_k: int,
    n_pcs: int,
    out_dir: Path,
    n_clusters: int = 0,
    no_plots: bool = False,
) -> None:
    # Load
    ranked = json.loads((sweep_dir / "sweep_ranked.json").read_text())
    npz    = np.load(sweep_dir / "sweep_activations.npz")
    with open(sweep_dir / "sweep_examples.pkl", "rb") as f:
        examples = pickle.load(f)

    # Peak layers
    peak_layers: set[int] | None = None
    if results_json is not None and results_json.exists():
        results = json.loads(results_json.read_text())
        peak_layers = _peak_layers_from_results(results)
        print(f"Using peak layers: {sorted(peak_layers)}")
    else:
        print("No results.json — using all layers")

    D, feat_labels = load_delta_matrix(sweep_dir, ranked, npz, top_k, peak_layers)
    print(f"Delta matrix shape: {D.shape}  (n_examples × n_features)")

    labels, label_name = extract_labels(examples)
    print(f"Label: {label_name}")
    print(f"  classes: {np.unique(labels)}  n_examples: {len(labels)}")

    # Standardise
    scaler = StandardScaler()
    D_std = scaler.fit_transform(D)

    # PCA
    n_pcs_actual = min(n_pcs, D_std.shape[1], D_std.shape[0])
    pca = PCA(n_components=n_pcs_actual, random_state=42)
    Z = pca.fit_transform(D_std)
    var_ratio = pca.explained_variance_ratio_
    sing_vals = np.sqrt(pca.explained_variance_)  # ∝ singular values

    print("\nExplained variance ratio per PC:")
    for i, (v, sv) in enumerate(zip(var_ratio, sing_vals)):
        print(f"  PC{i+1}: {v*100:.2f}%   singular value ∝ {sv:.4f}")
    print(f"  Total ({n_pcs_actual} PCs): {var_ratio.sum()*100:.2f}%")

    # ── Diagnostic: are singular values 1 and 2 equal? (rotational symmetry) ─
    ratio_12 = sing_vals[1] / sing_vals[0] if sing_vals[0] > 0 else 0.0
    print(f"\nSV2/SV1 = {ratio_12:.4f}  (→ 1.0 indicates rotational symmetry in PC1-PC2)")

    if not no_plots:
        out_dir.mkdir(parents=True, exist_ok=True)
        ps.apply()

        # ── Figure 1: PC pair scatter plots ──────────────────────────────────────
        pairs = [(0, 1), (1, 2), (2, 3)]
        pairs = [(a, b) for a, b in pairs if b < n_pcs_actual]
        n_pair = len(pairs)

        unique_labels = np.sort(np.unique(labels))
        n_cls = len(unique_labels)
        cmap = plt.get_cmap("tab10", n_cls)

        fig, axes = plt.subplots(1, n_pair, figsize=(5.5 * n_pair, 5.0), squeeze=False)
        for col, (i, j) in enumerate(pairs):
            ax = axes[0][col]
            for k, u in enumerate(unique_labels):
                mask = labels == u
                ax.scatter(Z[mask, i], Z[mask, j], color=cmap(k), alpha=0.45,
                           s=22, label=f"{label_name.split()[0]}={u}", zorder=3)
                cx, cy = Z[mask, i].mean(), Z[mask, j].mean()
                ax.scatter(cx, cy, color=cmap(k), s=90, marker="D", zorder=5,
                           edgecolors="white", linewidths=0.8)

            # Connect centroids in label order with dashed ring
            centroids = np.array([
                [Z[labels == u, i].mean(), Z[labels == u, j].mean()]
                for u in unique_labels
            ])
            ring_x = np.append(centroids[:, 0], centroids[0, 0])
            ring_y = np.append(centroids[:, 1], centroids[0, 1])
            ax.plot(ring_x, ring_y, color="#555555", lw=0.9, ls="--", alpha=0.6, zorder=4)

            vi = var_ratio[i] * 100
            vj = var_ratio[j] * 100
            ax.set_xlabel(f"PC{i+1} ({vi:.1f}%)", fontsize=10)
            ax.set_ylabel(f"PC{j+1} ({vj:.1f}%)", fontsize=10)
            ax.set_title(f"PC{i+1} vs PC{j+1}", fontsize=11)
            ax.legend(fontsize=7, ncol=2, loc="best")

        fig.suptitle(
            f"PCA of delta activations  —  {sweep_dir.parent.name}\n"
            f"top {top_k} features, standardised  |  label: {label_name}",
            fontsize=11, y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "pca_scatter_pairs.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {out_dir / 'pca_scatter_pairs.png'}")

        # ── Figure 2: circular diagnostics (PC1-PC2 scatter + centroid ring) ─────
        if n_pcs_actual >= 2:
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.5))
            diag = circular_diagnostics(
                Z[:, :2], labels, axes2[0], axes2[1],
                title=f"PC1 vs PC2  (top {top_k} feats, standardised)",
                label_name=label_name,
            )
            fig2.suptitle(
                f"Circular structure diagnostics  —  {sweep_dir.parent.name}\n"
                f"label: {label_name}",
                fontsize=11, y=1.02,
            )
            fig2.tight_layout()
            fig2.savefig(out_dir / "pca_circular_diagnostics.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"Saved: {out_dir / 'pca_circular_diagnostics.png'}")

            print("\nCircular diagnostics:")
            print(f"  Radius CV (want ≈0): {diag['radius_cv']:.4f}")
            print(f"  Angle spacing std (want ≈0, ideal={diag['ideal_spacing']:.1f}°): {diag['angle_spacing_std']:.2f}°")
            print(f"  Label–angle Pearson r (want ≈±1): {diag['angle_pearson_r']:.4f}")
            print(f"  Centroid radii: {[f'{r:.3f}' for r in diag['centroid_radii']]}")
            print(f"  Centroid angles (deg): {[f'{a:.1f}' for a in diag['centroid_angles_deg']]}")
            print(f"  Labels in angle order: {diag['sorted_labels']}")

        # ── Figure 3: scree plot + cumulative variance ────────────────────────────
        fig3, (ax_scree, ax_sv) = plt.subplots(1, 2, figsize=(10, 4))

        xs = np.arange(1, n_pcs_actual + 1)
        ax_scree.bar(xs, var_ratio * 100, color=ps.NAVY, alpha=0.75, width=0.7)
        ax_scree.plot(xs, np.cumsum(var_ratio) * 100, "o-", color=ps.RED,
                      lw=1.5, ms=5, label="cumulative")
        ax_scree.axhline(80, color=ps.GRAY, lw=0.8, ls="--", alpha=0.6)
        ax_scree.set_xlabel("PC", fontsize=10)
        ax_scree.set_ylabel("variance explained (%)", fontsize=10)
        ax_scree.set_title("Scree plot", fontsize=11)
        ax_scree.set_xticks(xs)
        ax_scree.legend(fontsize=8)

        ax_sv.bar(xs, sing_vals, color=ps.TEAL, alpha=0.75, width=0.7)
        ax_sv.set_xlabel("PC", fontsize=10)
        ax_sv.set_ylabel("singular value (√λ)", fontsize=10)
        ax_sv.set_title(r"Singular values — equal $\sigma_1 \approx \sigma_2$ → rotational symmetry", fontsize=10)
        ax_sv.set_xticks(xs)

        fig3.tight_layout()
        fig3.savefig(out_dir / "pca_scree.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved: {out_dir / 'pca_scree.png'}")

        # ── Figure 4: per-PC activation profile by label ─────────────────────────
        n_show = min(4, n_pcs_actual)
        fig4, axes4 = plt.subplots(1, n_show, figsize=(4.5 * n_show, 4.5), squeeze=False)
        for i in range(n_show):
            ax = axes4[0][i]
            means = [Z[labels == u, i].mean() for u in unique_labels]
            sems  = [Z[labels == u, i].std() / math.sqrt((labels == u).sum())
                     for u in unique_labels]
            ax.errorbar(unique_labels, means, yerr=sems,
                        fmt="o-", color=ps.NAVY, lw=1.4, ms=6, capsize=4)
            ax.axhline(0, color=ps.GRAY, lw=0.7, ls="--")
            ax.set_xlabel(label_name, fontsize=9)
            ax.set_ylabel(f"mean PC{i+1}", fontsize=9)
            ax.set_title(f"PC{i+1} by label class", fontsize=10)
            ax.set_xticks(unique_labels)
        fig4.suptitle(
            f"Mean PC score per label class — {sweep_dir.parent.name}", fontsize=11
        )
        fig4.tight_layout()
        fig4.savefig(out_dir / "pca_per_class.png", dpi=150, bbox_inches="tight")
        plt.close(fig4)
        print(f"Saved: {out_dir / 'pca_per_class.png'}")

        # ── UMAP (optional, skip if not installed) ────────────────────
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
            Z_umap = reducer.fit_transform(D_std)
            fig5, ax5 = plt.subplots(1, 1, figsize=(6, 5.5))
            for k, u in enumerate(unique_labels):
                mask = labels == u
                ax5.scatter(Z_umap[mask, 0], Z_umap[mask, 1], color=cmap(k),
                            alpha=0.5, s=22, label=f"{label_name.split()[0]}={u}", zorder=3)
            centroids_u = np.array([Z_umap[labels == u].mean(axis=0) for u in unique_labels])
            ring_x = np.append(centroids_u[:, 0], centroids_u[0, 0])
            ring_y = np.append(centroids_u[:, 1], centroids_u[0, 1])
            ax5.plot(ring_x, ring_y, color="#555555", lw=0.9, ls="--", alpha=0.6)
            ax5.legend(fontsize=7, ncol=2, loc="best")
            ax5.set_xlabel("UMAP 1", fontsize=10)
            ax5.set_ylabel("UMAP 2", fontsize=10)
            ax5.set_title(f"UMAP (n_neighbors=15)  —  {sweep_dir.parent.name}", fontsize=11)
            fig5.tight_layout()
            fig5.savefig(out_dir / "umap.png", dpi=150, bbox_inches="tight")
            plt.close(fig5)
            print(f"Saved: {out_dir / 'umap.png'}")
        except ImportError:
            print("umap-learn not installed — skipping UMAP (pip install umap-learn)")

        # ── Clustered PCA (optional) ─────────────────────────────────────────────
        if n_clusters > 1:
            run_clustered_pca(
                D, feat_labels, labels, label_name,
                n_clusters=n_clusters,
                n_pcs=n_pcs,
                out_dir=out_dir,
                anchor_name=sweep_dir.parent.name,
                no_plots=no_plots,
            )

        print(f"\nAll outputs saved to {out_dir}")


# ── Interpretation helper ─────────────────────────────────────────────────────

INTERPRETATION_GUIDE = """
Geometric interpretation guide
───────────────────────────────
Two dominant PCs with equal singular values (SV2/SV1 ≈ 1)
    Indicates rotational symmetry: the model uses a 2D Fourier basis (sin/cos pair)
    to represent the modular structure.  This is the signature of "clock" features
    as described by Nanda et al. (2023) for modular arithmetic.

Loop or ring in PC1-PC2
    Class centroids form an approximate regular polygon.  The ring closes (first
    and last labels connect) only if the concept is cyclic (Z/gZ with g classes).
    An arc that does not close indicates the labels are ordered but not periodic.

Low radius_cv (< 0.2)
    All class centroids are approximately equidistant from the centre of the ring,
    consistent with a genuine circular manifold.

Low angle_spacing_std (< ideal_spacing/3)
    Angular gaps between consecutive class centroids are roughly equal, consistent
    with a discrete Fourier representation of uniform spacing.

High |label-angle Pearson r| (> 0.9)
    Residue class rank correlates strongly with angular position, confirming that
    the model has embedded the cyclic group structure as a continuous ring rather
    than a set of arbitrary clusters.

Anti-correlated feature groups
    If some features have opposite sign in PC1, they may encode complementary
    Fourier components (sin vs cos of the same frequency).  Check the PCA loading
    matrix (pca.components_) for features with large |loading| of opposite sign.

UMAP vs PCA
    PCA finds the globally best linear subspace.  For modular arithmetic with a
    single dominant frequency, PCA is usually sufficient.  UMAP may reveal tighter
    clusters or multiple rings (harmonics) but can also introduce topological
    artefacts.  Trust PCA for linear structure; use UMAP to check for nonlinear
    separation that PCA misses.
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--sweep_dir",
        default="runs/concept_localization/gcd/anchor_rank1_pos6/sweep",
        help="Path to sweep directory (contains sweep_ranked.json etc.)",
    )
    parser.add_argument(
        "--results_json",
        default=None,
        help="Path to results.json for peak-layer selection (optional but recommended)",
    )
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of top-ranked features to include")
    parser.add_argument("--n_pcs", type=int, default=6,
                        help="Number of PCs to compute")
    parser.add_argument("--n_clusters", type=int, default=0,
                        help="If > 1, cluster features by cosine similarity and run PCA per cluster")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: <sweep_dir>/pca_analysis)",
    )
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating and saving any plot images")
    args = parser.parse_args()

    if args.guide:
        print(INTERPRETATION_GUIDE)
        return

    sweep_dir = Path(args.sweep_dir)
    results_json = (
        Path(args.results_json) if args.results_json
        else sweep_dir.parent / "results.json"
    )
    out_dir = Path(args.out_dir) if args.out_dir else sweep_dir / "pca_analysis"

    run_pca_analysis(
        sweep_dir=sweep_dir,
        results_json=results_json if results_json.exists() else None,
        top_k=args.top_k,
        n_pcs=args.n_pcs,
        out_dir=out_dir,
        n_clusters=args.n_clusters,
        no_plots=args.no_plots,
    )
    print(INTERPRETATION_GUIDE)


if __name__ == "__main__":
    main()
