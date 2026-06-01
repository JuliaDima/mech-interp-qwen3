"""Shared utilities for concept-specific transcoder feature sweeps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mechinterp_qwen3.transcoder.activation_functions import JumpReLU


def apply_transcoder_all(model, layer: int, H_l: np.ndarray) -> np.ndarray:
    """Apply JumpReLU transcoder encoding for all features at a layer.

    H_l: (N, d_model) float32.  Returns (N, d_tc) float32.
    """
    tc = model.transcoders[layer]
    H_t = torch.from_numpy(H_l).to(device=model.cfg.device, dtype=torch.float32)
    W_enc = tc.W_enc.detach().to(H_t.device, H_t.dtype)
    b_enc = tc.b_enc.detach().to(H_t.device, H_t.dtype)
    pre = H_t @ W_enc.T + b_enc
    del W_enc, b_enc
    act_fn = tc.activation_function
    if isinstance(act_fn, JumpReLU):
        thr = act_fn.threshold.detach().to(H_t.device, H_t.dtype)
        acts = pre * (pre > thr)
    else:
        acts = torch.relu(pre)
    del pre
    return acts.float().cpu().numpy()


def score_and_rank(
    acts: np.ndarray,
    pos_mask: np.ndarray,
    top_k: int = 50,
) -> list[tuple[int, float, float]]:
    """Score all features by discrimination and Jaccard, return top_k ranked.

    acts:     (N, d_tc) float32
    pos_mask: (N,) bool — True for positive concept examples
    Returns list of (feat_id, score, jaccard) sorted by jaccard × |score| descending.
    score > 0 means feature fires more for positive examples.
    """
    neg_mask = ~pos_mask
    scores = acts[pos_mask].mean(axis=0) - acts[neg_mask].mean(axis=0)
    active = acts > 0
    cm = pos_mask[:, None]
    ncm = neg_mask[:, None]
    inter_c = (active & cm).sum(axis=0).astype(np.float32)
    union_c = (active | cm).sum(axis=0).astype(np.float32)
    jac_c = np.where(union_c > 0, inter_c / union_c, 0.0)
    inter_nc = (active & ncm).sum(axis=0).astype(np.float32)
    union_nc = (active | ncm).sum(axis=0).astype(np.float32)
    jac_nc = np.where(union_nc > 0, inter_nc / union_nc, 0.0)
    jaccard = np.where(scores >= 0, jac_c, jac_nc)
    combined = jaccard * np.abs(scores)
    top_idx = np.argsort(combined)[::-1][:top_k]
    return [(int(f), float(scores[f]), float(jaccard[f])) for f in top_idx]


def cluster_top_features(
    acts: np.ndarray,
    pos_mask: np.ndarray,
    top_frac: float = 0.15,
    n_clusters: int = 10,
) -> list[tuple[int, float, int]]:
    """Select top features by |score|, cluster by normalised activation pattern.

    No mask or shape assumption — clustering discovers what patterns exist.
    Takes the top `top_frac` fraction of features by mean-difference amplitude,
    normalises each feature's activation vector to unit norm (so clustering is
    about shape, not magnitude), then runs k-means.

    Returns list of (feat_id, score, cluster_id) for all selected features,
    sorted by cluster_id then |score| descending so features from the same
    cluster are adjacent.
    """
    from sklearn.cluster import KMeans

    scores = acts[pos_mask].mean(axis=0) - acts[~pos_mask].mean(axis=0)
    n_top = max(n_clusters, int(acts.shape[1] * top_frac))
    top_idx = np.argsort(np.abs(scores))[::-1][:n_top]

    top_acts = acts[:, top_idx].T  # (n_top, N)
    norms = np.linalg.norm(top_acts, axis=1, keepdims=True).clip(min=1e-8)
    top_norm = top_acts / norms

    n_c = min(n_clusters, len(top_idx))
    km = KMeans(n_clusters=n_c, random_state=42, n_init=10)
    labels = km.fit_predict(top_norm)

    result = [
        (int(top_idx[i]), float(scores[top_idx[i]]), int(labels[i])) for i in range(len(top_idx))
    ]
    result.sort(key=lambda x: (x[2], -abs(x[1])))
    return result
