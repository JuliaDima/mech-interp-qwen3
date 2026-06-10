"""Select peak layers for a concept-localisation anchor.

Uses a combined score from four signals:
  - double-normalised delta trajectory  (concept magnitude)
  - null-excess                         (significance above permutation null)
  - activation patching                 (causal relevance)
  - gradient · delta (down-weighted)    (gradient-based relevance)

Anchors whose real signal never exceeds the null+1SD are flagged as invalid.
Stable-direction anchors (mean pairwise cosine > STABLE_COS_THRESH) are treated
differently: traj is dropped from the combined score because it is uniformly
high across all layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks


# ── tuneable defaults ────────────────────────────────────────────────────────
STABLE_COS_THRESH   = 0.50   # mean pairwise delta cosine above which anchor is "stable"
PATCH_WIN_THRESH    = 0.10   # patch_n threshold for causal window
PEAK_DISTANCE_FRAC  = 1 / 8  # distance = max(3, n_layers * frac)
PEAK_PROMINENCE     = 0.08
GRAD_WEIGHT         = 0.25   # relative weight of grad·delta vs excess/patch (each =1)
TOP_K_PEAKS         = 2


@dataclass
class PeakLayerResult:
    anchor_name: str
    valid: bool                          # False → real signal never above null+1SD
    stable: bool                         # True  → direction barely rotates across layers
    mean_cos: float
    traj_l0: float
    causal_window: list[int]             # layers inside the causal window
    peak_layers: list[int]               # top-k peak layers, sorted by score desc
    peak_scores: list[float]
    peak_prominences: list[float]
    layers: list[int]
    combined: np.ndarray = field(repr=False)   # full combined score (all layers)


def select_peak_layers(
    anchor_dir: Path | str,
    template: str = "T0",
    top_k: int = TOP_K_PEAKS,
    stable_cos_thresh: float = STABLE_COS_THRESH,
    patch_win_thresh: float = PATCH_WIN_THRESH,
    peak_distance_frac: float = PEAK_DISTANCE_FRAC,
    peak_prominence: float = PEAK_PROMINENCE,
    grad_weight: float = GRAD_WEIGHT,
) -> PeakLayerResult:
    anchor_dir = Path(anchor_dir)

    results = json.load(open(anchor_dir / "results.json"))
    raw_d   = torch.load(anchor_dir / "deltas.pt", map_location="cpu", weights_only=False)
    deltas  = raw_d.get(template) or raw_d.get("all") or {}
    null    = json.load(open(anchor_dir / "null" / "null_permutation.json"))
    causal  = results.get("causal", {})
    causal  = causal.get(template) or causal.get("all") or {}

    layers = sorted(int(l) for l in deltas.keys())
    L = len(layers)

    def pnorm(a: np.ndarray) -> np.ndarray:
        p = a.max()
        return a / p if p > 1e-12 else a

    # ── trajectory ───────────────────────────────────────────────────────────
    raw      = np.array([float(deltas[l].norm()) for l in layers])
    mean_act = {int(k): float(v) for k, v in results.get("mean_act_norm", {}).items()}
    act_norm = np.array([raw[i] / mean_act.get(l, 1.0) for i, l in enumerate(layers)])
    traj     = pnorm(act_norm)

    # ── null excess ──────────────────────────────────────────────────────────
    nlayers  = [int(x) for x in null["layers"]]
    real_n   = np.array(null["real_norms"])
    nulls    = np.array(null["null_norms"])
    null_m, null_s = nulls.mean(0), nulls.std(0)
    excess_raw = np.array([
        real_n[nlayers.index(l)] - (null_m + null_s)[nlayers.index(l)]
        if l in nlayers else -1.0
        for l in layers
    ])
    valid = bool(excess_raw.max() > 0)
    excess_n = pnorm(np.maximum(0.0, excess_raw))

    # ── patching & grad ──────────────────────────────────────────────────────
    patch_r = np.array([float(causal.get("patching_mean", {}).get(str(l), 0.0)) for l in layers])
    grad_r  = np.array([float(causal.get("grad_dot_delta_mean", {}).get(str(l), 0.0)) for l in layers])
    patch_n = pnorm(np.abs(patch_r))
    grad_n  = pnorm(np.abs(grad_r))

    # ── anchor type ──────────────────────────────────────────────────────────
    vecs = [deltas[l].float() for l in layers]
    cos_vals = [
        F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
        for i in range(L) for j in range(i + 1, L)
    ]
    mean_cos = float(np.mean(cos_vals)) if cos_vals else 0.0
    traj_l0  = float(traj[0])
    stable   = mean_cos > stable_cos_thresh

    # ── causal window ────────────────────────────────────────────────────────
    causal_mask = patch_n >= patch_win_thresh
    signal_mask = excess_raw > 0
    valid_mask  = causal_mask & signal_mask
    causal_window = [layers[i] for i in range(L) if valid_mask[i]]

    # ── combined score ───────────────────────────────────────────────────────
    norm_sum = 1.0 + 1.0 + grad_weight
    if stable:
        combined = (1.0 * excess_n + 1.0 * patch_n + grad_weight * grad_n) / norm_sum
    else:
        combined = traj * (1.0 * excess_n + 1.0 * patch_n + grad_weight * grad_n) / norm_sum

    combined_w = np.where(valid_mask, combined, 0.0)

    # ── peak detection ───────────────────────────────────────────────────────
    dist = max(3, int(L * peak_distance_frac))
    if valid and causal_window:
        peaks, props = find_peaks(combined_w, distance=dist, prominence=peak_prominence)
        ranked = sorted(
            zip(peaks, props["prominences"]),
            key=lambda x: -combined_w[x[0]],
        )[:top_k]
        peak_layers     = [layers[p] for p, _ in ranked]
        peak_scores     = [float(combined_w[p]) for p, _ in ranked]
        peak_prominences = [float(prom) for _, prom in ranked]
    else:
        peak_layers, peak_scores, peak_prominences = [], [], []

    return PeakLayerResult(
        anchor_name=anchor_dir.name,
        valid=valid,
        stable=stable,
        mean_cos=mean_cos,
        traj_l0=traj_l0,
        causal_window=causal_window,
        peak_layers=peak_layers,
        peak_scores=peak_scores,
        peak_prominences=peak_prominences,
        layers=layers,
        combined=combined_w,
    )
