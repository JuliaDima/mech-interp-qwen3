"""Sharpness analysis and transcoder feature projection for concept deltas.

Two analyses:
  1. Sharpness — norm trajectory, inter-layer cosine similarity, peak layer,
     sharpness index (fraction of total norm concentrated near the peak).
  2. Feature projection — project the aggregate delta onto each layer's
     transcoder encoder directions (W_enc rows) to identify which features
     constitute the concept direction.
  3. Template consistency — cross-template delta cosine similarity at each
     layer, validating that the carry direction is template-invariant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.concept_localization.extract_deltas import LayerDeltas

log = logging.getLogger(__name__)


@dataclass
class SharpnessResult:
    layers: list[int]
    norms: list[float]          # ||δ_l|| / E[||h_l||] when mean_act_norm available, else raw
    inter_layer_cos: list[float]  # cos_sim(δ_l, δ_{l+1})
    peak_layer: int
    sharpness_index: float      # fraction of normalised norm mass at peak ± 1 layers
    normalised: bool            # True when norms are activation-normalised


@dataclass
class FeatureMatch:
    feature_id: int
    projection: float   # ‖δ_l‖ · cos_sim(δ_l, W_enc_f)
    cos_sim: float      # cos_sim(δ_l, W_enc_f) — pure directional alignment in [-1, 1]
    layer: int


def compute_sharpness(ld: LayerDeltas) -> SharpnessResult:
    layers = sorted(ld.delta.keys())
    raw_norms = [ld.delta[l].norm().item() for l in layers]

    # Normalise by mean activation norm when available to remove residual-stream growth bias
    normalised = bool(ld.mean_act_norm)
    if normalised:
        norms = [r / ld.mean_act_norm.get(l, 1.0) for l, r in zip(layers, raw_norms)]
    else:
        norms = raw_norms

    inter_cos: list[float] = []
    for i in range(len(layers) - 1):
        a = ld.delta[layers[i]]
        b = ld.delta[layers[i + 1]]
        inter_cos.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    peak_idx = int(torch.tensor(norms).argmax().item())
    peak_layer = layers[peak_idx]

    window = slice(max(0, peak_idx - 1), min(len(norms), peak_idx + 2))
    sharpness_index = sum(norms[window]) / (sum(norms) + 1e-8)

    return SharpnessResult(
        layers=layers,
        norms=norms,
        inter_layer_cos=inter_cos,
        peak_layer=peak_layer,
        sharpness_index=sharpness_index,
        normalised=normalised,
    )


def compute_template_consistency(
    results: dict[str, LayerDeltas],
) -> dict[int, dict[str, float]]:
    """Return pairwise cosine similarity between per-template deltas at each layer.

    Returns {layer: {"T0_vs_T1": float, "T0_vs_T2": float, ...}}.
    """
    template_keys = [k for k in results if k != "all"]
    if len(template_keys) < 2:
        return {}

    all_layers = sorted(results["all"].delta.keys())
    consistency: dict[int, dict[str, float]] = {}

    for layer in all_layers:
        row: dict[str, float] = {}
        for i, k1 in enumerate(template_keys):
            for k2 in template_keys[i + 1 :]:
                if layer not in results[k1].delta or layer not in results[k2].delta:
                    continue
                a = results[k1].delta[layer]
                b = results[k2].delta[layer]
                cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                row[f"{k1}_vs_{k2}"] = round(cos, 4)
        consistency[layer] = row

    return consistency


def project_onto_features(
    model,
    ld: LayerDeltas,
    top_k: int = 15,
) -> dict[int, list[FeatureMatch]]:
    """For each layer, find top-k transcoder features most aligned with the delta.

    Uses the encoder input directions (W_enc rows) since those capture which
    features "respond to" the concept direction in the residual stream.
    """
    result: dict[int, list[FeatureMatch]] = {}

    for layer in sorted(ld.delta.keys()):
        try:
            transcoder = model.transcoders[layer]
        except (IndexError, KeyError):
            continue
        if not hasattr(transcoder, "W_enc"):
            continue

        # Load W_enc once (may trigger lazy load from disk)
        W_enc = transcoder.W_enc.detach()  # (n_features, d_model)
        delta = ld.delta[layer].to(device=W_enc.device, dtype=W_enc.dtype)

        delta_norm = delta.norm()
        if delta_norm < 1e-8:
            continue

        enc_norms = W_enc.norm(dim=1).clamp(min=1e-8)  # (n_features,)
        projections = (W_enc @ delta) / enc_norms       # ‖δ_l‖ · cos_sim
        cos_sims = projections / delta_norm             # pure cos_sim in [-1, 1]

        k = min(top_k, projections.numel())
        topk_vals, topk_ids = projections.abs().topk(k)
        result[layer] = [
            FeatureMatch(
                feature_id=int(topk_ids[i].item()),
                projection=float(projections[topk_ids[i]].item()),
                cos_sim=float(cos_sims[topk_ids[i]].item()),
                layer=layer,
            )
            for i in range(k)
        ]

        log.info(
            "Layer %2d  top feature: id=%d  projection=%.3f",
            layer,
            result[layer][0].feature_id,
            result[layer][0].projection,
        )

    return result


