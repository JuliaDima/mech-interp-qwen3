"""Sharpness analysis and transcoder feature projection for concept deltas.

Two analyses:
  1. Sharpness — norm trajectory, inter-layer cosine similarity, peak layer,
     sharpness index (fraction of total norm concentrated near the peak).
  2. Feature projection — project the aggregate delta onto each layer's
     transcoder decoder directions (E_dec = normalised W_dec rows) to identify
     which features write the concept direction into the residual stream.
  3. Template consistency — cross-template delta cosine similarity at each
     layer, validating that the carry direction is template-invariant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.concept_localization.extract_deltas import LayerDeltas
from experiments.concept_localization.extract_deltas_generic import (
    AnchorFactory,
    _resolve_anchor,
)
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

log = logging.getLogger(__name__)


@dataclass
class SharpnessResult:
    layers: list[int]
    norms: list[float]  # ||δ_l|| / E[||h_l||] when mean_act_norm available, else raw
    inter_layer_cos: list[float]  # cos_sim(δ_l, δ_{l+1})
    peak_layer: int
    sharpness_index: float  # fraction of normalised norm mass at peak ± 1 layers
    normalised: bool  # True when norms are activation-normalised


@dataclass
class FeatureMatch:
    feature_id: int
    projection: float  # ‖δ_l‖ · cos_sim(δ_l, E_dec_f)
    cos_sim: float  # cos_sim(δ_l, E_dec_f) — pure directional alignment in [-1, 1]
    layer: int


def compute_sharpness(ld: LayerDeltas) -> SharpnessResult:
    layers = sorted(ld.delta.keys())
    raw_norms = [ld.delta[l].norm().item() for l in layers]

    # Normalise by mean activation norm when available to remove residual-stream growth bias
    normalised = bool(ld.mean_act_norm)
    if normalised:
        norms = [r / ld.mean_act_norm.get(l, 1.0) for l, r in zip(layers, raw_norms, strict=False)]
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


def project_onto_E_dec_model(
    model,
    deltas: dict[int, torch.Tensor],
    top_k: int = 15,
) -> dict[int, list[FeatureMatch]]:
    """E_dec projection using the already-loaded model's transcoders.

    Mirrors project_onto_E_dec but reads W_dec from model.transcoders[layer], so
    callers that already hold a loaded model avoid re-reading transcoders from disk.
    """
    result: dict[int, list[FeatureMatch]] = {}
    for layer in sorted(deltas.keys()):
        try:
            tc = model.transcoders[layer]
        except (IndexError, KeyError):
            continue
        if not hasattr(tc, "W_dec"):
            continue
        W_dec = tc.W_dec.detach()  # (n_features, d_model)
        delta = deltas[layer].to(device=W_dec.device, dtype=W_dec.dtype)
        delta_norm = delta.norm()
        if delta_norm < 1e-8:
            continue
        dec_norms = W_dec.norm(dim=1).clamp(min=1e-8)
        cos_sims = (W_dec @ delta) / (dec_norms * delta_norm)
        projections = cos_sims * delta_norm
        k = min(top_k, cos_sims.numel())
        _, topk_ids = cos_sims.abs().topk(k)
        result[layer] = [
            FeatureMatch(
                feature_id=int(topk_ids[i].item()),
                projection=float(projections[topk_ids[i]].item()),
                cos_sim=float(cos_sims[topk_ids[i]].item()),
                layer=layer,
            )
            for i in range(k)
        ]
    return result


def project_onto_E_dec(
    deltas: dict[int, torch.Tensor],
    transcoder_cache: "Path",
    top_k: int = 15,
) -> dict[int, list[FeatureMatch]]:
    """For each layer, find top-k transcoder features whose decoder direction aligns with delta.

    Uses normalised decoder rows (E_dec = W_dec / ||W_dec||) — these capture which
    features *write* in the concept direction to the residual stream.
    Loads transcoders directly from disk; does not require a loaded model.
    """
    from pathlib import Path as _Path

    from mechinterp_qwen3.transcoder.single_layer_transcoder import load_relu_transcoder

    cache = _Path(transcoder_cache)
    result: dict[int, list[FeatureMatch]] = {}

    for layer in sorted(deltas.keys()):
        tc_path = cache / f"layer_{layer}.safetensors"
        if not tc_path.exists():
            continue
        tc = load_relu_transcoder(str(tc_path), layer=layer, lazy_encoder=True, lazy_decoder=False)
        W_dec = tc.W_dec.detach().float()   # (n_features, d_model)
        delta = deltas[layer].float()

        delta_norm = delta.norm()
        if delta_norm < 1e-8:
            continue

        dec_norms = W_dec.norm(dim=1).clamp(min=1e-8)
        cos_sims = (W_dec @ delta) / (dec_norms * delta_norm)   # [-1, 1]
        projections = cos_sims * delta_norm                      # ‖δ_l‖ · cos_sim

        k = min(top_k, cos_sims.numel())
        _, topk_ids = cos_sims.abs().topk(k)
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
            "Layer %2d  top E_dec feature: id=%d  cos_sim=%.3f",
            layer,
            result[layer][0].feature_id,
            result[layer][0].cos_sim,
        )

    return result


@torch.no_grad()
def sweep_concept_feature_activations(
    model,
    pairs,
    feature_ids: list[tuple[int, int]],
    anchor_mode: str = "delimiter",
    anchor_factory: AnchorFactory | None = None,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Collect transcoder feature activations at the anchor across all concept pairs.

    For each (layer, feat_id) in feature_ids, returns two arrays of length
    len(valid_pairs): activations on positive examples and on negative examples.

    Uses targeted residual-stream hooks and per-layer transcoder encoding only at
    the anchor position and only for the needed layers, rather than running the
    full get_activations (which computes all layers × all positions × full d_tc).
    """
    from collections import defaultdict

    from mechinterp_qwen3.transcoder.activation_functions import JumpReLU

    # Group feature indices by layer
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, feat_id in feature_ids:
        by_layer[layer].append(feat_id)

    # Pre-extract W_enc/b_enc rows for only the needed features per layer
    layer_info: dict[int, tuple] = {}
    for layer, feat_ids_l in by_layer.items():
        tc = model.transcoders[layer]
        idx = torch.tensor(feat_ids_l, dtype=torch.long)
        W_sub = tc.W_enc[idx].detach()  # (n_feats, d_model)
        b_sub = tc.b_enc[idx].detach()  # (n_feats,)
        act_fn = tc.activation_function
        is_jr = isinstance(act_fn, JumpReLU)
        if is_jr:
            thr = act_fn.threshold.detach()
            thr_sub = thr[idx] if thr.numel() > 1 else thr.expand(len(feat_ids_l))
        else:
            thr_sub = None
        layer_info[layer] = (feat_ids_l, W_sub, b_sub, is_jr, thr_sub)

    n = len(pairs)
    buf_pos = {fid: np.zeros(n, dtype=np.float32) for fid in feature_ids}
    buf_neg = {fid: np.zeros(n, dtype=np.float32) for fid in feature_ids}
    valid = np.zeros(n, dtype=bool)

    for i, pair in enumerate(tqdm(pairs, desc="Sweeping feature activations")):
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids
        if len(ids_pos) != len(ids_neg):
            continue

        anchor = _resolve_anchor(ids_pos, model.tokenizer, anchor_mode, anchor_factory, pair) + 1

        # Hook captures residual stream at anchor for needed layers only (no transcoder)
        resid_cache: dict[int, torch.Tensor] = {}
        hooks = [
            (
                f"blocks.{layer}.{model.feature_input_hook}",
                lambda acts, hook, _l=layer, _pos=anchor: (
                    resid_cache.update({_l: acts[0, _pos, :].detach().clone()})
                )
                or acts,
            )
            for layer in by_layer
        ]

        for ids, buf, label in [
            (ids_pos, buf_pos, "pos"),
            (ids_neg, buf_neg, "neg"),
        ]:
            resid_cache.clear()
            input_ids = tokenize_qwen_input(ids, model.tokenizer, model.cfg.device).unsqueeze(0)
            model.run_with_hooks(input_ids, fwd_hooks=hooks)

            for layer, (feat_ids_l, W_sub, b_sub, is_jr, thr_sub) in layer_info.items():
                if layer not in resid_cache:
                    continue
                h = resid_cache[layer]
                dev, dt = h.device, h.dtype
                pre = h @ W_sub.to(dev, dt).T + b_sub.to(dev, dt)  # (n_feats,)
                if is_jr:
                    out = pre * (pre > thr_sub.to(dev, dt))
                else:
                    out = torch.relu(pre)
                out_np = out.float().cpu().numpy()
                for j, fid in enumerate(feat_ids_l):
                    buf[(layer, fid)][i] = out_np[j]

        valid[i] = True

    return {fid: (buf_pos[fid][valid], buf_neg[fid][valid]) for fid in feature_ids}


@torch.no_grad()
def collect_layer_residuals(
    model,
    prompts_and_anchors: list[tuple[list[int], int]],
    target_layers: list[int],
) -> dict[int, np.ndarray]:
    """Capture residual-stream vectors at the anchor position for each prompt.

    prompts_and_anchors: list of (token_id_list, anchor_position) pairs.
    target_layers: which layers to hook (uses model.feature_input_hook).

    Returns dict layer → float32 array of shape (N, d_model),
    where N = len(prompts_and_anchors).  Missing entries (anchor out of range)
    are filled with zeros.

    This is the generic sweep primitive.  Concept-specific analysis (carry,
    gcd, ...) builds its prompts, resolves anchors, calls this function,
    and then applies its own scoring on the returned residuals.
    """
    N = len(prompts_and_anchors)
    H: dict[int, list[torch.Tensor]] = {l: [] for l in target_layers}

    for ids, anchor in tqdm(prompts_and_anchors, desc="Capturing residual stream"):
        # anchor was computed on raw ids; +1 for the sink token prepended by tokenize_qwen_input
        sink_anchor = anchor + 1
        resid_cache: dict[int, torch.Tensor] = {}
        hooks = [
            (
                f"blocks.{layer}.{model.feature_input_hook}",
                lambda acts, hook, _l=layer, _pos=sink_anchor: (
                    resid_cache.update({_l: acts[0, _pos, :].detach().clone()})
                    if _pos < acts.shape[1]
                    else None
                )
                or acts,
            )
            for layer in target_layers
        ]
        input_ids = tokenize_qwen_input(ids, model.tokenizer, model.cfg.device).unsqueeze(0)
        model.run_with_hooks(input_ids, fwd_hooks=hooks)

        for layer in target_layers:
            vec = resid_cache.get(layer)
            if vec is None:
                vec = torch.zeros(model.cfg.d_model, device=model.cfg.device)
            H[layer].append(vec)

    return {layer: torch.stack(vecs).float().cpu().numpy() for layer, vecs in H.items()}
