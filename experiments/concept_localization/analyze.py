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

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.concept_localization.extract_deltas_generic import (
    AnchorFactory,
    LayerDeltas,
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
    enc_cos_sim: float = 0.0  # cos_sim(δ_l, E_enc_f); 0.0 when score_mode="dec"


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
    score_mode: str = "enc+dec",
    direction: str = "pos",
) -> dict[int, list[FeatureMatch]]:
    """Project each layer's delta onto transcoder directions to find concept-aligned features.

    score_mode controls which weight matrix is used:
      "dec"     — score = cos(δ_l, W_dec[f]);            cos_sim = dec cosine
      "enc"     — score = cos(δ_l, W_enc[f]);            cos_sim = enc cosine
      "dec+enc" — score = dec_cos + enc_cos;             cos_sim = dec cosine, enc_cos_sim = enc cosine

    direction controls which end of the ranking to take:
      "pos" — highest-scoring features (most positive alignment with delta)
      "neg" — lowest-scoring features  (most negative alignment with delta)

    No absolute values are used; features are purely ranked by signed score.
    Only the weight matrices required by score_mode are loaded.
    """
    if score_mode not in ("dec", "enc", "dec+enc"):
        raise ValueError(f"score_mode must be 'dec', 'enc', or 'dec+enc', got {score_mode!r}")
    if direction not in ("pos", "neg"):
        raise ValueError(f"direction must be 'pos' or 'neg', got {direction!r}")

    need_dec = score_mode in ("dec", "dec+enc")
    need_enc = score_mode in ("enc", "dec+enc")
    largest = direction == "pos"

    result: dict[int, list[FeatureMatch]] = {}
    for layer in sorted(deltas.keys()):
        try:
            tc = model.transcoders[layer]
        except (IndexError, KeyError):
            continue

        if need_dec and not hasattr(tc, "W_dec"):
            continue
        if need_enc and not hasattr(tc, "W_enc"):
            continue

        # Use the first required weight to set device/dtype for delta
        ref_W = tc.W_dec.detach() if need_dec else tc.W_enc.detach()
        delta = deltas[layer].to(device=ref_W.device, dtype=ref_W.dtype)
        delta_norm = delta.norm()
        if delta_norm < 1e-8:
            continue

        if need_dec:
            W_dec = tc.W_dec.detach().to(device=ref_W.device, dtype=ref_W.dtype)
            dec_cos = (W_dec @ delta) / (W_dec.norm(dim=1).clamp(min=1e-8) * delta_norm)
        else:
            dec_cos = None

        if need_enc:
            W_enc = tc.W_enc.detach().to(device=ref_W.device, dtype=ref_W.dtype)
            enc_cos = (W_enc @ delta) / (W_enc.norm(dim=1).clamp(min=1e-8) * delta_norm)
        else:
            enc_cos = None

        if score_mode == "dec":
            rank_scores = dec_cos
            primary_cos, secondary_cos = dec_cos, torch.zeros_like(dec_cos)
        elif score_mode == "enc":
            rank_scores = enc_cos
            primary_cos, secondary_cos = enc_cos, torch.zeros_like(enc_cos)
        else:  # dec+enc
            rank_scores = dec_cos + enc_cos
            primary_cos, secondary_cos = dec_cos, enc_cos

        k = min(top_k, rank_scores.numel())
        _, topk_ids = rank_scores.topk(k, largest=largest)
        result[layer] = [
            FeatureMatch(
                feature_id=int(topk_ids[i].item()),
                projection=float((primary_cos[topk_ids[i]] * delta_norm).item()),
                cos_sim=float(primary_cos[topk_ids[i]].item()),
                enc_cos_sim=float(secondary_cos[topk_ids[i]].item()),
                layer=layer,
            )
            for i in range(k)
        ]
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
def sweep_multi_anchor_activations(
    model,
    pairs,
    anchor_features: dict[str, list[tuple[int, int]]],
) -> dict[str, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]:
    """One-pass activation sweep for multiple anchor positions simultaneously.

    anchor_features: {anchor_mode: [(layer, feat_id), ...]}

    For each pair we run ONE forward pass (pos) and ONE (neg), hooking every needed
    layer and capturing residuals at every needed anchor position in that single pass.
    This is O(n_pairs × 2) forward passes regardless of how many anchor modes there are,
    versus O(n_anchors × n_pairs × 2) if sweep_concept_feature_activations were called
    once per anchor.

    Returns {anchor_mode: {(layer, feat_id): (pos_acts, neg_acts)}}.
    """
    from collections import defaultdict

    from mechinterp_qwen3.transcoder.activation_functions import JumpReLU

    # Pre-build encoder weight slices per anchor per layer
    anchor_layer_info: dict[str, dict[int, tuple]] = {}
    all_layers: set[int] = set()
    for anchor_mode, feature_ids in anchor_features.items():
        by_layer: dict[int, list[int]] = defaultdict(list)
        for layer, feat_id in feature_ids:
            by_layer[layer].append(feat_id)
        all_layers.update(by_layer.keys())
        layer_info: dict[int, tuple] = {}
        for layer, feat_ids_l in by_layer.items():
            tc = model.transcoders[layer]
            idx = torch.tensor(feat_ids_l, dtype=torch.long)
            W_sub = tc.W_enc[idx].detach()
            b_sub = tc.b_enc[idx].detach()
            act_fn = tc.activation_function
            is_jr = isinstance(act_fn, JumpReLU)
            thr_sub = None
            if is_jr:
                thr = act_fn.threshold.detach()
                thr_sub = thr[idx] if thr.numel() > 1 else thr.expand(len(feat_ids_l))
            layer_info[layer] = (feat_ids_l, W_sub, b_sub, is_jr, thr_sub)
        anchor_layer_info[anchor_mode] = layer_info

    n = len(pairs)
    anchor_modes = list(anchor_features.keys())
    feature_ids_per_anchor = {am: anchor_features[am] for am in anchor_modes}

    buf_pos = {am: {fid: np.zeros(n, dtype=np.float32) for fid in fids}
               for am, fids in feature_ids_per_anchor.items()}
    buf_neg = {am: {fid: np.zeros(n, dtype=np.float32) for fid in fids}
               for am, fids in feature_ids_per_anchor.items()}
    valid = np.zeros(n, dtype=bool)

    for i, pair in enumerate(tqdm(pairs, desc="Multi-anchor activation sweep")):
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids
        if len(ids_pos) != len(ids_neg):
            continue

        # Resolve all anchor positions for this pair (sink token offset already included)
        anchor_pos_map: dict[str, int] = {
            am: _resolve_anchor(ids_pos, model.tokenizer, am, None, pair) + 1
            for am in anchor_modes
        }

        # For each layer, collect the set of positions we need to capture
        layer_positions: dict[int, set[int]] = defaultdict(set)
        for am, layer_info in anchor_layer_info.items():
            pos = anchor_pos_map[am]
            for layer in layer_info:
                layer_positions[layer].add(pos)

        # Single hook per layer captures residuals at all needed positions
        resid_cache: dict[tuple[int, int], torch.Tensor] = {}

        def _make_hook(layer: int, positions: frozenset) -> object:
            def _hook(acts, hook):
                for pos in positions:
                    if pos < acts.shape[1]:
                        resid_cache[(layer, pos)] = acts[0, pos, :].detach().clone()
                return acts
            return _hook

        hooks = [
            (f"blocks.{layer}.{model.feature_input_hook}",
             _make_hook(layer, frozenset(positions)))
            for layer, positions in layer_positions.items()
        ]

        for ids, bufs in [(ids_pos, buf_pos), (ids_neg, buf_neg)]:
            resid_cache.clear()
            input_ids = tokenize_qwen_input(ids, model.tokenizer, model.cfg.device).unsqueeze(0)
            model.run_with_hooks(input_ids, fwd_hooks=hooks)

            for am, layer_info in anchor_layer_info.items():
                pos = anchor_pos_map[am]
                for layer, (feat_ids_l, W_sub, b_sub, is_jr, thr_sub) in layer_info.items():
                    h = resid_cache.get((layer, pos))
                    if h is None:
                        continue
                    dev, dt = h.device, h.dtype
                    pre = h @ W_sub.to(dev, dt).T + b_sub.to(dev, dt)
                    out = pre * (pre > thr_sub.to(dev, dt)) if is_jr else torch.relu(pre)
                    out_np = out.float().cpu().numpy()
                    for j, fid in enumerate(feat_ids_l):
                        bufs[am][(layer, fid)][i] = out_np[j]

        valid[i] = True

    return {
        am: {fid: (buf_pos[am][fid][valid], buf_neg[am][fid][valid])
             for fid in feature_ids_per_anchor[am]}
        for am in anchor_modes
    }


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