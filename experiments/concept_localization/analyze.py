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


def save_edec_features(
    model,
    deltas: dict[int, torch.Tensor],
    out_path,
    top_k: int = 15,
    score_mode: str = "dec+enc",
    pairs=None,
    anchor_mode: str | None = None,
    anchor_factory: "AnchorFactory | None" = None,
) -> dict:
    """Project delta onto transcoder directions; save top-k positive and top-k negative features.

    Calls project_onto_E_dec_model twice — direction='pos' (highest score) and 'neg'
    (lowest score) — so the caller gets both ends without abs-value mixing.

    If pairs and anchor_mode are provided, also sweeps the model to compute mean/std
    activations on pos and neg examples for each feature, embedding them in the JSON so
    downstream plot scripts need no model.

    Saved JSON structure:
      {
        "config": {"score_mode": ..., "top_k": ...},
        "pos":  [{"feature": "LX_FY", "layer": X, "feature_id": Y,
                  "dec_cos": ..., "enc_cos": ..., "score": ...,
                  "mean_pos": ..., "mean_neg": ..., "std_pos": ..., "std_neg": ...}, ...],
        "neg": [...]
      }

    Features in "pos" have the highest dec_cos+enc_cos (most positive);
    "neg" have the lowest (most negative). mean_*/std_* are only present when
    pairs is provided.
    """
    top_matches  = project_onto_E_dec_model(model, deltas, top_k=top_k,
                                             score_mode=score_mode, direction="pos")
    last_matches = project_onto_E_dec_model(model, deltas, top_k=top_k,
                                             score_mode=score_mode, direction="neg")

    def _to_list(matches: dict) -> list[dict]:
        rows = []
        for ms in matches.values():
            for m in ms:
                dec_cos = float(m.cos_sim)
                enc_cos = float(m.enc_cos_sim)
                rows.append({
                    "feature":    f"L{m.layer}_F{m.feature_id}",
                    "layer":      m.layer,
                    "feature_id": m.feature_id,
                    "dec_cos":    round(dec_cos, 5),
                    "enc_cos":    round(enc_cos, 5),
                    "score":      round(dec_cos + enc_cos, 5),
                })
        return sorted(rows, key=lambda r: -r["score"])

    top_rows  = _to_list(top_matches)
    last_rows = _to_list(last_matches)

    # Optionally sweep activations and embed mean/std into each row
    if pairs is not None and anchor_mode is not None:
        unique_ids = list({(r["layer"], r["feature_id"]) for r in top_rows + last_rows})
        log.info("Sweeping activations for %d features …", len(unique_ids))
        act_map = sweep_concept_feature_activations(
            model, pairs, unique_ids, anchor_mode=anchor_mode,
            anchor_factory=anchor_factory,
        )
        act_stats: dict[str, dict] = {}
        for (layer, fid), (pos_arr, neg_arr) in act_map.items():
            act_stats[f"L{layer}_F{fid}"] = {
                "mean_pos": round(float(pos_arr.mean()), 6) if len(pos_arr) else 0.0,
                "mean_neg": round(float(neg_arr.mean()), 6) if len(neg_arr) else 0.0,
                "std_pos":  round(float(pos_arr.std()),  6) if len(pos_arr) else 0.0,
                "std_neg":  round(float(neg_arr.std()),  6) if len(neg_arr) else 0.0,
            }
        for row in top_rows + last_rows:
            row.update(act_stats.get(row["feature"], {}))

    data = {
        "config": {"score_mode": score_mode, "top_k": top_k},
        "pos": top_rows,
        "neg": last_rows,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    log.info("Saved edec features → %s  (pos=%d, neg=%d)",
             out_path, len(data["pos"]), len(data["neg"]))
    return data


# WILL DELETE THIS
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Compute and save edec feature projections for one or more anchor run dirs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_dirs", nargs="+", required=True,
        help="One or more anchor run directories containing deltas.pt",
    )
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--score_mode", default="dec+enc", choices=["dec", "enc", "dec+enc"])
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--transcoder_set", default="mwhanna/qwen3-4b-transcoders")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--concept", default=None,
                        help="Concept name — if provided, sweeps activations and embeds mean/std in JSON")
    parser.add_argument("--n", type=int, default=200, help="Pairs to sweep (only used with --concept)")
    parser.add_argument("--seed", type=int, default=42)
    cli_args = parser.parse_args()

    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    sys.path.insert(0, str(_REPO_ROOT))

    from mechinterp_qwen3.attribution_model import AttributionModel
    from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
    from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

    device = get_default_device()
    dtype = parse_dtype(cli_args.dtype)
    log.info("Loading model %s …", cli_args.model)
    tc_set, _ = load_transcoder_from_hub(
        cli_args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=False
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        cli_args.model, tc_set, dtype=dtype, device=device
    )
    model.eval()

    pairs = None
    if cli_args.concept:
        from experiments.concept_localization.run_concept import _load_concept
        pairs = _load_concept(cli_args.concept, cli_args.n, cli_args.seed)

    for run_dir_str in cli_args.run_dirs:
        run_dir = Path(run_dir_str)
        deltas_path = run_dir / "deltas.pt"
        if not deltas_path.exists():
            log.warning("deltas.pt not found in %s — skipping", run_dir)
            continue
        raw = torch.load(str(deltas_path), map_location=device)
        deltas = raw["all"]  # dict[int, Tensor]

        anchor_mode = None
        results_path = run_dir / "results.json"
        if results_path.exists() and pairs is not None:
            cfg = json.loads(results_path.read_text()).get("config", {})
            anchor_mode = str(cfg.get("anchor_mode", cfg.get("anchor_pos", "delimiter")))

        out_path = run_dir / "edec_features.json"
        data = save_edec_features(model, deltas, out_path,
                                  top_k=cli_args.top_k, score_mode=cli_args.score_mode,
                                  pairs=pairs, anchor_mode=anchor_mode)
        print(f"\n=== {run_dir.name} ===")
        print(f"  score_mode={cli_args.score_mode}  top_k={cli_args.top_k}")
        print("  POS (most positive):")
        for r in data["pos"][:5]:
            acts = f"  mean_pos={r['mean_pos']:+.4f} mean_neg={r['mean_neg']:+.4f}" if "mean_pos" in r else ""
            print(f"    {r['feature']:>14}  dec={r['dec_cos']:+.4f}  enc={r['enc_cos']:+.4f}  score={r['score']:+.4f}{acts}")
        print("  NEG (most negative):")
        for r in data["neg"][:5]:
            acts = f"  mean_pos={r['mean_pos']:+.4f} mean_neg={r['mean_neg']:+.4f}" if "mean_pos" in r else ""
            print(f"    {r['feature']:>14}  dec={r['dec_cos']:+.4f}  enc={r['enc_cos']:+.4f}  score={r['score']:+.4f}{acts}")
