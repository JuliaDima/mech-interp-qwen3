"""Run a carry feature sweep for multiple anchors in one prompt pass.

This is the multi-anchor variant of run_concept_sweep.py.  It captures residuals
at several raw token positions during the same forward pass for each prompt, then
applies each layer transcoder per anchor and saves the usual all_feature_grids
files under:

    <out_dir>/anchor_rank{rank}_pos{pos}/all_feature_grids/layer_XX_all_feature_grids.npz

The saved files are compatible with find_closest_all_feature_grid.py, which can
scan this root recursively and report both anchor and feature.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEPS_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _SWEEPS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sweep_utils import apply_transcoder_all

from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_concept_sweep_multi_anchor")

_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"


def _load_concept(name: str, n: int, seed: int):
    mod = importlib.import_module(f"data.concept_datasets.{name}_dataset")
    for fn in [
        f"generate_{name}_pairs",
        f"generate_{name.split('_')[-1]}_pairs",
        "generate_decimal_pairs",
        "generate_large_prime_pairs",
        "generate_wave_pairs",
    ]:
        if hasattr(mod, fn):
            return getattr(mod, fn)(n, seed=seed)
    for attr_name in dir(mod):
        if attr_name.startswith("generate_") and attr_name.endswith("_pairs"):
            return getattr(mod, attr_name)(n, seed=seed)
    raise ValueError(f"Cannot find a generate function in {name}_dataset")


def _parse_anchor_specs(specs: str) -> list[tuple[str, int]]:
    anchors: list[tuple[str, int]] = []
    for chunk in specs.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            label, pos_s = chunk.split(":", 1)
            pos = int(pos_s)
        else:
            pos = int(chunk)
            label = f"pos{pos}"
        anchors.append((label.strip(), pos))
    if not anchors:
        raise ValueError("No anchors specified")
    return anchors


def _safe_anchor_label(label: str, pos: int) -> str:
    label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
    return label if f"pos{pos}" in label else f"{label}_pos{pos}"


def _acts_to_grid(acts: np.ndarray, pairs) -> np.ndarray:
    sums = np.zeros((10, 10), dtype=np.float64)
    counts = np.zeros((10, 10), dtype=np.int64)
    for pair_i, pair in enumerate(pairs):
        for is_pos, act_idx in ((True, 2 * pair_i), (False, 2 * pair_i + 1)):
            if act_idx >= len(acts):
                continue
            a = pair.meta["a_pos"] if is_pos else pair.meta["a_neg"]
            b = pair.meta["b_pos"] if is_pos else pair.meta["b_neg"]
            sums[a % 10, b % 10] += acts[act_idx]
            counts[a % 10, b % 10] += 1
    grid = np.full((10, 10), np.nan, dtype=np.float32)
    mask = counts > 0
    grid[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    return grid


def _score_features(acts_np: np.ndarray, pos_mask: np.ndarray):
    neg_mask = ~pos_mask
    scores = acts_np[pos_mask].mean(axis=0) - acts_np[neg_mask].mean(axis=0)
    active = acts_np > 0
    eligible = np.where(active.any(axis=0))[0]

    cm = pos_mask[:, None]
    ncm = neg_mask[:, None]
    inter_c = (active & cm).sum(axis=0).astype(np.float32)
    union_c = (active | cm).sum(axis=0).astype(np.float32)
    jac_c = np.divide(inter_c, union_c, out=np.zeros_like(inter_c), where=union_c > 0)
    inter_nc = (active & ncm).sum(axis=0).astype(np.float32)
    union_nc = (active | ncm).sum(axis=0).astype(np.float32)
    jac_nc = np.divide(inter_nc, union_nc, out=np.zeros_like(inter_nc), where=union_nc > 0)
    jaccards = np.where(scores >= 0, jac_c, jac_nc)
    combined = jaccards * np.abs(scores)
    return eligible, scores, jaccards, combined


@torch.no_grad()
def collect_multi_anchor_residuals(
    model,
    prompt_records: list[tuple[list[int], bool, object]],
    anchors: list[tuple[str, int]],
    target_layers: list[int],
) -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray, list[object]]:
    anchor_pos = {label: pos + 1 for label, pos in anchors}
    H: dict[int, dict[str, list[torch.Tensor]]] = {
        layer: {label: [] for label, _ in anchors} for layer in target_layers
    }
    pos_mask: list[bool] = []
    pair_refs: list[object] = []

    zero = torch.zeros(model.cfg.d_model)
    for ids, is_pos, pair in tqdm(prompt_records, desc="Capturing residuals"):
        resid_cache: dict[int, dict[str, torch.Tensor]] = {
            layer: {} for layer in target_layers
        }

        hooks = []
        for layer in target_layers:
            def _hook(acts, hook, _l=layer):
                for label, sink_pos in anchor_pos.items():
                    if sink_pos < acts.shape[1]:
                        resid_cache[_l][label] = acts[0, sink_pos, :].detach().float().cpu().clone()
                return acts

            hooks.append((f"blocks.{layer}.{model.feature_input_hook}", _hook))

        input_ids = tokenize_qwen_input(ids, model.tokenizer, model.cfg.device).unsqueeze(0)
        model.run_with_hooks(input_ids, fwd_hooks=hooks)

        for layer in target_layers:
            for label, _ in anchors:
                H[layer][label].append(resid_cache[layer].get(label, zero).detach().clone())
        pos_mask.append(is_pos)
        if is_pos:
            pair_refs.append(pair)

    H_np = {
        layer: {label: torch.stack(vecs).float().cpu().numpy() for label, vecs in by_anchor.items()}
        for layer, by_anchor in H.items()
    }
    return H_np, np.array(pos_mask, dtype=bool), pair_refs


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", default="carry")
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", required=True)
    parser.add_argument(
        "--anchors",
        default="anchor_rank1_pos7:7,anchor_rank2_pos6:6,anchor_rank3_pos8:8,anchor_rank4_pos10:10,anchor_rank5_pos9:9,anchor_rank6_pos5:5",
        help="Comma-separated label:raw_pos anchors",
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--template", default="T0")
    parser.add_argument("--max_pairs", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        default="runs/concept_localization/carry/carry_T0/all_anchor_sweep_all_layers_T0",
    )
    args = parser.parse_args()

    if args.concept != "carry":
        raise ValueError("Multi-anchor all-feature grids currently support carry only")

    target_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    anchors_raw = _parse_anchor_specs(args.anchors)
    anchors = [(_safe_anchor_label(label, pos), pos) for label, pos in anchors_raw]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading model %s", args.model)
    dtype = parse_dtype(args.dtype)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=get_default_device()
    )
    model.eval()

    pairs = _load_concept(args.concept, args.n, args.seed)
    random.Random(args.seed).shuffle(pairs)
    pairs = [p for p in pairs if p.template == args.template][: args.max_pairs]
    if not pairs:
        raise ValueError(f"No pairs for template {args.template}")

    prompt_records = []
    valid_pairs = []
    for pair in pairs:
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids
        if len(ids_pos) != len(ids_neg):
            continue
        prompt_records.append((ids_pos, True, pair))
        prompt_records.append((ids_neg, False, pair))
        valid_pairs.append(pair)

    log.info(
        "Running %d prompts from %d %s pairs at %d anchors and %d layers",
        len(prompt_records),
        len(valid_pairs),
        args.template,
        len(anchors),
        len(target_layers),
    )

    H, pos_mask, pair_refs = collect_multi_anchor_residuals(
        model, prompt_records, anchors, target_layers
    )

    all_ranked: dict[str, list[dict]] = {label: [] for label, _ in anchors}
    top_acts: dict[str, dict[str, np.ndarray]] = {label: {} for label, _ in anchors}

    for layer in target_layers:
        for label, pos in anchors:
            try:
                acts_np = apply_transcoder_all(model, layer, H[layer][label])
            except (IndexError, KeyError, AttributeError):
                log.warning("No transcoder at layer %d — skipping", layer)
                continue

            eligible, scores, jaccards, combined = _score_features(acts_np, pos_mask)
            anchor_dir = out_dir / label / "all_feature_grids"
            anchor_dir.mkdir(parents=True, exist_ok=True)
            grids = np.stack([_acts_to_grid(acts_np[:, feat_id], pair_refs) for feat_id in eligible])
            np.savez_compressed(
                anchor_dir / f"layer_{layer:02d}_all_feature_grids.npz",
                feat_ids=eligible.astype(np.int32),
                grids=grids.astype(np.float32),
                scores=scores[eligible].astype(np.float32),
                jaccards=jaccards[eligible].astype(np.float32),
                combined=combined[eligible].astype(np.float32),
                anchor_label=np.array(label),
                anchor_pos=np.array(pos, dtype=np.int32),
            )

            top_idx = eligible[np.argsort(combined[eligible])[::-1][: args.top_k]]
            for feat_id in top_idx:
                all_ranked[label].append(
                    {
                        "anchor": label,
                        "anchor_pos": pos,
                        "layer": layer,
                        "feat_id": int(feat_id),
                        "score": round(float(scores[feat_id]), 6),
                        "jaccard": round(float(jaccards[feat_id]), 4),
                        "combined": round(float(combined[feat_id]), 6),
                    }
                )
                top_acts[label][f"L{layer}_F{int(feat_id)}"] = acts_np[:, feat_id]

            log.info(
                "%s layer %2d: active=%d saved=%s",
                label,
                layer,
                len(eligible),
                anchor_dir / f"layer_{layer:02d}_all_feature_grids.npz",
            )

    examples = [
        {"pair_idx": i, "template": p.template, "meta": p.meta, "label_pos": p.label_pos}
        for i, p in enumerate(pair_refs)
    ]
    for label, pos in anchors:
        anchor_out = out_dir / label
        all_ranked[label].sort(key=lambda r: -abs(r["score"]) * r["jaccard"])
        (anchor_out / "sweep_ranked.json").write_text(json.dumps(all_ranked[label], indent=2))
        np.savez_compressed(
            anchor_out / "sweep_activations.npz",
            pos_mask=pos_mask,
            **top_acts[label],
        )
        with open(anchor_out / "sweep_examples.pkl", "wb") as f:
            pickle.dump(examples, f)

    manifest = {
        "concept": args.concept,
        "template": args.template,
        "layers": target_layers,
        "anchors": [{"label": label, "pos": pos} for label, pos in anchors],
        "n_pairs": len(pair_refs),
        "out_dir": str(out_dir),
    }
    (out_dir / "multi_anchor_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
