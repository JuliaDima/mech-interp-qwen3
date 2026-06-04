"""Generic per-concept transcoder feature sweep.

Loads any registered concept's dataset, runs pos and neg prompts through the
model at target layers, and scores every transcoder feature by

    score   = mean(act | pos) − mean(act | neg)

Features are ranked by Jaccard similarity × |score|, which identifies features
that consistently separate pos from neg examples. Results are displayed as bar
charts where each bar is one example, coloured by pos (blue) or neg (red).

Usage
-----
    python scripts/sweeps/run_concept_sweep.py --concept carry --layers 19,20,21 --top_k 200
    python scripts/sweeps/run_concept_sweep.py --concept gcd --layers 4,5,17,18,19 --anchor delimiter
    python scripts/sweeps/run_concept_sweep.py --concept decimal_termination --layers 17,18,19,20 --anchor digit_1
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SWEEPS_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _SWEEPS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sweep_utils import apply_transcoder_all, resolve_anchor_from_positions, score_and_rank

from experiments.concept_localization.analyze import collect_layer_residuals
from experiments.concept_localization.extract_deltas_generic import (
    _resolve_anchor,
)
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_concept_sweep")

_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"

CONCEPTS = [
    "carry",
    "gcd",
    "residue_class",
    "transitive_ordering",
    "conservation",
    "causal_direction",
    "negation_scope",
    "balanced_parentheses",
    "decimal_termination",
    "decimal_termination_large_prime",
    "doppler_shift",
    "dot_product_sign",
    "geometric_series",
    "momentum_conservation",
    "perfect_square",
    "syllogism",
    "triangle_inequality",
    "wave_interference",
]


def _load_concept(name: str, n: int, seed: int):
    mod = importlib.import_module(f"data.concept_datasets.{name}_dataset")
    # Try different naming conventions: full name, suffix-only, then any generate_*_pairs function
    for fn in [
        f"generate_{name}_pairs",
        f"generate_{name.split('_')[-1]}_pairs",
        "generate_decimal_pairs",
        "generate_large_prime_pairs",
        "generate_wave_pairs",
    ]:
        if hasattr(mod, fn):
            return getattr(mod, fn)(n, seed=seed)
    # Fallback: find first generate_*_pairs function
    for attr_name in dir(mod):
        if attr_name.startswith("generate_") and attr_name.endswith("_pairs"):
            return getattr(mod, attr_name)(n, seed=seed)
    raise ValueError(f"Cannot find a generate function in {name}_dataset")


def _get_dataset_attr(concept: str, attr: str, default=None):
    try:
        mod = importlib.import_module(f"data.concept_datasets.{concept}_dataset")
        return getattr(mod, attr, default)
    except ImportError:
        return default


def _build_inputs(model, pairs, anchor_mode, anchor_factory, max_pairs):
    inputs, pos_mask, prompts = [], [], []
    for pair in pairs[:max_pairs]:
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids
        if len(ids_pos) != len(ids_neg):
            continue

        if anchor_factory:
            positions = anchor_factory(pair, model.tokenizer)
            anchor = resolve_anchor_from_positions(positions, anchor_mode, len(ids_pos) - 1)
        else:
            anchor = _resolve_anchor(ids_pos, model.tokenizer, anchor_mode, None, None)

        for ids, is_pos, prompt in [
            (ids_pos, True, pair.prompt_pos),
            (ids_neg, False, pair.prompt_neg),
        ]:
            inputs.append((ids, anchor))
            pos_mask.append(is_pos)
            prompts.append(prompt)

    return inputs, np.array(pos_mask, dtype=bool), prompts


def _acts_to_grid(acts: np.ndarray, pairs) -> np.ndarray:
    """Mean activation per (ones(a), ones(b)) cell for paired carry examples."""
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


@torch.no_grad()
def sweep_all_features(
    model,
    pairs,
    target_layers: list[int],
    anchor_mode: str = "delimiter",
    anchor_factory=None,
    top_k: int = 200,
    max_pairs: int = 200,
    save_all_features_dir: Path | None = None,
) -> tuple[
    list[tuple[int, int, float, float]], dict[tuple[int, int], np.ndarray], np.ndarray, list[str]
]:
    """Sweep all transcoder features, rank by Jaccard × |score|.

    Returns:
        ranked         list of (layer, feat_id, score, jaccard)
        acts_1d        dict (layer, feat_id) → (N,) activations
        pos_mask       bool array, True for pos examples
        prompts        list of formatted prompt strings
    """
    inputs, pos_mask, prompts = _build_inputs(model, pairs, anchor_mode, anchor_factory, max_pairs)
    if len(inputs) == 0:
        raise ValueError("No valid pairs found — check anchor_mode and pair lengths")

    log.info(
        "Running %d examples (%d pos, %d neg) at %d layers",
        len(inputs),
        pos_mask.sum(),
        (~pos_mask).sum(),
        len(target_layers),
    )

    H = collect_layer_residuals(model, inputs, target_layers)

    ranked: list[tuple[int, int, float, float]] = []
    acts_1d: dict[tuple[int, int], np.ndarray] = {}

    for layer in target_layers:
        if layer not in H:
            continue
        try:
            acts_np = apply_transcoder_all(model, layer, H[layer])
        except (IndexError, KeyError, AttributeError):
            log.warning("No transcoder at layer %d — skipping", layer)
            continue

        neg_mask = ~pos_mask
        scores = acts_np[pos_mask].mean(axis=0) - acts_np[neg_mask].mean(axis=0)
        active = acts_np > 0
        any_active = active.any(axis=0)
        eligible = np.where(any_active)[0]

        cm = pos_mask[:, None]
        ncm = neg_mask[:, None]
        inter_c = (active & cm).sum(axis=0).astype(np.float32)
        union_c = (active | cm).sum(axis=0).astype(np.float32)
        jac_c = np.where(union_c > 0, inter_c / union_c, 0.0)
        inter_nc = (active & ncm).sum(axis=0).astype(np.float32)
        union_nc = (active | ncm).sum(axis=0).astype(np.float32)
        jac_nc = np.where(union_nc > 0, inter_nc / union_nc, 0.0)
        jaccards = np.where(scores >= 0, jac_c, jac_nc)
        combined = jaccards * np.abs(scores)

        top_idx = eligible[np.argsort(combined[eligible])[::-1][:top_k]]
        top_feats = [(int(f), float(scores[f]), float(jaccards[f])) for f in top_idx]
        for feat_id, score, jaccard in top_feats:
            ranked.append((layer, feat_id, score, jaccard))
            acts_1d[(layer, feat_id)] = acts_np[:, feat_id]

        if save_all_features_dir is not None:
            save_all_features_dir.mkdir(parents=True, exist_ok=True)
            grids = np.stack([_acts_to_grid(acts_np[:, feat_id], pairs[: len(pos_mask) // 2]) for feat_id in eligible])
            np.savez_compressed(
                save_all_features_dir / f"layer_{layer:02d}_all_feature_grids.npz",
                feat_ids=eligible.astype(np.int32),
                grids=grids.astype(np.float32),
                scores=scores[eligible].astype(np.float32),
                jaccards=jaccards[eligible].astype(np.float32),
                combined=combined[eligible].astype(np.float32),
            )

        if len(top_feats):
            top_score = max(top_feats, key=lambda x: abs(x[1]))[1]
            log.info(
                "Layer %2d  d_tc=%d  active=%d  top_k=%d  top |score|=%.4f",
                layer,
                acts_np.shape[1],
                len(eligible),
                len(top_feats),
                abs(top_score),
            )

    ranked.sort(key=lambda x: -abs(x[2]) * x[3])
    return ranked, acts_1d, pos_mask, prompts


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", required=True, choices=CONCEPTS)
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices (e.g. '17,18,19,20')",
    )
    parser.add_argument(
        "--anchor",
        default=None,
        help="Anchor mode — uses dataset's first ANCHOR_MODE if not set, else 'delimiter'",
    )
    parser.add_argument("--n", type=int, default=100, help="Pairs per template to load")
    parser.add_argument("--template", default="T0",
                        help="Single template to sweep. Per-anchor sweeps must not mix templates.")
    parser.add_argument("--max_pairs", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=200, help="Top features to select per layer")
    parser.add_argument("--save_all_features", action="store_true",
                        help="Save every non-zero feature's 10x10 carry grid per layer")
    parser.add_argument(
        "--top_per_layer", type=int, default=5, help="Top features to display per layer"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze", action="store_true",
                        help="Run cluster analysis after sweep (no GPU needed)")
    parser.add_argument(
        "--out_dir", default=None, help="Output dir (default: runs/concept_localization/<concept>)"
    )
    args = parser.parse_args()

    target_layers = [int(x.strip()) for x in args.layers.split(",")]
    out_dir = (
        Path(args.out_dir) if args.out_dir else Path(f"runs/concept_localization/{args.concept}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_factory = None
    anchor_mode = args.anchor or "delimiter"

    log.info("Concept: %s  anchor: %s  layers: %s", args.concept, anchor_mode, target_layers)

    device = get_default_device()
    dtype = parse_dtype(args.dtype)

    log.info("Loading model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()

    log.info("Loading dataset for %s (%d pairs/template)", args.concept, args.n)
    pairs = _load_concept(args.concept, args.n, args.seed)
    random.Random(args.seed).shuffle(pairs)
    template = args.template
    if template is None:
        raise ValueError("run_concept_sweep must use a single template; use --template T0/T1/T2")
    pairs = [p for p in pairs if p.template == template]
    log.info(
        "Filtered to template %s: %d pairs. Multi-template data is only for run_concept/causal plots.",
        template,
        len(pairs),
    )

    ranked, acts_1d, pos_mask, prompts = sweep_all_features(
        model,
        pairs,
        target_layers,
        anchor_mode=anchor_mode,
        anchor_factory=anchor_factory,
        top_k=args.top_k,
        max_pairs=args.max_pairs,
        save_all_features_dir=(out_dir / "all_feature_grids") if args.save_all_features else None,
    )

    ranked_path = out_dir / "sweep_ranked.json"
    with open(ranked_path, "w") as f:
        json.dump(
            [
                {"layer": l, "feat_id": fi, "score": round(s, 6), "jaccard": round(j, 4)}
                for l, fi, s, j in ranked
            ],
            f,
            indent=2,
        )
    log.info("Saved ranking → %s", ranked_path)

    # Save activations for top-k features
    activations_path = out_dir / "sweep_activations.npz"
    np.savez_compressed(
        activations_path,
        pos_mask=pos_mask,
        **{f"L{layer}_F{feat_id}": acts for (layer, feat_id), acts in acts_1d.items()},
    )
    log.info("Saved %d feature activations → %s", len(acts_1d), activations_path)
    if args.save_all_features:
        log.info("Saved all non-zero feature grids → %s", out_dir / "all_feature_grids")

    # Save example metadata for alignment with theoretical analysis
    import pickle

    examples_path = out_dir / "sweep_examples.pkl"
    sweep_examples = []
    n_pairs = len(pos_mask) // 2
    for i, pair in enumerate(pairs[:n_pairs]):
        sweep_examples.append(
            {
                "pair_idx": i,
                "template": pair.template,
                "meta": pair.meta,
                "label_pos": pair.label_pos,
            }
        )
    with open(examples_path, "wb") as f:
        pickle.dump(sweep_examples, f)
    log.info("Saved example metadata → %s", examples_path)
    log.info("Done. Outputs in %s", out_dir)

    if args.analyze:
        from analyze_sweep_clusters import run_analysis
        log.info("Running cluster analysis (T0, k=6)…")
        run_analysis(out_dir, template=args.template or "T0",
                     top_k=min(args.top_k, 100), n_clusters=6)


if __name__ == "__main__":
    main()
