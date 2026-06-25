"""Cache per-anchor residual streams for a concept dataset.

Loads a registered concept dataset, runs positive and negative prompts through
the model at the requested anchor, and saves raw residual-stream vectors for
the requested layers. No feature ranking or selected feature activation cache is
written here; downstream analyses should apply transcoders to sweep_residuals.npz
for whichever features and ranking criteria they need.

Usage
-----
    python -m experiments.concept_localization.pipeline.run_concept_sweep --concept carry --layers all
    python -m experiments.concept_localization.pipeline.run_concept_sweep --concept gcd --layers 4,5,17,18,19 --anchor delimiter
    python -m experiments.concept_localization.pipeline.run_concept_sweep --concept decimal_termination --layers 17,18,19,20 --anchor digit_1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.analyze import collect_layer_residuals_batched as collect_layer_residuals
from experiments.concept_localization.extract_deltas_generic import (
    _resolve_anchor,
)
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from scripts.model_config import add_model_config_arg, default_model, default_transcoder_set, resolve_model_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_concept_sweep")

_MODEL = default_model()
_TRANSCODER_SET = default_transcoder_set()

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


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sweep_cache_metadata(args, target_layers: list[int], prompts: list[str], examples: list[dict]) -> dict:
    payload = {
        "concept": args.concept,
        "model": args.model,
        "transcoder_set": args.transcoder_set,
        "dtype": args.dtype,
        "layers": target_layers,
        "anchor": args.anchor or "delimiter",
        "n": args.n,
        "template": args.template,
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "prompts": prompts,
        "examples": examples,
    }
    return {
        "version": 1,
        "hash": _stable_hash(payload),
        "payload": _jsonable(payload),
    }


def _load_concept(name: str, n: int, seed: int):
    mod = importlib.import_module(f"experiments.concept_localization.concept_datasets.{name}_dataset")
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


def _build_inputs(model, pairs, anchor_mode, max_pairs):
    inputs, pos_mask, prompts, examples = [], [], [], []
    for pair_idx, pair in enumerate(pairs[:max_pairs]):
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids
        if len(ids_pos) != len(ids_neg):
            continue

        anchor = _resolve_anchor(ids_pos, model.tokenizer, anchor_mode, None, None)

        for ids, is_pos, prompt in [
            (ids_pos, True, pair.prompt_pos),
            (ids_neg, False, pair.prompt_neg),
        ]:
            inputs.append((ids, anchor))
            pos_mask.append(is_pos)
            prompts.append(prompt)

        examples.append(
            {
                "pair_idx": len(examples),
                "source_pair_idx": pair_idx,
                "template": pair.template,
                "meta": pair.meta,
                "label_pos": pair.label_pos,
                "predict_pos": pair.predict_pos,
                "predict_neg": pair.predict_neg,
                "prompt_pos": pair.prompt_pos,
                "prompt_neg": pair.prompt_neg,
                "anchor": anchor,
            }
        )

    return inputs, np.array(pos_mask, dtype=bool), prompts, examples


@torch.no_grad()
def cache_sweep_residuals(
    model,
    pairs,
    target_layers: list[int],
    anchor_mode: str = "delimiter",
    max_pairs: int = 200,
) -> tuple[dict[int, np.ndarray], np.ndarray, list[str], list[dict]]:
    """Return residual cache plus prompt/example metadata for one anchor."""
    inputs, pos_mask, prompts, examples = _build_inputs(
        model, pairs, anchor_mode, max_pairs
    )
    if len(inputs) == 0:
        raise ValueError("No valid pairs found; check anchor_mode and pair lengths")

    log.info(
        "Collecting residuals for %d prompts (%d pos, %d neg) at %d layers",
        len(inputs),
        pos_mask.sum(),
        (~pos_mask).sum(),
        len(target_layers),
    )
    residuals = collect_layer_residuals(model, inputs, target_layers)
    return residuals, pos_mask, prompts, examples


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", required=True, choices=CONCEPTS)
    add_model_config_arg(parser)
    parser.add_argument("--model", default=None)
    parser.add_argument("--transcoder_set", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices (e.g. '17,18,19,20') or 'all'",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir", default=None, help="Output dir (default: runs/concept_localization/<concept>)"
    )
    args = parser.parse_args()
    resolve_model_args(args)

    if args.layers == "all":
        target_layers = list(range(36))
    else:
        target_layers = [int(x.strip()) for x in args.layers.split(",")]
    out_dir = (
        Path(args.out_dir) if args.out_dir else Path(f"runs/concept_localization/{args.concept}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

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

    residuals, pos_mask, prompts, sweep_examples = cache_sweep_residuals(
        model,
        pairs,
        target_layers,
        anchor_mode=anchor_mode,
        max_pairs=args.max_pairs,
    )

    residuals_path = out_dir / "sweep_residuals.npz"
    np.savez_compressed(
        residuals_path,
        pos_mask=pos_mask,
        prompts=np.array(prompts, dtype=object),
        layers=np.array(sorted(residuals), dtype=np.int32),
        **{f"H_L{layer}": H for layer, H in residuals.items()},
    )
    log.info("Saved residual cache for %d layers → %s", len(residuals), residuals_path)

    # Save dataset examples used to build the residual cache.
    import pickle

    examples_path = out_dir / "sweep_dataset_examples.pkl"
    with open(examples_path, "wb") as f:
        pickle.dump(sweep_examples, f)
    log.info("Saved sweep dataset examples → %s", examples_path)

    metadata = _sweep_cache_metadata(args, target_layers, prompts, sweep_examples)
    metadata_path = out_dir / "sweep_residuals.meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    log.info("Saved residual cache metadata hash %s → %s", metadata["hash"], metadata_path)
    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
