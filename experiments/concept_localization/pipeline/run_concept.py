"""Generic concept localization runner.

Runs the full pipeline for any registered concept:
  generate pairs → extract deltas → analyse → save → plot

Usage
-----
    python -m experiments.concept_localization.pipeline.run_concept --concept gcd
    python -m experiments.concept_localization.pipeline.run_concept --concept residue_class --n 300
    python -m experiments.concept_localization.pipeline.run_concept --concept negation_scope --skip_features

Registered concepts
-------------------
    carry                   units digit carry in multi-digit addition
    gcd                     gcd(a,7)=7 vs gcd(a,7)=1
    residue_class           a%7=1 vs a%7=6
    transitive_ordering     a>b>c True vs False
    conservation            drop/bounce energy violation
    causal_direction        A causes B vs B causes A
    negation_scope          n not less than m True vs False
    balanced_parentheses    balanced vs unbalanced bracket sequences
    decimal_termination     terminating vs non-terminating decimal
    doppler_shift           approaching vs receding source frequency
    dot_product_sign        positive vs negative dot product
    geometric_series        convergent vs divergent geometric series
    momentum_conservation   momentum conserved vs violated
    perfect_square          perfect square vs non-perfect square
    syllogism               valid vs invalid syllogism
    triangle_inequality     valid vs invalid triangle side lengths
    wave_interference       constructive vs destructive interference
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import random
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from experiments.concept_localization.analyze import (
    collect_layer_residuals,
    compute_sharpness,
    project_onto_E_dec_model,
)
from experiments.concept_localization.causal_analysis import run_causal_analysis
from experiments.concept_localization.extract_deltas_generic import (
    extract_layer_deltas_generic,
    resolve_anchor_token,
)
from experiments.concept_localization.plots.plot_anchor_analysis import (
    load_emergence,
    top_k_anchors,
)
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_concept")

_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"


# ── concept registry ──────────────────────────────────────────────────────────
def _load_concept(name: str, n_per_template: int, seed: int):
    if name == "carry":
        from data.concept_datasets.carry_dataset import generate_carry_pairs

        return generate_carry_pairs(n_per_template, seed=seed)
    if name == "gcd":
        from data.concept_datasets.gcd_dataset import generate_gcd_pairs

        return generate_gcd_pairs(n_per_template, seed=seed)
    if name == "residue_class":
        from data.concept_datasets.residue_class_dataset import generate_residue_pairs

        return generate_residue_pairs(n_per_template, seed=seed)
    if name == "transitive_ordering":
        from data.concept_datasets.transitive_ordering_dataset import generate_ordering_pairs

        return generate_ordering_pairs(n_per_template, seed=seed)
    if name == "conservation":
        from data.concept_datasets.conservation_dataset import generate_conservation_pairs

        return generate_conservation_pairs(n_per_template, seed=seed)
    if name == "causal_direction":
        from data.concept_datasets.causal_direction_dataset import generate_causal_pairs

        return generate_causal_pairs(n_per_template, seed=seed)
    if name == "negation_scope":
        from data.concept_datasets.negation_scope_dataset import generate_negation_pairs

        return generate_negation_pairs(n_per_template, seed=seed)
    if name == "balanced_parentheses":
        from data.concept_datasets.balanced_parentheses_dataset import generate_parentheses_pairs

        return generate_parentheses_pairs(n_per_template, seed=seed)
    if name == "decimal_termination":
        from data.concept_datasets.decimal_termination_dataset import generate_decimal_pairs

        return generate_decimal_pairs(n_per_template, seed=seed)
    if name == "doppler_shift":
        from data.concept_datasets.doppler_shift_dataset import generate_doppler_pairs

        return generate_doppler_pairs(n_per_template, seed=seed)
    if name == "dot_product_sign":
        from data.concept_datasets.dot_product_sign_dataset import generate_dot_pairs

        return generate_dot_pairs(n_per_template, seed=seed)
    if name == "geometric_series":
        from data.concept_datasets.geometric_series_dataset import generate_geometric_pairs

        return generate_geometric_pairs(n_per_template, seed=seed)
    if name == "momentum_conservation":
        from data.concept_datasets.momentum_conservation_dataset import generate_momentum_pairs

        return generate_momentum_pairs(n_per_template, seed=seed)
    if name == "perfect_square":
        from data.concept_datasets.perfect_square_dataset import generate_perfect_square_pairs

        return generate_perfect_square_pairs(n_per_template, seed=seed)
    if name == "syllogism":
        from data.concept_datasets.syllogism_dataset import generate_syllogism_pairs

        return generate_syllogism_pairs(n_per_template, seed=seed)
    if name == "triangle_inequality":
        from data.concept_datasets.triangle_inequality_dataset import generate_triangle_pairs

        return generate_triangle_pairs(n_per_template, seed=seed)
    if name == "wave_interference":
        from data.concept_datasets.wave_interference_dataset import generate_wave_pairs

        return generate_wave_pairs(n_per_template, seed=seed)
    raise ValueError(f"Unknown concept: {name!r}")


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
    "doppler_shift",
    "dot_product_sign",
    "geometric_series",
    "momentum_conservation",
    "perfect_square",
    "syllogism",
    "triangle_inequality",
    "wave_interference",
]

SYMBOLIC_SUBSET = [
    "carry",
    "residue_class",
    "gcd",
    "perfect_square",
    "decimal_termination",
    "dot_product_sign",
    "triangle_inequality",
    "transitive_ordering",
    "balanced_parentheses",
    "syllogism",
]


def _get_dataset_attr(concept: str, attr: str, default=None):
    """Load an attribute from a concept's dataset module, returning default if absent."""
    try:
        mod = importlib.import_module(f"data.concept_datasets.{concept}_dataset")
        return getattr(mod, attr, default)
    except ImportError:
        return default


_FEATURE_PROJECTION_SCORE_MODES = ("dec", "enc+dec")


def _run_feature_projection_plots(
    model,
    pairs: list,
    anchor_mode: str,
    out_dir: Path,
    top_k: int,
    concept: str,
) -> None:
    """Run delta_feature_projections.run_one_mode for dec and enc+dec, saving edec_topk_grid.pdf.

    Collects residual streams once and reuses them for both score modes to avoid
    running the model forward pass twice.  Skips attribution-graph survival filter
    (survival_set=None) since graphs may not exist yet at this pipeline stage.
    """
    from experiments.concept_localization.pipeline.delta_feature_projections import (
        _build_inputs_and_examples,
        run_one_mode,
    )
    from experiments.concept_localization.sweep_utils import apply_transcoder_all

    inputs, examples = _build_inputs_and_examples(model, pairs, anchor_mode, max_pairs=None)
    if not inputs:
        log.info("No valid pairs for feature projection plots (all sequence lengths mismatched)")
        return

    all_layers = list(range(model.cfg.n_layers))
    log.info(
        "Collecting residuals for feature projection plots (%d prompts, %d layers)…",
        len(inputs), len(all_layers),
    )
    H_scan = collect_layer_residuals(model, inputs, all_layers)

    active_features: dict[int, set[int]] = {}
    for layer in all_layers:
        acts = apply_transcoder_all(model, layer, H_scan[layer])
        active_ids = set(np.where(acts.max(axis=0) > 0)[0].tolist())
        active_features[layer] = active_ids
    total_active = sum(len(v) for v in active_features.values())
    log.info("Active features: %d across %d layers", total_active, len(all_layers))

    for score_mode in _FEATURE_PROJECTION_SCORE_MODES:
        log.info("delta_feature_projections: score_mode=%s …", score_mode)
        run_one_mode(
            anchor_dir=out_dir,
            model=model,
            inputs=inputs,
            examples=examples,
            active_features=active_features,
            survival_set=None,
            score_mode=score_mode,
            top_k=top_k,
            concept=concept,
            H_cached=H_scan,
        )
        mode_suffix = score_mode.replace("+", "_")
        log.info(
            "Saved → %s",
            out_dir / "sweep" / f"delta_feature_projections_{mode_suffix}" / "edec_topk_grid.pdf",
        )


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--concept",
        required=True,
        choices=CONCEPTS + ["all", "symbolic"],
        help="Concept name, 'all' to run every concept, or 'symbolic' for the symbolic subset",
    )
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n", type=int, default=100, help="Pairs per template")
    parser.add_argument("--top_k", type=int, default=15, help="Top-k features per layer")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: runs/concept_localization/<concept>)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--anchor_modes",
        default=None,
        help=(
            "Which token positions to run the full pipeline for. "
            "Model and pairs are loaded once; each anchor saves to its own output directory. "
            "Omit to run the single delimiter anchor. "
            "'topN' (e.g. 'top5') — top-N anchors by non-monotonicity from emergence.npy "
            "(requires make_gif to have been run; N is clamped to the number of non-zero anchors). "
            "Comma-separated integers (e.g. '5,6,7') — explicit 0-indexed token positions."
        ),
    )
    parser.add_argument(
        "--skip_features", action="store_true", help="Skip transcoder feature projection (faster)"
    )
    parser.add_argument(
        "--feature_score_mode", default="enc+dec", choices=["dec", "enc", "enc+dec"],
        help="Ranking mode for top-k feature projection: dec=decoder cosine, "
             "enc=encoder cosine, (default) enc+dec=sum of both normalised cosine similarities",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Filter pairs to a single template (e.g. 'T0'). Default: use all templates.",
    )
    parser.add_argument(
        "--causal", action="store_true", help="Run activation patching + gradient-dot-delta"
    )
    parser.add_argument(
        "--causal_pairs",
        type=int,
        default=None,
        help="Max pairs for causal analysis (default: all, but 50 is usually enough)",
    )
    args = parser.parse_args()

    if args.concept in ("all", "symbolic"):
        batch = CONCEPTS if args.concept == "all" else SYMBOLIC_SUBSET
        base = "symbolic_subset" if args.concept == "symbolic" else None
        for concept in batch:
            log.info("=" * 60)
            log.info("Running concept: %s", concept)
            log.info("=" * 60)
            args.concept = concept
            args.out_dir = None
            _run_single(args, base_subdir=base)
        return

    _run_single(args)


def _run_single(args, base_subdir: str | None = None) -> None:
    device = get_default_device()
    dtype = parse_dtype(args.dtype)
    anchor_factory = None

    # ── 1. Dataset (loaded once) ──────────────────────────────────────────────
    log.info("Generating %d pairs/template for concept '%s'", args.n, args.concept)
    pairs = _load_concept(args.concept, args.n, args.seed)
    random.Random(args.seed).shuffle(pairs)
    if args.template:
        pairs = [p for p in pairs if p.template == args.template]
        log.info("Filtered to template '%s': %d pairs", args.template, len(pairs))
    templates = list(dict.fromkeys(p.template for p in pairs))
    log.info(
        "Generated %d pairs total across %d templates: %s", len(pairs), len(templates), templates
    )
    for t in templates:
        t_pairs = [p for p in pairs if p.template == t]
        if t_pairs:
            log.info("  %s (%d pairs)  e.g. pos=%r", t, len(t_pairs), t_pairs[0].prompt_pos)

    # ── 2. Model (loaded once) ────────────────────────────────────────────────
    log.info("Loading model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()

    n_layers = model.cfg.n_layers
    layers = list(range(n_layers))
    log.info("Model loaded: %d layers", n_layers)

    # ── Determine which anchors to run the full pipeline for ──────────────────
    if args.anchor_modes:
        import re as _re
        _top_m = _re.fullmatch(r"top(\d+)", args.anchor_modes)
        if _top_m:
            k = int(_top_m.group(1))
            em = load_emergence(args.concept)
            if em is None:
                raise FileNotFoundError(
                    f"emergence.npy not found for concept {args.concept!r}; cannot resolve "
                    f"--anchor_modes top{k}. Run the emergence stage first, for example: "
                    f"python -m experiments.concept_localization.concept_emergence_gif.make_gif "
                    f"--concept {args.concept}, or pass explicit valid anchor positions."
                )
            else:
                norms_raw = em["norms_raw"]
                n_nonzero = sum(
                    1 for a in range(norms_raw.shape[0]) if norms_raw[a].max() > 1e-8
                )
                actual_k = min(k, n_nonzero)
                if actual_k < k:
                    log.info(
                        "top%d requested but only %d non-zero anchors in emergence.npy — "
                        "using top%d",
                        k, n_nonzero, actual_k,
                    )
                top_anchors = top_k_anchors(em, args.concept, k=actual_k)
                anchors_to_run = [str(idx) for idx, _, _ in top_anchors]
                log.info(
                    "top%d anchors from emergence.npy for '%s': positions %s",
                    actual_k, args.concept, anchors_to_run,
                )
        else:
            anchors_to_run = [m.strip() for m in args.anchor_modes.split(",")]
    else:
        anchors_to_run = ["delimiter"]

    log.info("Anchor modes to run: %s", anchors_to_run)
    multi = len(anchors_to_run) > 1

    # ── Per-anchor loop ───────────────────────────────────────────────────────
    for anchor_mode in anchors_to_run:
        if multi:
            log.info("─" * 50)
            log.info("Anchor: %s", anchor_mode)

        # Output directory: when running multiple anchors, always use the
        # auto-suffixed path; when running a single anchor, respect --out_dir.
        suffix = f"_{anchor_mode}" if anchor_mode != "delimiter" else ""
        if args.out_dir and not multi:
            out_dir = Path(args.out_dir)
        elif base_subdir:
            out_dir = Path(f"runs/concept_localization/{base_subdir}/{args.concept}{suffix}")
        elif args.template:
            out_dir = Path(
                f"runs/concept_localization/{args.concept}/{args.concept}_{args.template}{suffix}"
            )
        else:
            out_dir = Path(f"runs/concept_localization/{args.concept}{suffix}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── 3. Delta extraction ───────────────────────────────────────────────
        log.info("Extracting per-layer residual-stream deltas (anchor=%s)…", anchor_mode)
        layer_results = extract_layer_deltas_generic(
            model,
            pairs,
            layers,
            device,
            dtype,
            per_template=True,
            anchor_mode=anchor_mode,
            anchor_factory=anchor_factory,
        )
        ld = layer_results["all"]

        # ── 4. Sharpness ──────────────────────────────────────────────────────
        sharpness = compute_sharpness(ld)
        log.info(
            "Peak layer: %d  sharpness_index: %.3f", sharpness.peak_layer, sharpness.sharpness_index
        )

        peak = sharpness.peak_layer
        tmpl_keys = [k for k in layer_results if k != "all"]
        consistency: dict[str, float] = {}
        if len(tmpl_keys) >= 2 and peak in ld.delta:
            ref = ld.delta[peak]
            ref_norm = ref.norm().item()
            for k in tmpl_keys:
                if peak in layer_results[k].delta and ref_norm > 0:
                    v = layer_results[k].delta[peak]
                    cos = (ref @ v / (ref_norm * v.norm().clamp(min=1e-8))).item()
                    consistency[k] = round(cos, 4)
                    log.info("  Template %s  cos with 'all' at peak L%d: %.3f", k, peak, cos)

        # ── 5. Causal analysis ────────────────────────────────────────────────
        causal: object = None
        if args.causal:
            max_pairs = args.causal_pairs
            log.info(
                "Running causal analysis (max_pairs=%s)…",
                max_pairs if max_pairs is not None else "all",
            )
            causal = run_causal_analysis(
                model,
                pairs,
                ld.delta,
                layers,
                device,
                dtype,
                max_pairs=max_pairs,
                anchor_mode=anchor_mode,
                anchor_factory=anchor_factory,
            )
            log.info("Causal analysis done (n_pairs=%d)", causal["all"].n_pairs)

        # ── 6. Feature projection ─────────────────────────────────────────────
        if not args.skip_features:
            log.info("Projecting delta onto transcoder decoder directions (E_dec)…")
            edec_features = project_onto_E_dec_model(
                model,
                ld.delta,
                top_k=args.top_k,
                score_mode=args.feature_score_mode,
            )
            edec_path = out_dir / "edec_features.json"
            with open(edec_path, "w") as f:
                json.dump(
                    {
                        str(layer): [
                            {
                                "feature_id": match.feature_id,
                                "projection": match.projection,
                                "cos_sim": match.cos_sim,
                                "layer": match.layer,
                                "enc_cos_sim": match.enc_cos_sim,
                            }
                            for match in matches
                        ]
                        for layer, matches in edec_features.items()
                    },
                    f,
                    indent=2,
                )
            log.info("Saved feature projection → %s", edec_path)

            _run_feature_projection_plots(
                model=model,
                pairs=pairs,
                anchor_mode=anchor_mode,
                out_dir=out_dir,
                top_k=args.top_k,
                concept=args.concept,
            )

        mean_act_norms = {l: v for l, v in ld.mean_act_norm.items()} if ld.mean_act_norm else {}
        delta_norms_raw = {l: ld.delta[l].norm().item() for l in layers if l in ld.delta}

        # ── 7. Save ───────────────────────────────────────────────────────────
        anchor_pos, anchor_tok = resolve_anchor_token(
            pairs[0].prompt_pos,
            model.tokenizer,
            anchor_mode,
            anchor_factory=anchor_factory,
            pair=pairs[0],
        )
        results_json = {
            "config": {
                "concept": args.concept,
                "model": args.model,
                "n_pairs": len(pairs),
                "n_pairs_used": ld.n_pairs,
                "skipped": ld.skipped,
                "templates": tmpl_keys,
                "anchor_mode": anchor_mode,
                "anchor_pos": anchor_pos,
                "anchor_token": anchor_tok,
                "top_k": args.top_k,
            },
            "sharpness": {
                "peak_layer": sharpness.peak_layer,
                "sharpness_index": round(sharpness.sharpness_index, 4),
                "normalised": sharpness.normalised,
                "norm_by_layer": {
                    str(l): round(v, 4)
                    for l, v in zip(sharpness.layers, sharpness.norms, strict=False)
                },
                "inter_layer_cos": {
                    f"{sharpness.layers[i]}-{sharpness.layers[i + 1]}": round(v, 4)
                    for i, v in enumerate(sharpness.inter_layer_cos)
                },
            },
            "template_consistency": consistency,
            "mean_act_norm": {str(l): round(v, 4) for l, v in mean_act_norms.items()},
            "causal": (
                {
                    key: {
                        "n_pairs": cs.n_pairs,
                        "patching_mean": {
                            str(l): round(v, 5) for l, v in cs.patching_mean.items()
                        },
                        "patching_std": {
                            str(l): round(v, 5) for l, v in cs.patching_std.items()
                        },
                        "grad_dot_delta_mean": {
                            str(l): round(v, 5) for l, v in cs.grad_dot_delta_mean.items()
                        },
                        "grad_dot_delta_std": {
                            str(l): round(v, 5) for l, v in cs.grad_dot_delta_std.items()
                        },
                    }
                    for key, cs in causal.items()
                }
                if causal is not None
                else None
            ),
        }

        results_path = out_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results_json, f, indent=2)
        log.info("Saved results → %s", results_path)

        deltas_path = out_dir / "deltas.pt"
        torch.save(
            {key: {l: v for l, v in lr.delta.items()} for key, lr in layer_results.items()},
            deltas_path,
        )
        log.info("Saved delta tensors → %s", deltas_path)

        # Individual non-null diagnostic plots are intentionally not saved here.
        # run_anchor_pipeline.py writes the combined anchor_layer_summary_<template>.png
        # after the null stage, so feature projection, layer-cosine, delta trajectory,
        # and causal overlay live together on aligned layer axes.

        log.info("Done for anchor '%s'. Outputs in %s", anchor_mode, out_dir)

    log.info("All done for concept '%s'.", args.concept)


if __name__ == "__main__":
    main()
