"""Generic concept localization runner.

Runs the full pipeline for any registered concept:
  generate pairs → extract deltas → analyse → save → plot

Usage
-----
    python -m experiments.concept_localization.run_concept --concept gcd
    python -m experiments.concept_localization.run_concept --concept residue_class --n 300
    python -m experiments.concept_localization.run_concept --concept negation_scope --skip_features

Registered concepts
-------------------
    gcd                 gcd(a,7)=7 vs gcd(a,7)=1
    residue_class       a%7=1 vs a%7=6
    transitive_ordering a>b>c True vs False
    conservation        drop/bounce energy violation
    causal_direction    A causes B vs B causes A
    negation_scope      n not less than m True vs False
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.analyze import (
    compute_sharpness,
    project_onto_features,
)
from experiments.concept_localization.causal_analysis import run_causal_analysis
from experiments.concept_localization.extract_deltas_generic import extract_layer_deltas_generic
from experiments.concept_localization.visualize import (
    plot_causal_efficiency,
    plot_causal_overlay,
    plot_causal_scores,
    plot_feature_projections,
    plot_norm_and_alignment,
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
    raise ValueError(f"Unknown concept: {name!r}")


CONCEPTS = [
    "gcd",
    "residue_class",
    "transitive_ordering",
    "conservation",
    "causal_direction",
    "negation_scope",
]


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", required=True, choices=CONCEPTS)
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
        "--anchor_mode",
        default="diff",
        help=(
            "diff (default): anchor at last token where pos/neg differ. "
            "last: anchor at final token. "
            "delimiter: anchor at the last '=', ':', or newline. "
            "pos_from_end:<n>: anchor n tokens from the end — set this to the "
            "sweep-determined best position after running run_positional_attribution --sweep "
            "(e.g. pos_from_end:2 anchors at the third-to-last token)."
        ),
    )
    parser.add_argument(
        "--skip_features", action="store_true", help="Skip transcoder feature projection (faster)"
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

    suffix = f"_{args.anchor_mode}" if args.anchor_mode != "diff" else ""
    out_dir = Path(args.out_dir or f"runs/concept_localization/{args.concept}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_default_device()
    dtype = parse_dtype(args.dtype)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    log.info("Generating %d pairs/template for concept '%s'", args.n, args.concept)
    pairs = _load_concept(args.concept, args.n, args.seed)
    templates = list(dict.fromkeys(p.template for p in pairs))
    log.info(
        "Generated %d pairs total across %d templates: %s", len(pairs), len(templates), templates
    )
    for t in templates:
        t_pairs = [p for p in pairs if p.template == t]
        if t_pairs:
            log.info("  %s (%d pairs)  e.g. pos=%r", t, len(t_pairs), t_pairs[0].prompt_pos)

    # ── 2. Model ──────────────────────────────────────────────────────────────
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

    # ── 3. Delta extraction ───────────────────────────────────────────────────
    log.info("Extracting per-layer residual-stream deltas…")
    log.info("Anchor mode: %s", args.anchor_mode)
    layer_results = extract_layer_deltas_generic(
        model, pairs, layers, device, dtype, per_template=True, anchor_mode=args.anchor_mode
    )
    ld = layer_results["all"]

    # ── 4. Sharpness ──────────────────────────────────────────────────────────
    sharpness = compute_sharpness(ld)
    log.info(
        "Peak layer: %d  sharpness_index: %.3f", sharpness.peak_layer, sharpness.sharpness_index
    )

    # Template consistency: cosine sim of delta at peak layer across templates
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

    # ── 5. Causal analysis ────────────────────────────────────────────────────
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
        )
        log.info("Causal analysis done (n_pairs=%d)", causal["all"].n_pairs)

    # ── 6. Feature projection ─────────────────────────────────────────────────
    projections: dict = {}
    if not args.skip_features:
        log.info("Projecting delta onto transcoder features…")
        projections = project_onto_features(model, ld, top_k=args.top_k)

    # Normalised delta norms (‖δ_l‖ / E[‖h_l‖]) for plotting
    mean_act_norms = {l: v for l, v in ld.mean_act_norm.items()} if ld.mean_act_norm else {}
    delta_norms_raw = {l: ld.delta[l].norm().item() for l in layers if l in ld.delta}
    delta_norms_plot = (
        {l: delta_norms_raw[l] / mean_act_norms.get(l, 1.0) for l in delta_norms_raw}
        if mean_act_norms
        else delta_norms_raw
    )

    # ── 6. Save ───────────────────────────────────────────────────────────────
    results_json = {
        "config": {
            "concept": args.concept,
            "model": args.model,
            "n_pairs": len(pairs),
            "n_pairs_used": ld.n_pairs,
            "skipped": ld.skipped,
            "templates": tmpl_keys,
            "anchor_mode": args.anchor_mode,
            "top_k": args.top_k,
        },
        "sharpness": {
            "peak_layer": sharpness.peak_layer,
            "sharpness_index": round(sharpness.sharpness_index, 4),
            "norm_by_layer": {
                str(l): round(v, 4) for l, v in zip(sharpness.layers, sharpness.norms, strict=False)
            },
            "inter_layer_cos": {
                f"{sharpness.layers[i]}-{sharpness.layers[i + 1]}": round(v, 4)
                for i, v in enumerate(sharpness.inter_layer_cos)
            },
        },
        "template_consistency": consistency,
        "mean_act_norm": {str(l): round(v, 4) for l, v in mean_act_norms.items()},
        "top_features_by_layer": {
            str(layer): [
                {"feature_id": m.feature_id, "cos_sim": round(m.cos_sim, 4)} for m in matches
            ]
            for layer, matches in projections.items()
        },
        "causal": (
            {
                key: {
                    "n_pairs": cs.n_pairs,
                    "patching_mean": {str(l): round(v, 5) for l, v in cs.patching_mean.items()},
                    "patching_std": {str(l): round(v, 5) for l, v in cs.patching_std.items()},
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

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    plot_norm_and_alignment(layer_results, out_dir / "norm_trajectory.png", concept=args.concept)
    log.info("Saved norm_trajectory.png")

    if projections:
        plot_feature_projections(
            projections,
            out_dir / "feature_projections_scatter.png",
            top_k=args.top_k,
            concept=args.concept,
        )
        log.info("Saved feature projection plots")

    if causal is not None:
        plot_causal_scores(
            causal,
            delta_norms_raw,
            out_dir / "causal_scores.png",
            concept=args.concept,
            mean_act_norms=mean_act_norms or None,
        )
        log.info("Saved causal_scores.png")
        plot_causal_overlay(
            causal,
            delta_norms_raw,
            out_dir / "causal_overlay.png",
            concept=args.concept,
            mean_act_norms=mean_act_norms or None,
        )
        log.info("Saved causal_overlay.png")
        plot_causal_efficiency(
            causal,
            delta_norms_raw,
            out_dir / "causal_efficiency.png",
            concept=args.concept,
        )
        log.info("Saved causal_efficiency.png")

    log.info("Done. All outputs in %s", out_dir)


if __name__ == "__main__":
    main()
