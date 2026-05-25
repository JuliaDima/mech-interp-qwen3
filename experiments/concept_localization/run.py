"""Concept localization: where does the model internalize 'carry'?

Pipeline
--------
1. Generate controlled carry-contrast pairs (units digit of second operand varies, the rest is identical).
2. Extract residual-stream deltas at every layer across all pairs and templates.
3. Compute sharpness metrics (norm trajectory, inter-layer cosine sim, peak layer).
4. Project delta onto transcoder W_enc to find top-k carry features per layer.
5. Check template consistency: do T0/T1/T2 agree on the direction?
6. Save results + four diagnostic plots.

Run
---
    python -m experiments.concept_localization.run
    python -m experiments.concept_localization.run --n_per_template 300 --top_k 20
    python -m experiments.concept_localization.run --templates T0 T1 --n_digits 2
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

from data.concept_datasets.carry_dataset import generate_carry_pairs
from experiments.concept_localization.analyze import (
    compute_sharpness,
    compute_template_consistency,
    project_onto_features,
)
from experiments.concept_localization.extract_deltas_generic import extract_layer_deltas_generic
from experiments.concept_localization.visualize import (
    plot_feature_projections,
    plot_norm_and_alignment,
    plot_template_consistency,
)
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("concept_localization")

_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"
_N_PER_TEMPLATE = 200
_N_DIGITS = 3
_TOP_K = 15
_OUT_DIR = "runs/concept_localization/carry"


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--n_per_template",
        type=int,
        default=_N_PER_TEMPLATE,
        help="Pairs per template (more = sharper delta estimate)",
    )
    parser.add_argument(
        "--n_digits", type=int, default=_N_DIGITS, help="Number of digits in each operand"
    )
    parser.add_argument(
        "--top_k", type=int, default=_TOP_K, help="Top-k features to report per layer"
    )
    parser.add_argument(
        "--templates", nargs="+", default=["T0", "T1", "T2"], choices=["T0", "T1", "T2"]
    )
    parser.add_argument("--out_dir", default=_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_feature_projection",
        action="store_true",
        help="Skip transcoder projection (faster; no W_enc load)",
    )
    args = parser.parse_args()

    device = get_default_device()
    dtype = parse_dtype(args.dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = args.templates

    # ------------------------------------------------------------------ #
    # 1. Dataset
    # ------------------------------------------------------------------ #
    log.info(
        "Generating carry pairs: %d per template × %d templates, %d-digit operands",
        args.n_per_template,
        len(templates),
        args.n_digits,
    )
    pairs = generate_carry_pairs(args.n_per_template, templates, args.n_digits, seed=args.seed)
    log.info("Generated %d pairs total", len(pairs))

    template_counts = {}
    for p in pairs:
        template_counts[str(p.template)] = template_counts.get(str(p.template), 0) + 1
    for k, v in template_counts.items():
        log.info("  %s: %d pairs", k, v)

    # ------------------------------------------------------------------ #
    # 2. Model
    # ------------------------------------------------------------------ #
    log.info("Loading model %s", args.model)
    # lazy_encoder=True: W_enc loaded on first access per transcoder (saves ~3 GB upfront)
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

    # ------------------------------------------------------------------ #
    # 3. Delta extraction
    # ------------------------------------------------------------------ #
    log.info("Extracting per-layer residual-stream deltas…")
    layer_results = extract_layer_deltas_generic(
        model, pairs, layers, device, dtype, per_template=True
    )

    # ------------------------------------------------------------------ #
    # 4. Sharpness
    # ------------------------------------------------------------------ #
    sharpness = compute_sharpness(layer_results["all"])
    log.info(
        "Peak layer: %d  sharpness_index: %.3f",
        sharpness.peak_layer,
        sharpness.sharpness_index,
    )

    consistency = compute_template_consistency(layer_results)

    # ------------------------------------------------------------------ #
    # 5. Feature projection
    # ------------------------------------------------------------------ #
    projections: dict = {}
    if not args.skip_feature_projection:
        log.info("Projecting delta onto transcoder features…")
        projections = project_onto_features(model, layer_results["all"], top_k=args.top_k)

    # ------------------------------------------------------------------ #
    # 6. Save
    # ------------------------------------------------------------------ #
    results_json = {
        "config": {
            "model": args.model,
            "templates": [str(t) for t in templates],
            "n_per_template": args.n_per_template,
            "n_digits": args.n_digits,
            "top_k": args.top_k,
            "n_pairs_total": len(pairs),
        },
        "sharpness": {
            "peak_layer": sharpness.peak_layer,
            "sharpness_index": round(sharpness.sharpness_index, 4),
            "norm_by_layer": {
                str(l): round(n, 4) for l, n in zip(sharpness.layers, sharpness.norms, strict=False)
            },
            "inter_layer_cos": {
                f"{sharpness.layers[i]}-{sharpness.layers[i + 1]}": round(v, 4)
                for i, v in enumerate(sharpness.inter_layer_cos)
            },
        },
        "template_consistency": {str(layer): row for layer, row in consistency.items()},
        "top_features_by_layer": {
            str(layer): [
                {"feature_id": m.feature_id, "projection": round(m.projection, 4)} for m in matches
            ]
            for layer, matches in projections.items()
        },
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    log.info("Saved results → %s", results_path)

    deltas_path = out_dir / "deltas.pt"
    torch.save(
        {key: {l: v for l, v in ld.delta.items()} for key, ld in layer_results.items()},
        deltas_path,
    )
    log.info("Saved delta tensors → %s", deltas_path)

    # ------------------------------------------------------------------ #
    # 7. Plots
    # ------------------------------------------------------------------ #
    plot_norm_and_alignment(layer_results, out_dir / "norm_trajectory.png")
    log.info("Saved norm_trajectory.png")

    plot_template_consistency(consistency, out_dir / "template_consistency.png")
    log.info("Saved template_consistency.png")

    if projections:
        plot_feature_projections(
            projections, out_dir / "feature_projections_scatter.png", top_k=args.top_k
        )
        log.info("Saved feature projection plots")

    log.info("Done. All outputs in %s", out_dir)


if __name__ == "__main__":
    main()
