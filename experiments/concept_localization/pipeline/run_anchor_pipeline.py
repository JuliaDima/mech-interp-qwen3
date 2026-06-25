"""Full per-anchor pipeline: run_concept → null → residual cache.

Stages
------
Stage 1  run_concept        — delta extraction, causal analysis, cross_layer_sim, feature projection.
Stage 2  null permutation   — within-class permutation test, reusing deltas.pt.
Stage 3  residual cache     — raw residual streams for the dataset at all layers.
Stage 4  delta projections  — activity-filtered feature projection onto transcoder directions.

Output directory layout
-----------------------
runs/concept_localization/{concept}/anchor_rank{R}_pos{P}/
    results.json, deltas.pt, anchor_layer_summary_T0.png
    null/null_permutation.{json,png}
    sweep/
        sweep_residuals.npz, sweep_dataset_examples.pkl
        delta_feature_projections_{mode}/

Usage
-----
    python -m experiments.concept_localization.pipeline.run_anchor_pipeline \
        --concept carry --anchor_pos 7 --anchor_rank 1

"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_config import add_model_config_arg, default_model, default_transcoder_set, resolve_model_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_anchor_pipeline")

_N_LAYERS = 36
_MODEL = default_model()
_TRANSCODER_SET = default_transcoder_set()


def _subprocess(cmd: list[str]) -> None:
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_model(model_name: str, transcoder_set: str, dtype_str: str = "bfloat16", compile: bool = False):
    import torch
    from mechinterp_qwen3.attribution_model import AttributionModel
    from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
    from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
    dtype = parse_dtype(dtype_str)
    device = get_default_device()
    tc_set, _ = load_transcoder_from_hub(
        transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        model_name, tc_set, dtype=dtype, device=device
    )
    model.eval()
    if compile:
        log.info("Compiling model with torch.compile...")
        model.model = torch.compile(model.model)
        log.info("torch.compile done")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--concept", required=True)
    parser.add_argument("--anchor_pos", default="delimiter",
                        help="0-indexed token position used as anchor, or 'delimiter'")
    parser.add_argument("--anchor_rank", type=int, default=1,
                        help="Rank of this anchor (1 = best) from emergence.npy")
    parser.add_argument("--template", default="T0",
                        help="Single template for all per-anchor analyses")
    parser.add_argument("--n", type=int, default=100,
                        help="Pairs per template for run_concept and null")
    parser.add_argument("--top_k", type=int, default=15,
                        help="Top-k features for directional projection in run_concept")
    parser.add_argument("--causal_pairs", type=int, default=50,
                        help="Max pairs for causal patching analysis")
    parser.add_argument("--null_k", type=int, default=20,
                        help="Number of null permutations")
    parser.add_argument("--feature_score_modes", nargs="+", default=["enc+dec", "dec"],
                        choices=["dec", "enc", "enc+dec"],
                        help="Edec score modes for delta_feature_projections (each gets its own subdir)")
    add_model_config_arg(parser)
    parser.add_argument("--model", default=None)
    parser.add_argument("--transcoder_set", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model for faster inference (first call is slow)")
    args = parser.parse_args()
    resolve_model_args(args)

    out_dir = (
        _REPO_ROOT
        / "runs"
        / "concept_localization"
        / args.concept
        / f"{args.concept}_{args.template}"
        / f"anchor_rank{args.anchor_rank}_pos{args.anchor_pos}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Concept %s · anchor rank=%d pos=%s · out_dir=%s",
        args.concept, args.anchor_rank, args.anchor_pos, out_dir,
    )

    # Load model once — reused across all stages that need it.
    log.info("=== Loading model (once) ===")
    model = _load_model(args.model, args.transcoder_set, args.dtype, compile=args.compile)

    # ── Stage 1: run_concept ──────────────────────────────────────────────
    log.info("=== Stage 1: run_concept ===")
    from experiments.concept_localization.pipeline.run_concept import _run_single
    concept_args = Namespace(
        concept=args.concept,
        model=args.model,
        transcoder_set=args.transcoder_set,
        dtype=args.dtype,
        n=args.n,
        top_k=args.top_k,
        out_dir=str(out_dir),
        seed=42,
        anchor_modes=str(args.anchor_pos),
        skip_features=False,
        feature_score_mode="enc+dec",
        template=args.template,
        causal=True,
        causal_pairs=args.causal_pairs,
    )
    _run_single(concept_args, model=model)

    # ── Stage 1b: edec activation bar plot (quick, no model needed) ───────
    log.info("=== Stage 1b: edec activation plot ===")
    try:
        _subprocess([
            sys.executable, "-m", "experiments.concept_localization.plots.plot_edec_activations",
            "--anchor_dir", str(out_dir),
            "--concept", args.concept,
            "--direction", "pos", "neg",
        ])
    except subprocess.CalledProcessError as e:
        log.warning("edec activation plot failed (non-fatal): %s", e)

    # ── Stage 2: null permutation (reuse deltas.pt from stage 1) ─────────
    log.info("=== Stage 2: null permutation ===")
    from experiments.concept_localization.pipeline.run_null_permutation import run_null_permutation
    null_dir = out_dir / "null"
    null_dir.mkdir(parents=True, exist_ok=True)
    run_null_permutation(
        concept=args.concept,
        model_name=args.model,
        transcoder_set=args.transcoder_set,
        n_per_template=args.n,
        k=args.null_k,
        out_dir=null_dir,
        dtype_str=args.dtype,
        seed=42,
        real_deltas_path=out_dir / "deltas.pt",
        anchor_mode=str(args.anchor_pos),
        template=args.template,
        model=model,
    )

    # ── Stage 2b: combined per-anchor layer summary (no model needed) ─────
    log.info("=== Stage 2b: combined layer summary plot ===")
    _subprocess([
        sys.executable, "-m", "experiments.concept_localization.plots.plot_anchor_layer_summary",
        "--anchor_dir", str(out_dir),
        "--template", args.template,
        "--out", str(out_dir / f"anchor_layer_summary_{args.template}.png"),
    ])

    # ── Stage 3: residual cache at all layers ─────────────────────────────
    log.info("=== Stage 3: all-layer residual cache ===")
    from experiments.concept_localization.pipeline.run_concept_sweep import (
        _load_concept, cache_sweep_residuals, _sweep_cache_metadata,
    )
    import json, pickle
    import numpy as np

    sweep_dir = out_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    target_layers = list(range(_N_LAYERS))

    pairs = _load_concept(args.concept, args.n, 42)
    import random
    random.Random(42).shuffle(pairs)
    pairs = [p for p in pairs if p.template == args.template]
    log.info("Sweep pairs for template %s: %d", args.template, len(pairs))

    residuals, pos_mask, prompts, sweep_examples = cache_sweep_residuals(
        model, pairs, target_layers, anchor_mode=str(args.anchor_pos), max_pairs=200,
    )

    residuals_path = sweep_dir / "sweep_residuals.npz"
    np.savez_compressed(
        residuals_path,
        pos_mask=pos_mask,
        prompts=np.array(prompts, dtype=object),
        layers=np.array(sorted(residuals), dtype=np.int32),
        **{f"H_L{layer}": H for layer, H in residuals.items()},
    )
    log.info("Saved residual cache → %s", residuals_path)

    examples_path = sweep_dir / "sweep_dataset_examples.pkl"
    with open(examples_path, "wb") as f:
        pickle.dump(sweep_examples, f)

    # Build sweep metadata using a minimal args namespace for _sweep_cache_metadata
    sweep_meta_args = Namespace(
        concept=args.concept,
        model=args.model,
        transcoder_set=args.transcoder_set,
        dtype=args.dtype,
        layers=target_layers,
        anchor=str(args.anchor_pos),
        n=args.n,
        template=args.template,
        max_pairs=200,
        seed=42,
    )
    metadata = _sweep_cache_metadata(sweep_meta_args, target_layers, prompts, sweep_examples)
    (sweep_dir / "sweep_residuals.meta.json").write_text(json.dumps(metadata, indent=2))
    log.info("Saved sweep metadata → %s", sweep_dir / "sweep_residuals.meta.json")

    # ── Stage 4: activity-filtered delta projections (requires sweep) ─────
    log.info("=== Stage 4: delta_feature_projections ===")
    from experiments.concept_localization.pipeline.delta_feature_projections import run_for_anchor
    dfp_args = Namespace(
        concept=args.concept,
        score_mode=args.feature_score_modes,
        top_k=args.top_k,
        seed=42,
        dtype=args.dtype,
        rank_by="score",
        display_n_pairs=None,
        no_attr_filter=True,
        attr_min_survival=0.05,
        attr_survival_file=None,
        graphs_dir=None,
        template=args.template,
        model=args.model,
        transcoder_set=args.transcoder_set,
    )
    run_for_anchor(out_dir, model, dfp_args)

    log.info("All stages complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
