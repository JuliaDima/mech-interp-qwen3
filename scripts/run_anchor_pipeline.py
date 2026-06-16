"""Full per-anchor pipeline: run_concept → null → residual cache.

Stages
------
Stage 1  run_concept        — delta extraction, causal analysis, cross_layer_sim, feature projection.
Stage 2  null permutation   — within-class permutation test, reusing deltas.pt.
Stage 3  residual cache     — raw residual streams for the dataset at all layers.

Output directory layout
-----------------------
runs/concept_localization/{concept}/anchor_rank{R}_pos{P}/
    results.json, deltas.pt, anchor_layer_summary_T0.png
    null/null_permutation.{json,png}
    sweep/
        sweep_residuals.npz, sweep_dataset_examples.pkl

Usage
-----
    python scripts/run_anchor_pipeline.py \
        --concept carry --anchor_pos 7 --anchor_rank 1

"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_anchor_pipeline")

_N_LAYERS = 36


def _run(cmd: list[str]) -> None:
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--concept", required=True)
    parser.add_argument("--anchor_pos", type=int, required=True,
                        help="0-indexed token position used as anchor")
    parser.add_argument("--anchor_rank", type=int, required=True,
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
    parser.add_argument("--feature_score_modes", nargs="+", default=["dec+enc", "dec"],
                        choices=["dec", "enc", "dec+enc"],
                        help="Edec score modes for delta_feature_projections (each gets its own subdir)")
    args = parser.parse_args()

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
        "Concept %s · anchor rank=%d pos=%d · out_dir=%s",
        args.concept, args.anchor_rank, args.anchor_pos, out_dir,
    )

    # ── Stage 1: run_concept ──────────────────────────────────────────────
    log.info("=== Stage 1: run_concept ===")
    _run([
        sys.executable, "-m", "experiments.concept_localization.run_concept",
        "--concept", args.concept,
        "--anchor_modes", str(args.anchor_pos),
        "--template", args.template,
        "--n", str(args.n),
        "--causal",
        "--causal_pairs", str(args.causal_pairs),
        "--top_k", str(args.top_k),
        "--skip_features",
        "--out_dir", str(out_dir),
    ])

    # ── Stage 1b: edec activation bar plot (quick, no activity filter) ───
    log.info("=== Stage 1b: edec activation plot ===")
    try:
        _run([
            sys.executable, "-m", "experiments.concept_localization.plot_edec_activations",
            "--anchor_dir", str(out_dir),
            "--concept", args.concept,
            "--direction", "pos", "neg",
        ])
    except subprocess.CalledProcessError as e:
        log.warning("edec activation plot failed (non-fatal): %s", e)

    # ── Stage 1c: activity-filtered delta projections + grid plots ────────
    log.info("=== Stage 1c: delta_feature_projections ===")
    try:
        _run([
            sys.executable, "-m",
            "experiments.concept_localization.delta_feature_projections",
            "--anchor_dir", str(out_dir),
            "--concept", args.concept,
            "--top_k", str(args.top_k),
            "--score_mode", *args.feature_score_modes,
        ])
    except subprocess.CalledProcessError as e:
        log.warning("delta_feature_projections failed (non-fatal): %s", e)

    # ── Stage 2: null permutation (reuse deltas.pt from stage 1) ─────────
    log.info("=== Stage 2: null permutation ===")
    deltas_path = out_dir / "deltas.pt"
    null_dir = out_dir / "null"
    null_dir.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, "-m", "experiments.concept_localization.run_null_permutation",
        "--concept", args.concept,
        "--anchor_mode", str(args.anchor_pos),
        "--k", str(args.null_k),
        "--real_deltas", str(deltas_path),
        "--out_dir", str(null_dir),
        "--template", args.template,
    ])

    # ── Stage 2b: combined per-anchor layer summary ───────────────────────
    log.info("=== Stage 2b: combined layer summary plot ===")
    _run([
        sys.executable, "-m", "experiments.concept_localization.plot_anchor_layer_summary",
        "--anchor_dir", str(out_dir),
        "--template", args.template,
        "--out", str(out_dir / f"anchor_layer_summary_{args.template}.png"),
    ])

    # ── Stage 3: residual cache at all layers ─────────────────────────────
    log.info("=== Stage 3: all-layer residual cache ===")
    layers_str = ",".join(str(l) for l in range(_N_LAYERS))
    log.info("Caching residuals for all layers: %s", layers_str)

    sweep_dir = out_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "sweeps" / "run_concept_sweep.py"),
        "--concept", args.concept,
        "--layers", layers_str,
        "--anchor", str(args.anchor_pos),
        "--out_dir", str(sweep_dir),
    ]
    # Per-anchor sweeps must use one template because anchor positions are
    # template-specific. Multi-template comparison plots live at the concept root.
    sweep_cmd += ["--template", args.template]
    _run(sweep_cmd)

    log.info("All stages complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
