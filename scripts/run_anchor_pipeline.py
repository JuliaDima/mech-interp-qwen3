"""Full per-anchor pipeline: run_concept → null → sweep → cluster analysis → (PySR).

Stages
------
Stage 1  run_concept        — delta extraction, causal analysis, cross_layer_sim, feature projection.
Stage 2  null permutation   — within-class permutation test, reusing deltas.pt.
Stage 3  sweep              — Jaccard×|score| transcoder feature ranking at peak layers.
Stage 4  cluster analysis   — cosine clustering + PCA/Fourier plots per cluster.
Stage 5  PySR (--pysr)      — symbolic regression per cluster + top-k E_dec features (carry=grid, others=generic).

The cross-anchor peak-feature plot (plot_sweep_peak_features.py) is submitted once
by the coordinator after all anchor jobs finish, since it aggregates every anchor.

Output directory layout
-----------------------
runs/concept_localization/{concept}/anchor_rank{R}_pos{P}/
    results.json, deltas.pt, anchor_layer_summary_T0.png
    null/null_permutation.{json,png}
    sweep/
        sweep_ranked.json, sweep_activations.npz, sweep_examples.pkl
        top_features_peak_layers.png
        cluster_analysis_T0/
            cluster_features.json
            cluster_NN_{pca,top3}.png
            cosine_similarity.png
            pysr_*/  (if --pysr)

Usage
-----
    python scripts/run_anchor_pipeline.py \\
        --concept carry --anchor_pos 7 --anchor_rank 1

    python scripts/run_anchor_pipeline.py \\
        --concept carry --anchor_pos 7 --anchor_rank 1 --pysr
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--sweep_top_k", type=int, default=200,
                        help="Top-k features per layer for the transcoder sweep")
    parser.add_argument("--causal_pairs", type=int, default=50,
                        help="Max pairs for causal patching analysis")
    parser.add_argument("--null_k", type=int, default=20,
                        help="Number of null permutations")
    parser.add_argument("--cluster_top_k", type=int, default=100,
                        help="Top-k features fed into cluster analysis")
    parser.add_argument("--n_clusters", type=int, default=6,
                        help="Number of feature clusters")
    parser.add_argument("--pysr", action="store_true",
                        help="Run PySR per cluster + top-k E_dec features (carry=grid, others=generic; slow)")
    parser.add_argument("--pysr_niterations", type=int, default=40,
                        help="PySR iterations per feature")
    parser.add_argument("--edec_top_k", type=int, default=15,
                        help="Top-k E_dec-aligned features for the extra PySR pass")
    parser.add_argument("--r2_threshold", type=float, default=0.5,
                        help="Only plot PySR fits whose R² exceeds this")
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
        "--out_dir", str(out_dir),
    ])

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

    # ── Stage 3: transcoder sweep at peak ±2 layers ───────────────────────
    log.info("=== Stage 3: transcoder sweep ===")
    results = json.loads((out_dir / "results.json").read_text())
    peak_layer = results["sharpness"]["peak_layer"]
    layers = list(range(max(0, peak_layer - 2), min(_N_LAYERS, peak_layer + 3)))
    layers_str = ",".join(str(l) for l in layers)
    log.info("Peak layer=%d → sweeping layers %s", peak_layer, layers_str)

    sweep_dir = out_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "sweeps" / "run_concept_sweep.py"),
        "--concept", args.concept,
        "--layers", layers_str,
        "--anchor", str(args.anchor_pos),
        "--top_k", str(args.sweep_top_k),
        "--out_dir", str(sweep_dir),
    ]
    # Per-anchor sweeps must use one template because anchor positions are
    # template-specific. Multi-template comparison plots live at the concept root.
    sweep_cmd += ["--template", args.template]
    _run(sweep_cmd)

    # ── Stage 4: cluster analysis ─────────────────────────────────────────
    log.info("=== Stage 4: cluster analysis ===")
    _run([
        sys.executable,
        str(_REPO_ROOT / "scripts" / "sweeps" / "analyze_sweep_clusters.py"),
        "--sweep_dir", str(sweep_dir),
        "--top_k", str(args.cluster_top_k),
        "--n_clusters", str(args.n_clusters),
        "--template", args.template,
    ])

    # NOTE: the cross-anchor peak-feature plot (plot_sweep_peak_features.py)
    # globs every anchor_rank*_pos* dir at once, so it is submitted once by the
    # coordinator (select_and_submit_anchors.py) after all anchor jobs finish,
    # rather than redundantly per anchor here.

    # ── Stage 5: PySR per cluster + top-k E_dec features (optional) ──────
    if args.pysr:
        log.info("=== Stage 5: PySR ===")
        cluster_json = sweep_dir / f"cluster_analysis_{args.template}" / "cluster_features.json"
        if not cluster_json.exists():
            log.warning("cluster_features.json not found at %s — skipping PySR", cluster_json)
        else:
            try:
                _run([
                    sys.executable,
                    str(_REPO_ROOT / "scripts" / "sweeps" / "fit_pysr_sweep.py"),
                    "--sweep_dir", str(sweep_dir),
                    "--cluster_features_json", str(cluster_json),
                    "--out_dir", str(cluster_json.parent),
                    "--niterations", str(args.pysr_niterations),
                    "--r2_threshold", str(args.r2_threshold),
                ])
            except subprocess.CalledProcessError as e:
                log.warning("PySR (cluster) stage failed (non-fatal): %s", e)

        # ── Stage 5b: PySR on top-k E_dec-aligned features ───────────────
        log.info("=== Stage 5b: PySR on top-%d E_dec features ===", args.edec_top_k)
        try:
            _run([
                sys.executable,
                str(_REPO_ROOT / "scripts" / "sweeps" / "pysr_top_edec_features.py"),
                "--anchor_dir", str(out_dir),
                "--concept", args.concept,
                "--top_k", str(args.edec_top_k),
                "--niterations", str(args.pysr_niterations),
                "--r2_threshold", str(args.r2_threshold),
                "--template", args.template,
            ])
        except subprocess.CalledProcessError as e:
            log.warning("PySR (E_dec) stage failed (non-fatal): %s", e)

    log.info("All stages complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
