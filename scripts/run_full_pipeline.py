"""Full concept pipeline: make_gif → emergence_per_anchor → top-k anchors → per-anchor pipeline → cross-layer sim.

Dependency order
----------------
Phase A  (model, once per concept)
  1. make_gif           → emergence.npy, emergence.gif
  2. emergence_per_anchor → emergence_per_anchor.pdf

Phase B  (model, per anchor in top-k)
  For each anchor:
    3. run_concept       → results.json, deltas.pt, causal plots, cross_layer_sim.png
    4. null permutation  → null/
    5. sweep             → sweep/sweep_ranked.json etc.
    6. cluster analysis  → sweep/cluster_analysis_T0/
    7. peak-feature plot → sweep/top_features_peak_layers.png
    8. PySR (optional)   → sweep/cluster_analysis_T0/pysr_*.{csv,json,png}

Phase C  (no model)
    9. plot_localisation → cross_layer_sim.pdf, template_consistency.pdf

Usage
-----
    python scripts/run_full_pipeline.py --concept carry --k 6
    python scripts/run_full_pipeline.py --concept carry --k 6 --pysr
    python scripts/run_full_pipeline.py --concept gcd --k 4 --skip_gif
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

from experiments.concept_localization.plot_anchor_analysis import (
    load_emergence,
    top_k_anchors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_full_pipeline")

_N_LAYERS = 36
_BASE = _REPO_ROOT / "runs" / "concept_localization"


def _run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def _run_nofail(cmd: list[str], label: str) -> bool:
    log.info("$ %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
        return True
    except subprocess.CalledProcessError as e:
        log.warning("%s failed (non-fatal): %s", label, e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--concept", required=True,
                        help="Concept name (must be registered in run_concept.py)")
    parser.add_argument("--k", type=int, default=6,
                        help="Number of top anchors to process")

    # make_gif options
    parser.add_argument("--skip_gif", action="store_true",
                        help="Skip make_gif if emergence.npy already exists")
    parser.add_argument("--gif_n", type=int, default=50,
                        help="Pairs per template for make_gif")

    # run_concept / null / sweep options
    parser.add_argument("--n", type=int, default=100,
                        help="Pairs per template for run_concept and null")
    parser.add_argument("--top_k", type=int, default=15,
                        help="Top-k features for directional projection")
    parser.add_argument("--sweep_top_k", type=int, default=200,
                        help="Top-k features per layer for transcoder sweep")
    parser.add_argument("--causal_pairs", type=int, default=50)
    parser.add_argument("--null_k", type=int, default=20)
    parser.add_argument("--cluster_top_k", type=int, default=100)
    parser.add_argument("--n_clusters", type=int, default=6)

    # PySR
    parser.add_argument("--pysr", action="store_true",
                        help="Run PySR on top-3 features per cluster (carry-only; slow)")
    parser.add_argument("--pysr_niterations", type=int, default=40)

    # Model / dtype
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--transcoder_set", default="mwhanna/qwen3-4b-transcoders")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    concept_dir = _BASE / args.concept
    concept_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase A ───────────────────────────────────────────────────────────────

    emergence_path = concept_dir / "emergence.npy"
    if args.skip_gif and emergence_path.exists():
        log.info("Skipping make_gif (--skip_gif, emergence.npy exists)")
    else:
        log.info("=== Phase A-1: make_gif ===")
        _run([
            sys.executable, "-m",
            "experiments.concept_localization.concept_emergence_gif.make_gif",
            "--concept", args.concept,
            "--n", str(args.gif_n),
            "--model", args.model,
            "--transcoder_set", args.transcoder_set,
            "--dtype", args.dtype,
            "--seed", str(args.seed),
        ])

    log.info("=== Phase A-2: emergence_per_anchor ===")
    _run([
        sys.executable, "-m",
        "experiments.concept_localization.plot_emergence_per_anchor",
        "--concept", args.concept,
    ])

    # ── Select top-k anchors ──────────────────────────────────────────────────

    em = load_emergence(args.concept)
    if em is None:
        log.error("emergence.npy not found for '%s' — cannot continue.", args.concept)
        sys.exit(1)

    anchors = top_k_anchors(em, args.concept, k=args.k)
    if not anchors:
        log.error("No active anchors found for '%s'.", args.concept)
        sys.exit(1)

    log.info("Selected %d anchors for '%s':", len(anchors), args.concept)
    for rank, (idx, _, label) in enumerate(anchors, start=1):
        log.info("  Rank %d: pos=%d token=%r", rank, idx, label)

    # ── Phase B: per-anchor pipeline ──────────────────────────────────────────

    for rank, (anchor_pos, _, label) in enumerate(anchors, start=1):
        log.info("=" * 60)
        log.info("Phase B — anchor rank=%d pos=%d token=%r", rank, anchor_pos, label)
        log.info("=" * 60)

        anchor_args = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run_anchor_pipeline.py"),
            "--concept", args.concept,
            "--anchor_pos", str(anchor_pos),
            "--anchor_rank", str(rank),
            "--n", str(args.n),
            "--top_k", str(args.top_k),
            "--sweep_top_k", str(args.sweep_top_k),
            "--causal_pairs", str(args.causal_pairs),
            "--null_k", str(args.null_k),
            "--cluster_top_k", str(args.cluster_top_k),
            "--n_clusters", str(args.n_clusters),
            "--pysr_niterations", str(args.pysr_niterations),
        ]
        if args.pysr:
            anchor_args.append("--pysr")

        _run(anchor_args)

    # ── Phase C: cross-concept plots ──────────────────────────────────────────

    log.info("=== Phase C: plot_localisation ===")
    _run_nofail([
        sys.executable, "-m",
        "experiments.concept_localization.plot_localisation",
        "--concept", args.concept,
    ], "plot_localisation")

    log.info("Full pipeline complete for concept '%s'. Outputs in %s", args.concept, concept_dir)


if __name__ == "__main__":
    main()
