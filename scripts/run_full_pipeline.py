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

Phase C
    Per-anchor layer summaries are produced during Phase B.

Usage
-----
    python scripts/run_full_pipeline.py --concept carry --k 6
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

from experiments.concept_localization.plot_emergence_per_anchor import load_concept_anchor_data

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
    parser.add_argument("--template", default="T0",
                        help="Single template for per-anchor analyses")

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
            "--template", args.template,
        ])

    log.info("=== Phase A-2: emergence_per_anchor ===")
    _run([
        sys.executable, "-m",
        "experiments.concept_localization.plot_emergence_per_anchor",
        "--concept", args.concept,
    ])

    # ── Select top-k anchors by mean_cos (direction stability) ───────────────
    # Mirrors select_and_submit_anchors.py: rank active positions by mean pairwise
    # cosine similarity of delta vectors across layers. This puts stable-direction
    # operand positions (e.g. carry digit tokens) first, which abruptness misses.

    data = load_concept_anchor_data(args.concept)
    if data is None:
        log.error("emergence.npy not found for '%s' — cannot continue.", args.concept)
        sys.exit(1)

    mean_cos = data.get("mean_cos", {})
    if not mean_cos:
        log.error("mean_cos missing from emergence.npy for '%s' — re-run make_gif.", args.concept)
        sys.exit(1)

    active  = data["active"]
    labels  = data.get("labels", [])
    ranked  = sorted(active, key=lambda i: mean_cos.get(i, 0.0), reverse=True)
    selected = ranked[: args.k]

    anchors = [
        (pos, labels[pos] if pos < len(labels) else str(pos))
        for pos in selected
    ]

    log.info("Selected %d anchors for '%s' (by mean_cos):", len(anchors), args.concept)
    for rank, (idx, label) in enumerate(anchors, start=1):
        log.info("  Rank %d: pos=%d token=%r  mean_cos=%.3f", rank, idx, label, mean_cos.get(idx, 0.0))

    # ── Phase B: per-anchor pipeline ──────────────────────────────────────────

    for rank, (anchor_pos, label) in enumerate(anchors, start=1):
        log.info("=" * 60)
        log.info("Phase B — anchor rank=%d pos=%d token=%r", rank, anchor_pos, label)
        log.info("=" * 60)

        anchor_args = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run_anchor_pipeline.py"),
            "--concept", args.concept,
            "--anchor_pos", str(anchor_pos),
            "--anchor_rank", str(rank),
            "--template", args.template,
            "--n", str(args.n),
            "--top_k", str(args.top_k),
            "--sweep_top_k", str(args.sweep_top_k),
            "--causal_pairs", str(args.causal_pairs),
            "--null_k", str(args.null_k),
            "--cluster_top_k", str(args.cluster_top_k),
            "--n_clusters", str(args.n_clusters),
        ]

        _run(anchor_args)

    # ── Phase C: cross-concept plots ──────────────────────────────────────────

    # Per-anchor layer diagnostics are saved by run_anchor_pipeline.py as a single
    # combined anchor_layer_summary_<template>.png.  Do not regenerate individual
    # cross_layer_sim.png files here.

    log.info("Full pipeline complete for concept '%s'. Outputs in %s", args.concept, concept_dir)


if __name__ == "__main__":
    main()
