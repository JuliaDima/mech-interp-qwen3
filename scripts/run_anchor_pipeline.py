"""Full per-anchor pipeline: run_concept → null permutation → transcoder sweep.

Runs three stages sequentially for one (concept, anchor_position) pair.
Intended to be submitted as a single GPU SLURM job after
select_and_submit_anchors.py has determined the anchor rank and position.

Stage 1  run_concept   — delta extraction, causal analysis, feature projection.
Stage 2  null          — within-class permutation test, reusing deltas.pt.
Stage 3  sweep         — Jaccard×|score| transcoder feature ranking at peak layers.

Output directory layout
-----------------------
runs/concept_localization/{concept}/anchor_rank{R}_pos{P}/
    results.json
    deltas.pt
    feature_projections_scatter.png
    causal_scores.png / causal_overlay.png
    null/
        null_permutation.json
        null_permutation.png
    sweep/
        sweep_ranked.json
        sweep_activations.npz
        sweep_examples.pkl

Usage
-----
    python scripts/run_anchor_pipeline.py \\
        --concept carry --anchor_pos 7 --anchor_rank 1

    python scripts/run_anchor_pipeline.py \\
        --concept gcd --anchor_pos 12 --anchor_rank 2 --n 150 --null_k 30
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
    args = parser.parse_args()

    out_dir = (
        _REPO_ROOT
        / "runs"
        / "concept_localization"
        / args.concept
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
    _run([
        sys.executable,
        str(_REPO_ROOT / "scripts" / "sweeps" / "run_concept_sweep.py"),
        "--concept", args.concept,
        "--layers", layers_str,
        "--anchor", str(args.anchor_pos),
        "--top_k", str(args.sweep_top_k),
        "--out_dir", str(sweep_dir),
    ])

    log.info(
        "All stages complete. Results in %s",
        out_dir,
    )


if __name__ == "__main__":
    main()
