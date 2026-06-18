"""Coordinator: select candidate anchors from emergence.npy, submit per-anchor jobs.

Submits --candidates pipeline jobs (default 2×--k) ranked by non-monotonicity (NM)
score — the only signal available before the pipeline runs.  After all jobs
complete, discover_anchors() re-selects the best --k by combined score
(null-excess + patching + grad), so the final plot shows the truly best anchors.

Intended to run as a lightweight job after make_gif completes, with
  sbatch --dependency=afterok:{gif_jid} scripts/sbatch_run.sh python scripts/select_and_submit_anchors.py ...

Calling sbatch from within a SLURM job is supported on CSD3.

Usage
-----
    python scripts/select_and_submit_anchors.py --concept carry
    python scripts/select_and_submit_anchors.py --concept carry --k 3 --dry_run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.plots.plot_emergence_per_anchor import (
    load_concept_anchor_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("select_and_submit_anchors")

_SBATCH_RUN = str(_REPO_ROOT / "scripts" / "sbatch_run.sh")
_PIPELINE_SCRIPT = str(_REPO_ROOT / "experiments" / "concept_localization" / "pipeline" / "run_anchor_pipeline.py")


def _submit(cmd: list[str], dry_run: bool) -> str:
    """Run sbatch --parsable and return the job ID string."""
    if dry_run:
        log.info("[DRY] %s", " ".join(cmd))
        return "0"
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--concept", required=True,
                        help="Concept name (must have emergence.npy)")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of top anchors to display (combined-score re-rank after pipeline)")
    parser.add_argument("--candidates", type=int, default=6,
                        help="Candidate anchors to run pipeline on (ranked by mean_cos)")
    parser.add_argument("--anchor_time", default="01:00:00",
                        help="SLURM --time for each anchor pipeline job")
    parser.add_argument("--template", default="T0",
                        help="Single template for per-anchor jobs")
    parser.add_argument("--n", type=int, default=100,
                        help="Pairs per template for run_concept and null")
    parser.add_argument("--top_k", type=int, default=15,
                        help="Top-k features for directional projection")
    parser.add_argument("--causal_pairs", type=int, default=50)
    parser.add_argument("--null_k", type=int, default=20)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print sbatch commands without submitting")
    args = parser.parse_args()

    n_candidates = args.candidates

    data = load_concept_anchor_data(args.concept)
    if data is None:
        log.error(
            "emergence.npy not found for concept '%s'. "
            "Run make_gif first.",
            args.concept,
        )
        sys.exit(1)

    non_mono = data["non_mono"]
    mean_cos = data.get("mean_cos", {})
    labels   = data.get("labels", [])
    active   = data["active"]

    if not mean_cos:
        log.error(
            "mean_cos not found in emergence.npy for concept '%s'. "
            "Re-run make_gif to regenerate emergence.npy with mean_cos.",
            args.concept,
        )
        sys.exit(1)

    candidates = sorted(active, key=lambda i: mean_cos.get(i, 0.0), reverse=True)[:n_candidates]

    if not candidates:
        log.error("No active anchors found for '%s'.", args.concept)
        sys.exit(1)

    log.info(
        "Concept '%s': submitting %d candidate anchor jobs ranked by mean_cos "
        "(will display top %d by combined score)",
        args.concept, len(candidates), args.k,
    )
    for rank, pos_idx in enumerate(candidates, start=1):
        label = labels[pos_idx] if pos_idx < len(labels) else str(pos_idx)
        log.info("  Candidate %d: pos=%d token=%r  mean_cos=%.3f", rank, pos_idx, label, mean_cos.get(pos_idx, 0.0))

    submitted: list[tuple[int, int, str, str]] = []

    for rank, anchor_idx in enumerate(candidates, start=1):
        label = labels[anchor_idx] if anchor_idx < len(labels) else str(anchor_idx)
        job_name = f"anchor_{args.concept}_r{rank}"
        cmd = [
            "sbatch", "--parsable",
            f"--job-name={job_name}",
            f"--time={args.anchor_time}",
            _SBATCH_RUN,
            sys.executable, _PIPELINE_SCRIPT,
            "--concept", args.concept,
            "--anchor_pos", str(anchor_idx),
            "--anchor_rank", str(rank),
            "--template", args.template,
            "--n", str(args.n),
            "--top_k", str(args.top_k),
            "--causal_pairs", str(args.causal_pairs),
            "--null_k", str(args.null_k),
        ]
        jid = _submit(cmd, args.dry_run)
        submitted.append((rank, anchor_idx, label, jid))
        log.info(
            "  Submitted rank=%d pos=%d '%s' → job %s",
            rank, anchor_idx, label, jid,
        )

    print(f"\n{'Concept':<30} {'Rank':<6} {'Pos':<6} {'Token':<20} {'JobID'}")
    print("-" * 75)
    for rank, pos, label, jid in submitted:
        print(f"{args.concept:<30} {rank:<6} {pos:<6} {label!r:<20} {jid}")


if __name__ == "__main__":
    main()
