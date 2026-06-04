"""Coordinator: select top-k anchors from emergence.npy, submit per-anchor jobs.

Reads emergence.npy for a concept, ranks anchors by early-weighted abruptness
via top_k_anchors(), then submits one SLURM job per anchor via sbatch.

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
log = logging.getLogger("select_and_submit_anchors")

_SBATCH_RUN = str(_REPO_ROOT / "scripts" / "sbatch_run.sh")
_PIPELINE_SCRIPT = str(_REPO_ROOT / "scripts" / "run_anchor_pipeline.py")


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
    parser.add_argument("--k", type=int, default=4,
                        help="Number of top anchors to select")
    parser.add_argument("--anchor_time", default="01:00:00",
                        help="SLURM --time for each anchor pipeline job")
    parser.add_argument("--n", type=int, default=100,
                        help="Pairs per template for run_concept and null")
    parser.add_argument("--top_k", type=int, default=15,
                        help="Top-k features for directional projection")
    parser.add_argument("--sweep_top_k", type=int, default=200,
                        help="Top-k features per layer for sweep")
    parser.add_argument("--causal_pairs", type=int, default=50)
    parser.add_argument("--null_k", type=int, default=20)
    parser.add_argument("--cluster_top_k", type=int, default=100,
                        help="Top-k features fed into cluster analysis")
    parser.add_argument("--n_clusters", type=int, default=6,
                        help="Number of feature clusters")
    parser.add_argument("--pysr", action="store_true",
                        help="Run PySR per cluster + top-k E_dec features (carry=grid, others=generic; slow)")
    parser.add_argument("--pysr_niterations", type=int, default=40)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print sbatch commands without submitting")
    args = parser.parse_args()

    em = load_emergence(args.concept)
    if em is None:
        log.error(
            "emergence.npy not found for concept '%s'. "
            "Run make_gif first.",
            args.concept,
        )
        sys.exit(1)

    anchors = top_k_anchors(em, args.concept, k=args.k)
    if not anchors:
        log.error("top_k_anchors returned empty list for '%s'.", args.concept)
        sys.exit(1)

    log.info(
        "Concept '%s': selected %d/%d anchors",
        args.concept, len(anchors), args.k,
    )
    for rank, (anchor_idx, _, label) in enumerate(anchors, start=1):
        log.info("  Rank %d: pos=%d token=%r", rank, anchor_idx, label)

    submitted: list[tuple[int, int, str, str]] = []

    for rank, (anchor_idx, _, label) in enumerate(anchors, start=1):
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
            cmd.append("--pysr")
        jid = _submit(cmd, args.dry_run)
        submitted.append((rank, anchor_idx, label, jid))
        log.info(
            "  Submitted rank=%d pos=%d '%s' → job %s",
            rank, anchor_idx, label, jid,
        )

    # ── Final cross-anchor peak-feature plot ────────────────────────────────
    #   plot_sweep_peak_features.py globs every anchor_rank*_pos* dir at once,
    #   so it must run a single time after all anchor pipelines complete (not
    #   per-anchor). Depend on every anchor job via afterany so it runs even if
    #   some anchors fail.
    if submitted:
        dep = ":".join(jid for *_, jid in submitted)
        peak_cmd = [
            "sbatch", "--parsable",
            f"--job-name=peakfeat_{args.concept}",
            "--time=00:20:00",
            f"--dependency=afterany:{dep}",
            _SBATCH_RUN,
            sys.executable,
            str(_REPO_ROOT / "scripts" / "sweeps" / "plot_sweep_peak_features.py"),
            "--concept", args.concept,
        ]
        peak_jid = _submit(peak_cmd, args.dry_run)
        log.info("  Submitted peak-feature plot → job %s (after all anchors)", peak_jid)

    print(f"\n{'Concept':<30} {'Rank':<6} {'Pos':<6} {'Token':<20} {'JobID'}")
    print("-" * 75)
    for rank, pos, label, jid in submitted:
        print(f"{args.concept:<30} {rank:<6} {pos:<6} {label!r:<20} {jid}")


if __name__ == "__main__":
    main()
