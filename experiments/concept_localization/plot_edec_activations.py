"""Plot mean feature activations for features from edec_features.json.

Reads the pre-saved edec_features.json (which must contain embedded mean/std
activation stats — produced when analyze.py is run with --concept, or when
run_concept.py runs with features enabled). No model loading required.

Usage
-----
    python -m experiments.concept_localization.plot_edec_activations \\
        --anchor_dir runs/concept_localization/carry/carry_T0/anchor_rank1_pos5

    # plot both directions
    python -m experiments.concept_localization.plot_edec_activations \\
        --anchor_dir runs/concept_localization/carry/carry_T0/anchor_rank1_pos5 \\
        --direction pos neg

Produces:
    <anchor_dir>/edec_activations_<direction>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.concept_localization.visualize import plot_edec_mean_activations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--anchor_dir", required=True)
    parser.add_argument("--concept", default="", help="Concept name (used in plot title only)")
    parser.add_argument("--direction", nargs="+", default=["pos"], choices=["pos", "neg"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchor_dir = Path(args.anchor_dir)

    edec_path = anchor_dir / "edec_features.json"
    if not edec_path.exists():
        raise FileNotFoundError(
            f"edec_features.json not found at {edec_path}. "
            "Run analyze.py --concept <name> or run_concept.py first."
        )
    edec_data = json.loads(edec_path.read_text())

    concept = args.concept or anchor_dir.name
    for direction in args.direction:
        out_path = anchor_dir / f"edec_activations_{direction}.png"
        plot_edec_mean_activations(
            edec_data=edec_data,
            out_path=out_path,
            direction=direction,
            concept=concept,
        )
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
