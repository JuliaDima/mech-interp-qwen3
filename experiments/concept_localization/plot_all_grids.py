"""Generate combined grid figures for every plot type across all concepts.

For causal_overlay the grid is re-generated from data (preserving vector
quality).  All other plot types are assembled from existing PNGs via
image tiling.

Outputs written to runs/concept_localization/:
    causal_overlay_all.png
    causal_scores_all.png
    norm_trajectory_all.png
    feature_projections_scatter_all.png

Usage
-----
    python -m experiments.concept_localization.plot_all_grids
    python -m experiments.concept_localization.plot_all_grids --ncols 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.causal_analysis import CausalScores
from experiments.concept_localization.visualize import (
    assemble_png_grid,
    plot_causal_overlay_grid,
)

_DEFAULT_RUNS = _REPO_ROOT / "runs" / "concept_localization"

# Plot types to tile from existing PNGs (filename stem → human title)
_PNG_PLOT_TYPES: list[tuple[str, str]] = [
    ("norm_trajectory", "Norm trajectory"),
    ("causal_scores", "Causal scores"),
    ("feature_projections_scatter", "Feature projections"),
]


def _load_causal_scores(data: dict) -> CausalScores:
    layers = sorted(int(k) for k in data["patching_mean"])
    return CausalScores(
        layers=layers,
        patching_mean={int(k): v for k, v in data["patching_mean"].items()},
        patching_std={int(k): v for k, v in data["patching_std"].items()},
        grad_dot_delta_mean={int(k): v for k, v in data["grad_dot_delta_mean"].items()},
        grad_dot_delta_std={int(k): v for k, v in data["grad_dot_delta_std"].items()},
        n_pairs=data["n_pairs"],
    )


def _load_overlay_entry(concept_dir: Path) -> tuple[str, dict, dict[int, float]] | None:
    results_path = concept_dir / "results.json"
    deltas_path = concept_dir / "deltas.pt"
    if not results_path.exists() or not deltas_path.exists():
        return None
    results = json.loads(results_path.read_text())
    if not results.get("causal"):
        return None
    causal_results = {k: _load_causal_scores(v) for k, v in results["causal"].items()}
    agg_deltas = torch.load(deltas_path, weights_only=False).get("all", {})
    delta_norms = {l: t.norm().item() for l, t in agg_deltas.items()}
    return concept_dir.name, causal_results, delta_norms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs_dir", type=Path, default=_DEFAULT_RUNS)
    ap.add_argument("--ncols", type=int, default=3)
    args = ap.parse_args()

    concept_dirs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir())

    # --- causal_overlay: re-generated from data ---
    overlay_entries = [e for d in concept_dirs if (e := _load_overlay_entry(d)) is not None]
    if overlay_entries:
        out = args.runs_dir / "causal_overlay_all.png"
        plot_causal_overlay_grid(overlay_entries, out, ncols=args.ncols)
        print(f"causal_overlay_all.png  ({len(overlay_entries)} concepts)")
    else:
        print("causal_overlay: no data found, skipping")

    # --- all other plot types: tile existing PNGs ---
    for stem, title in _PNG_PLOT_TYPES:
        png_entries: list[tuple[str, Path]] = []
        for d in concept_dirs:
            p = d / f"{stem}.png"
            if p.exists():
                png_entries.append((d.name, p))
        if not png_entries:
            print(f"{stem}: no PNGs found, skipping")
            continue
        out = args.runs_dir / f"{stem}_all.png"
        assemble_png_grid(png_entries, out, ncols=args.ncols, title=title)
        print(f"{stem}_all.png  ({len(png_entries)} concepts)")


if __name__ == "__main__":
    main()
