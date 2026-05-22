"""Assemble per-concept causal overlay plots into one combined grid figure.

Scans runs/concept_localization/*/results.json, reconstructs CausalScores
from JSON and delta norms from deltas.pt, then saves one combined PNG.

Usage
-----
    python -m experiments.concept_localization.plot_causal_overlay_grid
    python -m experiments.concept_localization.plot_causal_overlay_grid --ncols 2
    python -m experiments.concept_localization.plot_causal_overlay_grid --runs_dir /path/to/runs
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
from experiments.concept_localization.visualize import plot_causal_overlay_grid

_DEFAULT_RUNS = _REPO_ROOT / "runs" / "concept_localization"


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


def _load_entry(concept_dir: Path) -> tuple[str, dict, dict[int, float], dict[int, float]] | None:
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

    mean_act_norms = {int(k): v for k, v in results.get("mean_act_norm", {}).items()}

    return concept_dir.name, causal_results, delta_norms, mean_act_norms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs_dir", type=Path, default=_DEFAULT_RUNS)
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    concept_dirs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir())
    entries = [e for d in concept_dirs if (e := _load_entry(d)) is not None]

    if not entries:
        print(f"No concept dirs with causal results found under {args.runs_dir}")
        return

    out_path = args.out or args.runs_dir / "causal_overlay_all.png"
    plot_causal_overlay_grid(entries, out_path, ncols=args.ncols)
    print(f"Saved {len(entries)}-concept grid to {out_path}")


if __name__ == "__main__":
    main()
