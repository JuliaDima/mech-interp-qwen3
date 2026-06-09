"""PySR on top-k E_dec features for the carry concept.

Uses the 10×10 units-digit grid mode (a mod 10 × b mod 10) with
PySR + Fourier combined plots.  Auto-generates edec_pysr/sweep_activations.npz
if it is missing — no separate extraction step needed.

Usage
-----
    python experiments/concept_localization/concept_fits/pysr_carry.py \
        --anchor_dir runs/concept_localization/carry/carry_T0/anchor_rank5_pos9

    python experiments/concept_localization/concept_fits/pysr_carry.py \
        --all_anchors --template T0
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SWEEPS_DIR = _REPO_ROOT / "scripts" / "sweeps"
for p in (_REPO_ROOT, _REPO_ROOT / "src", _SWEEPS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fit_pysr_sweep import fit_feature, _meta_mode
from experiments.concept_localization.concept_fits.base import (
    _ensure_edec_data, make_parser, resolve_anchor_dirs,
)
from experiments.concept_localization.concept_fits.pysr_modular import _plot_topk_grid

_CONCEPT      = "carry"
_ANCHORS_ROOT = _REPO_ROOT / "runs" / "concept_localization" / _CONCEPT / f"{_CONCEPT}_T0"


def _filter_by_template(npz, examples: list[dict], template: str):
    """Return (npz_dict, examples) filtered to template, with re-indexed arrays.

    The activations npz has interleaved rows: 2*i = pos, 2*i+1 = neg for
    pair i in the original examples list.  After filtering to a template,
    we re-slice so that enumerate(filtered_examples) aligns with the new arrays.
    """
    if not template:
        return {k: np.asarray(npz[k]) for k in npz.files}, examples

    orig_indices = [i for i, ex in enumerate(examples)
                    if ex.get("template") == template]
    filtered = [examples[i] for i in orig_indices]
    if not filtered:
        return None, []

    row_indices = np.array([j for i in orig_indices for j in (2 * i, 2 * i + 1)])
    npz_dict = {}
    for k in npz.files:
        arr = np.asarray(npz[k])
        valid = row_indices[row_indices < len(arr)]
        npz_dict[k] = arr[valid]
    return npz_dict, filtered


def run_anchor_carry(
    anchor_dir: Path,
    template: str,
    top_k: int,
    n_pairs: int,
    niterations: int,
    r2_threshold: float,
    seed: int,
    dtype: str,
    plot_only: bool = False,
) -> list[dict]:
    ok = _ensure_edec_data(anchor_dir, _CONCEPT,
                           n_pairs=n_pairs, top_k=top_k, seed=seed, dtype_str=dtype)
    if not ok:
        return []

    edec_dir = anchor_dir / "sweep" / "edec_pysr"
    raw_npz  = np.load(edec_dir / "sweep_activations.npz")
    with open(edec_dir / "sweep_examples.pkl", "rb") as f:
        all_examples = pickle.load(f)

    npz_dict, examples = _filter_by_template(raw_npz, all_examples, template)
    if not examples:
        print(f"  [skip] {anchor_dir.name}: no examples for template={template!r}")
        return []

    mode = _meta_mode(examples)
    if mode == "skip":
        print(f"  [skip] {anchor_dir.name}: no numeric meta — skipping")
        return []

    keys = sorted(k for k in npz_dict if k.startswith("L"))
    if not keys:
        print(f"  [skip] {anchor_dir.name}: no features in npz")
        return []

    # Load cos_sims if available (saved by _ensure_edec_data)
    features_meta_path = edec_dir / "edec_features.json"
    if features_meta_path.exists():
        meta = json.load(open(features_meta_path))
        features  = [(m["layer"], m["feature_id"]) for m in meta if m["key"] in npz_dict]
        cos_sims  = [m["cos_sim"] for m in meta if m["key"] in npz_dict]
    else:
        features  = []
        cos_sims  = []

    # Rank by |mean_pos - mean_neg| over the filtered set
    def _delta(k: str) -> float:
        a = npz_dict[k]
        return float(abs(np.mean(a[0::2]) - np.mean(a[1::2])))

    keys_ranked = sorted(keys, key=_delta, reverse=True)[:top_k]
    print(f"  {anchor_dir.name}: fitting {len(keys_ranked)} features "
          f"(mode={mode}, template={template or 'all'})")

    out_dir = edec_dir / "carry_pysr"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overview grid of all top-k feature activations (ungated by R²)
    if features and cos_sims:
        _plot_topk_grid(features, cos_sims, npz_dict, examples,
                        out_dir / "edec_topk_grid.pdf")

    if plot_only:
        return []

    results = []
    for key in keys_ranked:
        r = fit_feature(key, npz_dict, examples, mode, out_dir,
                        niterations=niterations, r2_threshold=r2_threshold,
                        seed=seed, with_fourier=True)
        if r:
            results.append(r)

    summary_path = out_dir / "pysr_carry_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Summary → {summary_path}")
    return results


def main():
    ap = make_parser(_CONCEPT, _ANCHORS_ROOT)
    ap.add_argument("--plot_only", action="store_true",
                    help="Generate overview grid only; skip PySR fits")
    args = ap.parse_args()
    anchor_dirs = resolve_anchor_dirs(args, _ANCHORS_ROOT)
    all_results = []
    for anchor_dir in anchor_dirs:
        print(f"\n=== {anchor_dir.name} ===")
        all_results.extend(run_anchor_carry(
            anchor_dir,
            template=args.template,
            top_k=args.top_k,
            n_pairs=args.n_pairs,
            niterations=args.niterations,
            r2_threshold=args.r2_threshold,
            seed=args.seed,
            dtype=args.dtype,
            plot_only=args.plot_only,
        ))

    if all_results:
        best = sorted(all_results, key=lambda r: r.get("r2") or 0, reverse=True)
        print("\nTop fits by R²:")
        for r in best[:10]:
            print(f"  {r['feature']}  R²={r['r2']}  eq={r['best_equation']}")


if __name__ == "__main__":
    main()
