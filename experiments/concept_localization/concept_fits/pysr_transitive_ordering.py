"""PySR analysis for transitive-ordering concept features.

Adds derived variables that directly encode the relational structure:

    b_minus_c  = b − c   (positive for True/pos, negative for False/neg)
    c_over_b   = c / b   (< 1 for True/pos, > 1 for False/neg)

Key diagnostic: if PySR picks b_minus_c the feature encodes the relational
concept; if it picks c alone it's a magnitude artifact (c_pos is
systematically smaller than c_neg across the dataset).

Usage
-----
    python experiments/concept_localization/concept_fits/pysr_transitive_ordering.py \\
        --anchor_dir runs/concept_localization/transitive_ordering/transitive_ordering_T0/anchor_rank1_pos16

    python experiments/concept_localization/concept_fits/pysr_transitive_ordering.py \\
        --all_anchors --template T0
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import experiments.plot_style as ps
from experiments.concept_localization.concept_fits.base import (
    make_parser, resolve_anchor_dirs, run_anchor,
)

_CONCEPT      = "transitive_ordering"
_ANCHORS_ROOT = _REPO_ROOT / "runs" / "concept_localization" / _CONCEPT / f"{_CONCEPT}_T0"
_VAR_NAMES    = ["a", "b", "c", "b_minus_c", "c_over_b"]


def _build_Xy(npz, examples, key, orig_indices):
    if key not in npz:
        return None, None
    acts   = np.asarray(npz[key], dtype=np.float64)
    rows_X, rows_y = [], []
    for orig_i, ex in zip(orig_indices, examples):
        meta = ex["meta"]
        for use_pos in (True, False):
            act_i = 2 * orig_i if use_pos else 2 * orig_i + 1
            if act_i >= len(acts):
                continue
            suffix    = "_pos" if use_pos else "_neg"
            a, b      = float(meta["a"]), float(meta["b"])
            c         = float(meta[f"c{suffix}"])
            rows_X.append([a, b, c, b - c, c / b if b != 0 else 0.0])
            rows_y.append(float(acts[act_i]))
    if not rows_X:
        return None, None
    X = np.array(rows_X, dtype=np.float64)
    y = np.array(rows_y, dtype=np.float64)
    lo, hi = y.min(), y.max()
    if hi - lo > 1e-12:
        y = (y - lo) / (hi - lo)
    return X, y


def _plot(X, y, model, key, out_path, r2_threshold=0.0):
    """Scatter of activation vs b−c, coloured by pos/neg."""
    ps.apply()
    y_pred  = model.predict(X)
    best_eq = str(model.get_best()["equation"])
    ss_res  = float(np.sum((y - y_pred) ** 2))
    ss_tot  = float(np.sum((y - y.mean()) ** 2))
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    if np.isnan(r2) or r2 < r2_threshold:
        print(f"    [skip plot] {key}: R²={r2:.3f} < threshold {r2_threshold}")
        return r2 if not np.isnan(r2) else 0.0

    bmc    = X[:, _VAR_NAMES.index("b_minus_c")]
    is_pos = np.array([(i % 2 == 0) for i in range(len(y))])
    idx    = np.argsort(bmc)
    xv, yv, yp, isp = bmc[idx], y[idx], y_pred[idx], is_pos[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.scatter(xv[isp],  yv[isp],  color=ps.NAVY, s=8, alpha=0.7, label="pos (True)",  zorder=3)
    ax.scatter(xv[~isp], yv[~isp], color=ps.RED,  s=8, alpha=0.7, label="neg (False)", zorder=3)
    ax.axvline(0, color=ps.GRAY, ls="--", lw=0.8, label="b−c = 0")
    ax.set_xlabel("b − c"); ax.set_ylabel("activation (normalised)")
    ax.set_title(f"{key}  actual"); ax.legend(fontsize=7)

    ax = axes[1]
    ax.scatter(yv[isp],  yp[isp],  color=ps.NAVY, s=8, alpha=0.7, label="pos")
    ax.scatter(yv[~isp], yp[~isp], color=ps.RED,  s=8, alpha=0.7, label="neg")
    lims = [min(yv.min(), yp.min()) - 0.02, max(yv.max(), yp.max()) + 0.02]
    ax.plot(lims, lims, color=ps.GRAY, lw=0.8, ls="--")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("actual"); ax.set_ylabel("predicted")
    ax.set_title(f"R²={r2:.3f}  eq: {best_eq[:60]}"); ax.legend(fontsize=7)

    fig.suptitle(f"Transitive ordering — {key}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved → {out_path}")
    return r2


def _plot_topk_overview(anchor_dir: Path, template: str, top_k: int) -> None:
    """Grid of activation vs b-c scatter for all top-k features (ungated by R²)."""
    import json as _json
    import pickle as _pickle
    edec_dir  = anchor_dir / "sweep" / "edec_pysr"
    acts_path = edec_dir / "sweep_activations.npz"
    ex_path   = edec_dir / "sweep_examples.pkl"
    if not acts_path.exists() or not ex_path.exists():
        return

    npz      = np.load(acts_path)
    examples = _pickle.load(open(ex_path, "rb"))
    if template:
        orig_indices = [i for i, ex in enumerate(examples) if ex.get("template") == template]
        examples     = [examples[i] for i in orig_indices]
    else:
        orig_indices = list(range(len(examples)))

    keys = [k for k in npz.files if k.startswith("L")]
    if not keys:
        return

    def _delta(k):
        a = np.asarray(npz[k])
        pos = [a[2 * i]     for i in orig_indices if 2 * i     < len(a)]
        neg = [a[2 * i + 1] for i in orig_indices if 2 * i + 1 < len(a)]
        return abs(np.mean(pos) - np.mean(neg)) if pos and neg else 0.0

    keys_ranked = sorted(keys, key=_delta, reverse=True)[:top_k]
    ncols = 5
    nrows = math.ceil(len(keys_ranked) / ncols)
    ps.apply()
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.8), squeeze=False)

    for idx, key in enumerate(keys_ranked):
        ax   = axes[idx // ncols][idx % ncols]
        acts = np.asarray(npz[key])
        rows_bmc, rows_act, rows_pos = [], [], []
        for orig_i, ex in zip(orig_indices, examples):
            meta = ex["meta"]
            b, c_pos, c_neg = float(meta["b"]), float(meta["c_pos"]), float(meta["c_neg"])
            for use_pos in (True, False):
                act_i = 2 * orig_i if use_pos else 2 * orig_i + 1
                if act_i >= len(acts):
                    continue
                c = c_pos if use_pos else c_neg
                rows_bmc.append(b - c)
                rows_act.append(float(acts[act_i]))
                rows_pos.append(use_pos)
        if not rows_bmc:
            ax.set_visible(False)
            continue
        bmc = np.array(rows_bmc)
        act = np.array(rows_act)
        isp = np.array(rows_pos)
        ax.scatter(bmc[isp],  act[isp],  color=ps.NAVY, s=5, alpha=0.6)
        ax.scatter(bmc[~isp], act[~isp], color=ps.RED,  s=5, alpha=0.6)
        ax.axvline(0, color=ps.GRAY, ls="--", lw=0.6)
        ax.set_title(key, fontsize=8)
        ax.set_xlabel("b−c", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(len(keys_ranked), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Top-{len(keys_ranked)} E_dec features  —  {anchor_dir.name}  (template={template or 'all'})",
                 fontsize=10)
    fig.tight_layout()
    out_path = edec_dir / f"edec_topk_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overview grid → {out_path}")


def main():
    args        = make_parser(_CONCEPT, _ANCHORS_ROOT).parse_args()
    anchor_dirs = resolve_anchor_dirs(args, _ANCHORS_ROOT)
    all_results = []
    for anchor_dir in anchor_dirs:
        print(f"\n=== {anchor_dir.name} ===")
        _plot_topk_overview(anchor_dir, args.template, args.top_k)
        all_results.extend(run_anchor(
            anchor_dir, _CONCEPT, _VAR_NAMES, _build_Xy, _plot,
            top_k=args.top_k, niterations=args.niterations,
            r2_threshold=args.r2_threshold, seed=args.seed,
            template=args.template, n_pairs=args.n_pairs, dtype=args.dtype,
            features=args.features,
        ))
    if all_results:
        best = sorted(all_results, key=lambda r: r["r2"] or 0, reverse=True)
        print("\nTop fits by R²:")
        for r in best[:10]:
            print(f"  {r['anchor']} {r['feature']}  R²={r['r2']}  eq={r['best_equation']}")


if __name__ == "__main__":
    main()
