"""
1D feature activation scatter: top-K E_dec-projected features vs a scalar
input variable from the sweep dataset.

Complements the 2D Fourier grid plots (plot_fourier_selection.py) for
concepts where the relevant input is a single scalar rather than a pair of
modular operands.

Usage
-----
    python experiments/concept_localization/plots/plot_edec_1d.py \
        --anchor_dir runs/concept_localization/gcd/gcd_T0/anchor_rank1_pos4 \
        --meta_key a_pos \
        --top_k 6

    python experiments/concept_localization/plots/plot_edec_1d.py \
        --anchor_dir runs/concept_localization/residue_class/residue_class_T0/anchor_rank1_pos3 \
        --meta_key a_pos \
        --top_k 6 \
        --out runs/concept_localization/residue_class/edec_1d_pos3.pdf
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors import safe_open

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import experiments.plot_style as ps

TC_DIR = pathlib.Path(
    "/rds/user/eid23/hpc-work/p28/cache/hf/hub/"
    "models--mwhanna--qwen3-4b-transcoders/snapshots/"
    "94d176260ac39ce2f882b8b09aba8c118df29bb3"
)


# ── Transcoder helpers ──────────────────────────────────────────────────────

def _encode_layer(layer: int, H_l: np.ndarray) -> np.ndarray:
    """Load W_enc+b_enc for one layer and apply ReLU. CPU-friendly."""
    import torch
    with safe_open(str(TC_DIR / f"layer_{layer}.safetensors"),
                   framework="pt", device="cpu") as f:
        W_enc = f.get_tensor("W_enc").float()
        b_enc = f.get_tensor("b_enc").float()
    H_t = torch.from_numpy(H_l).float()
    pre = H_t @ W_enc.T + b_enc
    return torch.relu(pre).numpy()


# ── Data loading ────────────────────────────────────────────────────────────

def load_sweep(anchor_dir: pathlib.Path):
    """Return (examples, pos_mask, npz) from the anchor sweep directory."""
    sweep = anchor_dir / "sweep"
    meta  = json.loads((sweep / "sweep_residuals.meta.json").read_text())
    examples = meta["payload"]["examples"]
    npz      = np.load(str(sweep / "sweep_residuals.npz"))
    pos_mask = npz["pos_mask"].astype(bool)          # shape (2N,)
    return examples, pos_mask, npz


def load_topk(anchor_dir: pathlib.Path, k: int) -> list[dict]:
    """Load top-k features by |cos_sim| from anchor-level edec_features.json."""
    edec = json.loads((anchor_dir / "edec_features.json").read_text())
    rows = []
    for layer_s, feats in edec.items():
        if isinstance(feats, list):
            for f in feats:
                rows.append({
                    "layer":      int(layer_s),
                    "feature_id": int(f["feature_id"]),
                    "cos_sim":    float(f.get("cos_sim", 0)),
                })
    rows.sort(key=lambda r: -r["cos_sim"])
    return rows[:k]


# ── Plotting ────────────────────────────────────────────────────────────────

def _smooth(x: np.ndarray, y: np.ndarray, n_bins: int = 20):
    """Bin-mean smoothing for a regression guide line."""
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return np.array([x_min]), np.array([y.mean()])
    edges  = np.linspace(x_min, x_max, n_bins + 1)
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() >= 2:
            cx.append(x[mask].mean())
            cy.append(y[mask].mean())
    return np.array(cx), np.array(cy)


def plot_edec_1d(
    anchor_dir: pathlib.Path,
    meta_key: str,
    top_k: int = 6,
    out_path: pathlib.Path | None = None,
    show_neg: bool = True,
) -> pathlib.Path:
    ps.apply()
    plt.rcParams.update({"font.family": "serif"})

    feat_rows = load_topk(anchor_dir, top_k)
    examples, pos_mask, npz = load_sweep(anchor_dir)

    # x values from metadata; derive neg key by replacing _pos → _neg if present
    neg_meta_key = meta_key.replace("_pos", "_neg")
    x_pos = np.array([ex["meta"].get(meta_key,     np.nan) for ex in examples],
                     dtype=float)
    x_neg = np.array([ex["meta"].get(neg_meta_key,
                                     ex["meta"].get(meta_key, np.nan))
                      for ex in examples], dtype=float)

    # Group features by layer to load transcoder once per layer
    by_layer: dict[int, list[int]] = {}
    for r in feat_rows:
        by_layer.setdefault(r["layer"], []).append(r["feature_id"])

    # Compute activations per layer
    acts: dict[tuple[int, int], np.ndarray] = {}
    for layer, feat_ids in sorted(by_layer.items()):
        h_key = f"H_L{layer}"
        if h_key not in npz:
            continue
        H_l  = npz[h_key].astype(np.float32)
        enc  = _encode_layer(layer, H_l)          # (2N, d_tc)
        for fid in feat_ids:
            acts[(layer, fid)] = enc[:, fid]       # (2N,)

    n_cols = min(3, top_k)
    n_rows = (top_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 2.8),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})
    axes_flat = np.array(axes).reshape(-1)

    for idx, r in enumerate(feat_rows):
        ax  = axes_flat[idx]
        key = (r["layer"], r["feature_id"])
        if key not in acts:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=ps.GRAY)
            ax.set_visible(False)
            continue

        a = acts[key]
        a_pos_vals  = a[pos_mask]
        a_neg_vals  = a[~pos_mask]

        ax.scatter(x_pos, a_pos_vals, s=14, alpha=0.65,
                   color=ps.NAVY, zorder=3, label="pos")
        if show_neg:
            ax.scatter(x_neg, a_neg_vals, s=14, alpha=0.40,
                       color=ps.GRAY, zorder=2, label="neg")

        # bin-mean guide
        cx, cy = _smooth(x_pos, a_pos_vals)
        if len(cx) > 1:
            ax.plot(cx, cy, color=ps.TEAL, lw=1.6, zorder=4)

        ax.set_xlabel(meta_key, fontsize=8)
        ax.set_ylabel("activation", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

        cs_str = f"{r['cos_sim']:+.3f}"
        ax.set_title(f"L{r['layer']:02d}  F{r['feature_id']}\n"
                     f"$E_{{\\mathrm{{dec}}}}$ cos = {cs_str}",
                     fontsize=8, pad=4)

    # hide spare panels
    for spare_ax in axes_flat[len(feat_rows):]:
        spare_ax.set_visible(False)

    # legend on first panel
    handles = [
        matplotlib.lines.Line2D([], [], marker="o", color="w",
                                markerfacecolor=ps.NAVY, markersize=5,
                                label="positive"),
    ]
    if show_neg:
        handles.append(matplotlib.lines.Line2D([], [], marker="o", color="w",
                                               markerfacecolor=ps.GRAY,
                                               markersize=5, label="negative"))
    handles.append(matplotlib.lines.Line2D([], [], color=ps.TEAL, lw=1.6,
                                           label="pos bin mean"))
    axes_flat[0].legend(handles=handles, fontsize=7, frameon=False,
                        loc="upper left")

    if out_path is None:
        concept_dir = anchor_dir.parent.parent
        anchor_name = anchor_dir.name
        out_path = concept_dir / f"edec_1d_{anchor_name}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_dir", type=pathlib.Path, required=True)
    parser.add_argument("--meta_key",   default="a_pos",
                        help="Key in example['meta'] to use as x-axis")
    parser.add_argument("--top_k",      type=int, default=6)
    parser.add_argument("--out",        type=pathlib.Path, default=None)
    parser.add_argument("--no_neg",     action="store_true",
                        help="Hide negative-example points")
    args = parser.parse_args()

    plot_edec_1d(
        anchor_dir=args.anchor_dir,
        meta_key=args.meta_key,
        top_k=args.top_k,
        out_path=args.out,
        show_neg=not args.no_neg,
    )


if __name__ == "__main__":
    main()
