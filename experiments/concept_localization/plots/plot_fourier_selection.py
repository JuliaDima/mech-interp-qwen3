"""
8-panel figure showing selected Fourier-structured transcoder features found
at the ones_b anchor (carry_T0, anchor_rank2_pos9).

Each panel shows the normalised 10x10 activation grid f(a mod 10, b mod 10)
and its top-K Fourier approximation side-by-side, labelled with dominant mode.

Saved to runs/concept_localization/carry/fourier_selection.pdf.
"""
import pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
from safetensors import safe_open

from experiments.concept_localization.concept_fits.fourier_feature_analysis import (
    _aggregate_grids, _score_grids_batch, find_min_k, load_examples_ab,
)
import experiments.plot_style as ps
from scripts.model_config import transcoder_snapshot_dir

TC_DIR: pathlib.Path | None = None


def _tc_dir() -> pathlib.Path:
    global TC_DIR
    if TC_DIR is None:
        TC_DIR = transcoder_snapshot_dir()
    return TC_DIR


def _encode_layer(layer: int, H_l: np.ndarray) -> np.ndarray:
    """Load only W_enc+b_enc for one layer and apply ReLU (no threshold in safetensors)."""
    with safe_open(str(_tc_dir() / f"layer_{layer}.safetensors"), framework="pt", device="cpu") as f:
        W_enc = f.get_tensor("W_enc").float()
        b_enc = f.get_tensor("b_enc").float()
    H_t  = torch.from_numpy(H_l).float()
    pre  = H_t @ W_enc.T + b_enc
    return torch.relu(pre).numpy()

ANCHOR = REPO / "runs/concept_localization/carry/carry_T0/anchor_rank2_pos9/sweep"
OUT    = REPO / "runs/concept_localization/carry/fourier_selection.pdf"

FEATURES = [
    ("L12", "F1468",   12, 1468),
    ("L13", "F75746",  13, 75746),
    ("L21", "F30390",  21, 30390),
    ("L13", "F57984",  13, 57984),
    ("L14", "F114215", 14, 114215),
    ("L13", "F107956", 13, 107956),
    ("L13", "F56616",  13, 56616),
    ("L13", "F132660", 13, 132660),
]

CMAP = LinearSegmentedColormap.from_list("wv", ["#f8f8f8", ps.NAVY], N=256)


def load_grids_for_features(features, anchor_dir: pathlib.Path):
    """Load normalised 10x10 grids for the specified (layer, feat_id) pairs."""
    npz = np.load(str(anchor_dir / "sweep_residuals.npz"))
    a_vals, b_vals = load_examples_ab(anchor_dir / "sweep_dataset_examples.pkl")
    a_mod = (a_vals % 10).astype(np.int64)
    b_mod = (b_vals % 10).astype(np.int64)

    by_layer: dict[int, list[int]] = {}
    for _, _, layer, feat_id in features:
        by_layer.setdefault(layer, []).append(feat_id)

    grids: dict[tuple[int, int], np.ndarray] = {}
    for layer, feat_ids in sorted(by_layer.items()):
        h_key = f"H_L{layer}"
        if h_key not in npz:
            raise KeyError(f"Layer {layer} not in sweep_residuals.npz")
        H_l  = npz[h_key].astype(np.float32)
        acts = _encode_layer(layer, H_l)           # (N, d_tc)

        for feat_id in feat_ids:
            col      = acts[:, feat_id]
            grid_raw = _aggregate_grids(col[:, None], a_mod, b_mod)[0]
            lo, hi   = np.nanmin(grid_raw), np.nanmax(grid_raw)
            span     = hi - lo if hi - lo > 1e-12 else 1.0
            grid     = np.nan_to_num((grid_raw - lo) / span, nan=0.0)
            grids[(layer, feat_id)] = grid

    return grids


def dominant_type_label(X: np.ndarray) -> str:
    from experiments.concept_localization.concept_fits.fourier_feature_analysis import (
        _score_grids_batch,
    )
    scores = _score_grids_batch(X[None], subtract_mean=True)
    cats = {k: float(scores[k][0]) for k in
            ("diff", "sum", "parity", "row", "col", "mixed", "row_parity", "col_parity")}
    top  = max(cats, key=cats.__getitem__)
    frac = cats[top]
    label_map = {
        "diff":      r"iso-diff ($b-a$)",
        "sum":       r"iso-sum ($a+b$)",
        "parity":    r"parity $(-1)^{a+b}$",
        "row":       r"$a$-only",
        "col":       r"$b$-only",
        "mixed":     "mixed",
        "row_parity": r"row parity",
        "col_parity": r"col parity",
    }
    return f"{label_map.get(top, top)} ({frac:.0%})"


def clean_ax(ax):
    N = 10
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.tick_params(which="both", length=0, labelsize=6)
    ax.grid(which="minor", color="#dddddd", linewidth=0.3)
    ax.grid(which="major", visible=False)
    ax.set_axisbelow(False)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")


def main():
    print("Loading grids …")
    grids = load_grids_for_features(FEATURES, ANCHOR)

    n_feats = len(FEATURES)
    ncols   = 3
    nrows   = (n_feats + ncols - 1) // ncols   # 3 rows of 3 (last row has 2)
    # Each feature: 2 subplots (original + K-approx) side-by-side
    fig, axes = plt.subplots(nrows, ncols * 2,
                             figsize=(ncols * 2 * 2.2, nrows * 2.6),
                             gridspec_kw={"wspace": 0.10, "hspace": 0.90})
    plt.rcParams.update({"font.family": "serif"})

    for idx, (l_str, f_str, layer, feat_id) in enumerate(FEATURES):
        row = idx // ncols
        col = idx % ncols
        ax_orig  = axes[row, col * 2]
        ax_approx = axes[row, col * 2 + 1]

        X = grids[(layer, feat_id)]
        k_used, r2, modes, mu, C, Xhat = find_min_k(X, r2_target=0.90, k_max=6)
        Xhat_c = np.clip(Xhat, 0.0, 1.0)

        dom = dominant_type_label(X)

        for ax, data, label in [(ax_orig, X, "orig"), (ax_approx, Xhat_c, f"$K$={k_used}")]:
            ax.imshow(data.T, origin="lower", aspect="equal",
                      cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
            clean_ax(ax)
            ax.set_xlabel("$a$", fontsize=6, labelpad=1)

        ax_orig.set_ylabel("$b$", fontsize=7, labelpad=1)
        ax_approx.set_ylabel("")

        feat_label = f"L{layer}  F{feat_id}"
        ax_orig.set_title(feat_label, fontsize=9, fontweight="bold", pad=3)
        ax_approx.set_title(
            f"{dom}\n$K$={k_used},  $R^2$={r2:.2f}",
            fontsize=7, color="#333", pad=3, linespacing=1.4,
        )

    # Hide unused panels in last row
    for spare in range(n_feats, nrows * ncols):
        row, col = spare // ncols, spare % ncols
        axes[row, col * 2].set_visible(False)
        axes[row, col * 2 + 1].set_visible(False)

    fig.suptitle(
        "Selected Fourier-structured features — carry ones$_b$ anchor (layers 12–21)",
        fontsize=10, y=1.01,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=200)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
