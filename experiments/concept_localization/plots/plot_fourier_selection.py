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

CMAP = LinearSegmentedColormap.from_list("wv", ["#ffffff", ps.VIOLET], N=256)


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
        "diff":       r"Iso-difference ($b - a$)",
        "sum":        r"Iso-sum ($a + b$)",
        "parity":     r"Global parity $(-1)^{a+b}$",
        "row":        r"Row-periodic ($a$)",
        "col":        r"Column-periodic ($b$)",
        "mixed":      "Mixed",
        "row_parity": "Row parity",
        "col_parity": "Column parity",
    }
    return f"{label_map.get(top, top)} ({frac:.0%})"


def clean_ax(ax):
    N = 10
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.tick_params(which="both", length=0, labelsize=7)
    ax.grid(which="minor", color="#ebebeb", linewidth=0.35)
    ax.grid(which="major", visible=False)
    ax.set_axisbelow(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    print("Loading grids …")
    grids = load_grids_for_features(FEATURES, ANCHOR)

    ps.apply()

    n_feats     = len(FEATURES)
    NCOLS_PAIRS = 2   # 2 feature-Fourier pairs per row
    nrows       = (n_feats + NCOLS_PAIRS - 1) // NCOLS_PAIRS  # 4 rows

    fig, axes = plt.subplots(
        nrows, NCOLS_PAIRS * 2,
        figsize=(NCOLS_PAIRS * 2 * 2.6, nrows * 3.0),
        gridspec_kw={"wspace": 0.12, "hspace": 0.55},
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for idx, (l_str, f_str, layer, feat_id) in enumerate(FEATURES):
        row      = idx // NCOLS_PAIRS
        col_pair = idx % NCOLS_PAIRS
        ax_orig   = axes[row, col_pair * 2]
        ax_approx = axes[row, col_pair * 2 + 1]

        X = grids[(layer, feat_id)]
        k_used, r2, modes, mu, C, Xhat = find_min_k(X, r2_target=0.90, k_max=6)
        Xhat_c = np.clip(Xhat, 0.0, 1.0)

        dom = dominant_type_label(X)

        for ax, data in [(ax_orig, X), (ax_approx, Xhat_c)]:
            ax.imshow(data.T, origin="lower", aspect="equal",
                      cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
            clean_ax(ax)
            ax.set_xlabel("$a$ mod 10", fontsize=8, labelpad=2)

        ax_orig.set_ylabel("$b$ mod 10", fontsize=8, labelpad=2)
        ax_approx.set_ylabel("")

        ax_orig.set_title(
            rf"Activation matrix of $L^{{{layer}}}_{{{feat_id}}}$",
            fontsize=8, pad=5,
        )
        ax_approx.set_title(
            rf"$F_{{\mathrm{{Fourier}}}}(a,b)$,  $K={k_used}$",
            fontsize=9, fontweight="normal", color="#333", pad=5,
        )

    # Hide unused panels
    for spare in range(n_feats, nrows * NCOLS_PAIRS):
        row, col_pair = spare // NCOLS_PAIRS, spare % NCOLS_PAIRS
        axes[row, col_pair * 2].set_visible(False)
        axes[row, col_pair * 2 + 1].set_visible(False)

    # Single shared colorbar
    import matplotlib
    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=CMAP
        ),
        ax=axes.ravel().tolist(),
        fraction=0.018, pad=0.02, shrink=0.6,
    )
    cb.set_label("normalised activation", fontsize=9)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=8, length=0)
    cb.outline.set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=200)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
