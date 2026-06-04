"""Feature projection scatter for concept localisation (E_dec).

Usage
-----
    python -m experiments.concept_localization.plot_feature_projections \\
        --run_dir runs/concept_localization/carry/anchor_rank5_pos9

Produces:
    <run_dir>/feature_projections_scatter.png
"""

import argparse
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments.concept_localization.analyze import project_onto_E_dec
from experiments.plot_style import apply

_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run_dir", required=True, help="Path to anchor run directory")
    parser.add_argument(
        "--transcoder_set", default=_TRANSCODER_SET, help="HF repo id for transcoders"
    )
    parser.add_argument("--top_k", type=int, default=15, help="Top-k features per layer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply()

    run_dir = Path(args.run_dir)
    cache = Path.home() / ".cache" / "mechinterp_qwen3" / args.transcoder_set

    raw = torch.load(str(run_dir / "deltas.pt"), map_location="cpu")
    projections = project_onto_E_dec(raw["all"], cache, top_k=args.top_k)

    all_layers, all_fids, all_cos = [], [], []
    for layer, matches in projections.items():
        for m in matches:
            all_layers.append(layer)
            all_fids.append(m.feature_id)
            all_cos.append(m.cos_sim)

    all_layers = np.array(all_layers)
    all_fids = np.array(all_fids)
    all_cos = np.array(all_cos)

    # ── Plot ──────────────────────────────────────────────────────────────────
    vmax = np.abs(all_cos).max()
    norm_c = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sizes = 22 + 85 * (np.abs(all_cos) / vmax) ** 1.5
    n_layers = int(all_layers.max()) + 1

    fig, ax = plt.subplots(figsize=(13, 6))
    sc = ax.scatter(
        all_layers,
        all_fids,
        c=all_cos,
        s=sizes,
        cmap="RdBu_r",
        norm=norm_c,
        alpha=0.78,
        linewidths=0.2,
        edgecolors="white",
        zorder=3,
    )
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Transcoder feature ID", fontsize=11)
    ax.set_title(
        rf"Top-{args.top_k} transcoder features aligned with concept delta $\delta_l$ (E_dec)"
        "\n"
        r"colour $= \cos(\delta_l,\,\mathbf{e}^{\mathrm{dec}}_f)$;"
        r"  dot size $\propto |\cos(\delta_l,\,\mathbf{e}^{\mathrm{dec}}_f)|$",
        fontsize=11,
        pad=10,
    )
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cb = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.022, pad=0.01)
    cb.set_label(r"$\cos(\delta_l,\,\mathbf{e}^{\mathrm{dec}}_f)$", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    fig.tight_layout()
    out_path = run_dir / "feature_projections_scatter.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
