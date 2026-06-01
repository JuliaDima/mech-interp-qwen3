"""Rank all transcoder features by carry discrimination score.

Sweeps the full 10×10 ones-digit grid and scores every feature at the
target layers by

    carry_score = mean(act | carry) − mean(act | no_carry)
    jaccard     = |active ∩ mask| / |active ∪ mask|

ranked by jaccard × |carry_score|.  Outputs a JSON ranking, a compressed
NPZ with raw activation arrays, and a heatmap grid plot.

Usage
-----
    python scripts/carry/run_carry_sweep.py --layers 16,17,18,19,20,21
    python scripts/carry/run_carry_sweep.py --layers 4,5,6,19,20,21 --top_k 50 --template T0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
from sweep_utils import apply_transcoder_all, resolve_anchor_from_positions, score_and_rank  # noqa: E402

import experiments.plot_style as ps
from data.concept_datasets.carry_dataset import TEMPLATES as _TEMPLATES
from data.concept_datasets.carry_dataset import make_anchor_positions
from experiments.concept_localization.analyze import collect_layer_residuals
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_carry_sweep")

_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"

_CARRY_MASK = np.array(
    [d_a + d_b >= 10 for d_a in range(10) for d_b in range(10)],
    dtype=bool,
)


def _build_ones_grid_inputs(
    model,
    anchor_mode: str,
    template_str: str,
    higher_a: int,
    higher_b: int,
) -> list[tuple[list[int], int]]:
    """Build (token_ids, anchor_pos) for all 100 cells of the ones-digit grid.

    Grid order is row-major: i = d_a * 10 + d_b for d_a, d_b in {0..9}.
    """
    result = []
    for d_a in range(10):
        for d_b in range(10):
            a = higher_a * 10 + d_a
            b = higher_b * 10 + d_b
            prompt = template_str.format(a=a, b=b)
            ids = model.tokenizer(prompt, add_special_tokens=False).input_ids
            positions = make_anchor_positions(template_str, a, b, model.tokenizer)
            anchor = resolve_anchor_from_positions(positions, anchor_mode, len(ids) - 1)
            result.append((ids, anchor))
    return result


@torch.no_grad()
def sweep_all_features_carry_score(
    model,
    target_layers: list[int],
    anchor_mode: str = "ones_b",
    template_str: str = "calc: {a}+{b}= ",
    higher_a: int = 13,
    higher_b: int = 13,
    top_k: int = 50,
) -> tuple[list[tuple[int, int, float, float]], dict[tuple[int, int], np.ndarray]]:
    """Rank all transcoder features at target_layers by carry discrimination.

    Returns:
        ranked   list of (layer, feat_id, carry_score, jaccard) sorted by
                 jaccard × |carry_score| descending
        heatmaps dict (layer, feat_id) → (10, 10) float32 array for the top_k features
    """
    inputs = _build_ones_grid_inputs(model, anchor_mode, template_str, higher_a, higher_b)
    H = collect_layer_residuals(model, inputs, target_layers)

    all_ranked: list[tuple[int, int, float, float]] = []
    layer_acts: dict[int, np.ndarray] = {}

    for layer in target_layers:
        if layer not in H:
            continue
        try:
            acts_np = apply_transcoder_all(model, layer, H[layer])
        except (IndexError, KeyError, AttributeError):
            log.warning("No transcoder at layer %d — skipping", layer)
            continue

        ranked_layer = score_and_rank(acts_np, _CARRY_MASK, top_k=top_k)
        for feat_id, score, jaccard in ranked_layer:
            all_ranked.append((layer, feat_id, score, jaccard))

        layer_acts[layer] = acts_np
        top_score, top_jac = ranked_layer[0][1], ranked_layer[0][2]
        log.info(
            "Layer %2d  d_tc=%d  top combined=%.4f  top jaccard=%.4f  top cs=%.4f",
            layer,
            acts_np.shape[1],
            abs(top_score) * top_jac,
            top_jac,
            top_score,
        )

    all_ranked.sort(key=lambda x: x[3] * abs(x[2]), reverse=True)
    top = all_ranked[:top_k]

    heatmaps: dict[tuple[int, int], np.ndarray] = {}
    for layer, feat_id, _, _ in top:
        if layer in layer_acts:
            heatmaps[(layer, feat_id)] = layer_acts[layer][:, feat_id].reshape(10, 10)

    return top, heatmaps


def plot_carry_score_heatmaps(
    ranked: list[tuple[int, int, float, float]],
    heatmaps: dict[tuple[int, int], np.ndarray],
    out_path: Path,
    concept: str,
    anchor_label: str,
    xlabel: str = "ones(a)",
    ylabel: str = "ones(b)",
    ncols: int = 5,
) -> None:
    """Grid of 2-D heatmaps for the top features ranked by jaccard × |carry_score|."""
    items = [
        (layer, feat_id, score, jac)
        for layer, feat_id, score, jac in ranked
        if (layer, feat_id) in heatmaps
    ]
    if not items:
        return

    n = len(items)
    nrows = math.ceil(n / ncols)
    cell = 3.2
    ps.apply()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * cell, nrows * cell + 1.2),
        squeeze=False,
    )

    for idx, (layer, feat_id, score, jac) in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        mat = heatmaps[(layer, feat_id)]
        n_x, n_y = mat.shape
        vmax = float(mat.max()) or 1.0
        cmap = "Blues" if score >= 0 else "Reds"

        im = ax.imshow(
            mat.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            extent=[-0.5, n_x - 0.5, -0.5, n_y - 0.5],
        )
        sign = "+" if score >= 0 else ""
        ax.set_title(
            f"L{layer:02d} F{feat_id:06d}\ncs={sign}{score:.2f}  jac={jac:.2f}",
            fontsize=7,
            pad=3,
        )
        ax.set_xlabel(xlabel, fontsize=7, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
        ax.set_xticks(range(n_x))
        ax.set_yticks(range(n_y))
        ax.tick_params(labelsize=5)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"{concept} — carry score sweep  anchor={anchor_label}  "
        f"(x={xlabel}, y={ylabel})\n"
        r"ranked by jaccard $\times$ |carry_score|",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--template",
        default="T0",
        help="Template key (T0/T1/T2) from carry_dataset.TEMPLATES",
    )
    parser.add_argument(
        "--anchor_mode",
        default="ones_b",
        help="Anchor for the ones-digit grid (ones_a / ones_b / plus)",
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices to sweep (e.g. '16,17,18,19,20,21')",
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--higher_a", type=int, default=13)
    parser.add_argument("--higher_b", type=int, default=13)
    parser.add_argument("--out_dir", default="runs/concept_localization/carry")
    args = parser.parse_args()

    target_layers = [int(x.strip()) for x in args.layers.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmpl_str = _TEMPLATES.get(args.template, ("calc: {a}+{b}= ", ""))[0]
    log.info("Template %r → %r", args.template, tmpl_str)
    log.info("Layers: %s  anchor: %s  top_k: %d", target_layers, args.anchor_mode, args.top_k)

    device = get_default_device()
    dtype = parse_dtype(args.dtype)

    log.info("Loading model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()

    ranked, heatmaps = sweep_all_features_carry_score(
        model,
        target_layers=target_layers,
        anchor_mode=args.anchor_mode,
        template_str=tmpl_str,
        higher_a=args.higher_a,
        higher_b=args.higher_b,
        top_k=args.top_k,
    )

    ranked_path = out_dir / "carry_score_ranked.json"
    with open(ranked_path, "w") as f:
        json.dump(
            [
                {"layer": l, "feat_id": fi, "carry_score": round(s, 6), "jaccard": round(j, 6)}
                for l, fi, s, j in ranked
            ],
            f,
            indent=2,
        )
    log.info("Saved ranking → %s", ranked_path)

    npz_path = out_dir / "carry_score_heatmaps.npz"
    np.savez_compressed(
        npz_path,
        **{f"L{l}_F{f}": mat for (l, f), mat in heatmaps.items()},
    )
    log.info("Saved arrays → %s", npz_path)

    plot_path = out_dir / "carry_score_heatmaps.png"
    plot_carry_score_heatmaps(
        ranked,
        heatmaps,
        plot_path,
        concept="carry",
        anchor_label=args.anchor_mode,
        xlabel="ones(a)",
        ylabel="ones(b)",
    )
    log.info("Saved plot → %s", plot_path)
    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
