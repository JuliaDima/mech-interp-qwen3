"""Rank all transcoder features by carry discrimination score.

Uses the full carry dataset (all 3-digit pairs) to score every feature at the
target layers by

    carry_score = mean(act | carry) − mean(act | no_carry)
    jaccard     = |active ∩ mask| / |active ∪ mask|

ranked by jaccard × |carry_score|.  Heatmaps are built by averaging activations
over all dataset examples that share the same (ones(a), ones(b)) cell.

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
from data.concept_datasets.carry_dataset import generate_carry_pairs, make_anchor_positions
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

def _build_dataset_inputs(
    model,
    anchor_mode: str,
    template_str: str,
    n_pairs: int = 200,
) -> tuple[list[tuple[list[int], int]], np.ndarray, np.ndarray, np.ndarray]:
    """Build inputs from the carry dataset instead of a fixed higher-digit context.

    Returns
    -------
    inputs   list of (token_ids, anchor_pos), length 2*n_examples
    pos_mask (2*n_examples,) bool — True for carry examples
    ones_a   (2*n_examples,) int — ones digit of a
    ones_b   (2*n_examples,) int — ones digit of b
    """
    pairs = generate_carry_pairs(n_per_template=n_pairs, templates=["T0"])
    inputs, pos_mask, ones_a, ones_b = [], [], [], []
    for pair in pairs:
        for is_pos in (True, False):
            a = pair.meta["a_pos"] if is_pos else pair.meta["a_neg"]
            b = pair.meta["b_pos"] if is_pos else pair.meta["b_neg"]
            prompt = pair.prompt_pos if is_pos else pair.prompt_neg
            ids = model.tokenizer(prompt, add_special_tokens=False).input_ids
            try:
                anchor = min(int(anchor_mode), len(ids) - 1)
            except ValueError:
                positions = make_anchor_positions(template_str, a, b, model.tokenizer)
                anchor = resolve_anchor_from_positions(positions, anchor_mode, len(ids) - 1)
            inputs.append((ids, anchor))
            pos_mask.append(is_pos)
            ones_a.append(a % 10)
            ones_b.append(b % 10)
    return inputs, np.array(pos_mask), np.array(ones_a), np.array(ones_b)


def _bin_to_heatmap(acts_col: np.ndarray, ones_a: np.ndarray, ones_b: np.ndarray) -> np.ndarray:
    """Average a feature's activations into a (10, 10) ones-digit grid."""
    sums   = np.zeros((10, 10), dtype=np.float64)
    counts = np.zeros((10, 10), dtype=np.int64)
    for i in range(len(acts_col)):
        sums[ones_a[i], ones_b[i]]   += acts_col[i]
        counts[ones_a[i], ones_b[i]] += 1
    grid = np.full((10, 10), np.nan, dtype=np.float32)
    mask = counts > 0
    grid[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    return grid


@torch.no_grad()
def sweep_all_features_carry_score(
    model,
    target_layers: list[int],
    anchor_mode: str = "ones_b",
    template_str: str = "calc: {a}+{b}= ",
    n_pairs: int = 200,
    top_k: int = 50,
) -> tuple[list[tuple[int, int, float, float]], dict[tuple[int, int], np.ndarray]]:
    """Rank all transcoder features at target_layers by carry discrimination.

    Uses the full carry dataset so heatmaps average over all higher-digit
    contexts rather than a single fixed one.

    Returns:
        ranked   list of (layer, feat_id, carry_score, jaccard) sorted by
                 jaccard × |carry_score| descending
        heatmaps dict (layer, feat_id) → (10, 10) float32 array for the top_k features
    """
    inputs, pos_mask, ones_a, ones_b = _build_dataset_inputs(
        model, anchor_mode, template_str, n_pairs
    )
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

        ranked_layer = score_and_rank(acts_np, pos_mask, top_k=top_k)
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
            heatmaps[(layer, feat_id)] = _bin_to_heatmap(
                layer_acts[layer][:, feat_id], ones_a, ones_b
            )

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
        help=(
            "Anchor for the ones-digit grid. Use an integer (token position), "
            "a named key (ones_a / ones_b / plus), or 'topN' (e.g. 'top5') to "
            "run the sweep for the top-N anchors from emergence.npy."
        ),
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices to sweep (e.g. '16,17,18,19,20,21')",
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--n_pairs", type=int, default=200,
                        help="Pairs per template passed to generate_carry_pairs")
    parser.add_argument("--out_dir", default="runs/concept_localization/carry")
    args = parser.parse_args()

    target_layers = [int(x.strip()) for x in args.layers.split(",")]
    base_out_dir  = Path(args.out_dir)
    tmpl_str      = _TEMPLATES.get(args.template, ("calc: {a}+{b}= ", ""))[0]

    # Resolve anchor list — supports "topN" to read from emergence.npy
    import re as _re
    m_top = _re.fullmatch(r"top(\d+)", args.anchor_mode)
    if m_top:
        n_top = int(m_top.group(1))
        sys.path.insert(0, str(_REPO_ROOT))
        from experiments.concept_localization.plot_anchor_analysis import (
            load_emergence, top_k_anchors,
        )
        em = load_emergence("carry")
        if em is None:
            raise RuntimeError("emergence.npy not found for carry — run make_gif first")
        anchors_info = top_k_anchors(em, "carry", k=n_top)
        anchor_list = [(str(idx), f"pos{idx}_{tok}") for idx, _, tok in anchors_info]
        log.info("Top-%d anchors from emergence.npy: %s",
                 n_top, [(a, lbl) for a, lbl in anchor_list])
    else:
        anchor_list = [(args.anchor_mode, args.anchor_mode)]

    device = get_default_device()
    dtype  = parse_dtype(args.dtype)

    log.info("Loading model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()

    for anchor_mode, anchor_label in anchor_list:
        out_dir = base_out_dir / f"anchor_{anchor_label}" / "carry_sweep" \
            if len(anchor_list) > 1 else base_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("─" * 60)
        log.info("Anchor: %s  label: %s  layers: %s  top_k: %d",
                 anchor_mode, anchor_label, target_layers, args.top_k)

        ranked, heatmaps = sweep_all_features_carry_score(
            model,
            target_layers=target_layers,
            anchor_mode=anchor_mode,
            template_str=tmpl_str,
            n_pairs=args.n_pairs,
            top_k=args.top_k,
        )

        ranked_path = out_dir / "carry_score_ranked.json"
        with open(ranked_path, "w") as f:
            json.dump(
                [
                    {"layer": l, "feat_id": fi,
                     "carry_score": round(s, 6), "jaccard": round(j, 6)}
                    for l, fi, s, j in ranked
                ],
                f, indent=2,
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
            ranked, heatmaps, plot_path,
            concept="carry", anchor_label=anchor_label,
            xlabel="ones(a)", ylabel="ones(b)",
        )
        log.info("Saved plot → %s", plot_path)
        log.info("Done for anchor %s. Outputs in %s", anchor_label, out_dir)


if __name__ == "__main__":
    main()
