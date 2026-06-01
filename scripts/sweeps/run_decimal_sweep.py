"""Rank all transcoder features by decimal-termination discrimination score.

Sweeps all denominator values from the _POS and _NEG lists under template T0
and scores each transcoder feature by

    score   = mean(act | terminates) − mean(act | repeats)
    jaccard = |active ∩ mask| / |active ∪ mask|

ranked by jaccard × |score|.  Outputs a JSON ranking and a bar-chart grid
showing activation per denominator value, coloured by terminating (blue)
vs repeating (red).

Usage
-----
    python scripts/decimal/run_decimal_sweep.py --layers 4,5,6,17,18,19,20
    python scripts/decimal/run_decimal_sweep.py --layers 18 19 20 --top_k 50 --anchor digit_3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.patches import Patch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
from sweep_utils import apply_transcoder_all, cluster_top_features, resolve_anchor_from_positions  # noqa: E402

import experiments.plot_style as ps
from data.concept_datasets.decimal_termination_dataset import (
    _NEG,
    _POS,
    TEMPLATES,
    make_anchor_positions,
)
from experiments.concept_localization.analyze import collect_layer_residuals
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_decimal_sweep")

_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"
_TEMPLATE = "T0"


def _build_inputs(
    model, template_str: str, n_values: list[int], anchor_mode: str
) -> tuple[list[tuple[list[int], int]], list[int]]:
    """Return (inputs, valid_n) — n values without the requested digit are skipped."""
    inputs, valid_n = [], []
    for n in n_values:
        positions = make_anchor_positions(template_str, n, model.tokenizer)
        prompt = template_str.format(n=n)
        ids = model.tokenizer(prompt, add_special_tokens=False).input_ids
        anchor = resolve_anchor_from_positions(positions, anchor_mode, len(ids) - 1)
        inputs.append((ids, anchor))
        valid_n.append(n)
    return inputs, valid_n


@torch.no_grad()
def sweep_all_features_decimal_score(
    model,
    target_layers: list[int],
    anchor_mode: str = "digit_1",
    template_str: str = TEMPLATES[_TEMPLATE][0],
    top_frac: float = 0.15,
    n_clusters: int = 10,
) -> tuple[
    list[tuple[int, int, float, int]], dict[tuple[int, int], np.ndarray], list[int], np.ndarray
]:
    """Sweep features at target_layers; cluster top features by activation pattern.

    No mask or shape prior — clustering discovers which patterns exist.

    Returns:
        all_clustered  list of (layer, feat_id, score, cluster_id) sorted by
                       layer, then cluster_id, then |score| descending
        acts_1d        dict (layer, feat_id) → (N,) float32 activation per denominator
        n_values       list of denominator values in sweep order (sorted by magnitude)
        pos_mask       bool array, True where n is terminating
    """
    all_n = sorted(set(_POS) | set(_NEG))
    pos_set = set(_POS)

    inputs, n_values = _build_inputs(model, template_str, all_n, anchor_mode)
    pos_mask = np.array([n in pos_set for n in n_values], dtype=bool)
    log.info("Anchor %r covers %d / %d n values", anchor_mode, len(n_values), len(all_n))

    H = collect_layer_residuals(model, inputs, target_layers)

    all_clustered: list[tuple[int, int, float, int]] = []
    acts_1d: dict[tuple[int, int], np.ndarray] = {}

    for layer in target_layers:
        if layer not in H:
            continue
        try:
            acts_np = apply_transcoder_all(model, layer, H[layer])
        except (IndexError, KeyError, AttributeError):
            log.warning("No transcoder at layer %d — skipping", layer)
            continue

        clustered = cluster_top_features(
            acts_np, pos_mask, top_frac=top_frac, n_clusters=n_clusters
        )
        for feat_id, score, cluster_id in clustered:
            all_clustered.append((layer, feat_id, score, cluster_id))
            acts_1d[(layer, feat_id)] = acts_np[:, feat_id]

        n_selected = len(clustered)
        top_score = max(clustered, key=lambda x: abs(x[1]))[1]
        log.info(
            "Layer %2d  d_tc=%d  top_frac→%d features  %d clusters  top |score|=%.4f",
            layer,
            acts_np.shape[1],
            n_selected,
            n_clusters,
            abs(top_score),
        )

    # Sort by layer, then cluster_id, then |score| descending for the plot
    all_clustered.sort(key=lambda x: (x[0], x[3], -abs(x[2])))
    return all_clustered, acts_1d, n_values, pos_mask


def plot_decimal_activation_bars(
    clustered: list[tuple[int, int, float, int]],
    acts_1d: dict[tuple[int, int], np.ndarray],
    n_values: list[int],
    pos_mask: np.ndarray,
    out_path: Path,
    top_per_cluster: int = 2,
) -> None:
    """Bar chart grid grouped by layer and cluster.

    For each (layer, cluster) group, shows the top `top_per_cluster` features
    by |score|.  Features in the same cluster have similar activation shapes.
    Bars are blue for terminating denominators, red for non-terminating.
    """
    # Build ordered list: for each (layer, cluster_id) group, keep top N by |score|
    from itertools import groupby

    display_items: list[tuple[int, int, float, int]] = []
    for (layer, cluster_id), group in groupby(clustered, key=lambda x: (x[0], x[3])):
        top = sorted(group, key=lambda x: -abs(x[2]))[:top_per_cluster]
        display_items.extend(top)

    if not display_items:
        return

    # Number of columns = top_per_cluster × n_clusters (per layer row)
    n_layers = len(dict.fromkeys(x[0] for x in display_items))
    n_clusters_shown = len(dict.fromkeys((x[0], x[3]) for x in display_items)) // max(n_layers, 1)
    ncols = max(1, top_per_cluster * n_clusters_shown)
    nrows = n_layers

    ps.apply()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2.8 + 0.5, nrows * 2.8 + 1.2),
        squeeze=False,
    )

    x = np.arange(len(n_values))
    bar_colors = ["#2196F3" if m else "#E53935" for m in pos_mask]
    step = max(1, len(n_values) // 10)

    layer_order = list(dict.fromkeys(x[0] for x in display_items))
    col_idx_per_layer = {l: 0 for l in layer_order}

    for layer, feat_id, score, cluster_id in display_items:
        row = layer_order.index(layer)
        col = col_idx_per_layer[layer]
        col_idx_per_layer[layer] += 1
        if col >= ncols:
            continue
        ax = axes[row][col]
        acts = acts_1d.get((layer, feat_id))
        if acts is None:
            ax.set_visible(False)
            continue
        ax.bar(x, acts, color=bar_colors, width=0.8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(
            [str(n_values[i]) for i in range(0, len(n_values), step)],
            fontsize=4,
            rotation=45,
            ha="right",
        )
        ax.tick_params(labelsize=4)
        sign = "+" if score >= 0 else ""
        ax.set_title(
            f"L{layer:02d} F{feat_id:06d}  C{cluster_id}\n{sign}{score:.2f}",
            fontsize=6,
            pad=2,
        )

    for row in range(nrows):
        for col in range(col_idx_per_layer.get(layer_order[row], 0), ncols):
            axes[row][col].set_visible(False)

    fig.legend(
        handles=[
            Patch(facecolor="#2196F3", label="terminates"),
            Patch(facecolor="#E53935", label="repeats"),
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.8,
    )
    fig.suptitle(
        "decimal termination — features grouped by cluster (same column = same cluster)\n"
        "rows = layers, columns = cluster × rank within cluster",
        fontsize=9,
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
        "--anchor",
        default="digit_1",
        help="Anchor mode: digit_1 (ones), digit_2 (tens), digit_3 (hundreds). "
        "n values without that many digits are skipped.",
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices (e.g. '4,5,6,17,18,19,20')",
    )
    parser.add_argument(
        "--top_frac",
        type=float,
        default=0.15,
        help="Fraction of features per layer to select by |score| before clustering",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="Number of k-means clusters per layer",
    )
    parser.add_argument(
        "--top_per_cluster",
        type=int,
        default=2,
        help="Features to display per (layer, cluster) group",
    )
    parser.add_argument("--out_dir", default="runs/concept_localization/decimal_termination")
    args = parser.parse_args()

    target_layers = [int(x.strip()) for x in args.layers.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmpl_str = TEMPLATES[_TEMPLATE][0]
    log.info("Template T0: %r  anchor: %s", tmpl_str, args.anchor)
    log.info(
        "Layers: %s  top_frac: %.2f  n_clusters: %d", target_layers, args.top_frac, args.n_clusters
    )

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

    clustered, acts_1d, n_values, pos_mask = sweep_all_features_decimal_score(
        model,
        target_layers=target_layers,
        anchor_mode=args.anchor,
        template_str=tmpl_str,
        top_frac=args.top_frac,
        n_clusters=args.n_clusters,
    )

    ranked_path = out_dir / "decimal_score_ranked.json"
    with open(ranked_path, "w") as f:
        json.dump(
            [
                {"layer": l, "feat_id": fi, "score": round(s, 6), "cluster": c}
                for l, fi, s, c in clustered
            ],
            f,
            indent=2,
        )
    log.info("Saved ranking → %s", ranked_path)

    plot_path = out_dir / "decimal_score_bars.png"
    plot_decimal_activation_bars(
        clustered,
        acts_1d,
        n_values,
        pos_mask,
        plot_path,
        top_per_cluster=args.top_per_cluster,
    )
    log.info("Saved plot → %s", plot_path)
    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
