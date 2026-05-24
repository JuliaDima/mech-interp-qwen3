"""Targeted operand sweep capturing activations at any token position.

Unlike the default operand_plots.py (which always captures at the '=' token),
this script captures at a specified token position within the prompt.  This
is needed to characterise ADD FUNCTION features at the operand digit positions
and SUM/LOOKUP features at the '=' position, matching Anthropic's taxonomy:

  ctx4 (token "3") — tens digit of first operand
  ctx5 (token "6") — ones digit of first operand  →  ADD FUNCTION (vertical stripes)
  ctx8 (token "9") — ones digit of second operand →  ADD FUNCTION (horizontal stripes)
  ctx9 (token "=") — equal sign                  →  SUM / LOOKUP features

Only 2-digit operands (a, b ∈ [10, 99]) are used so that every prompt has the
same 10-token sequence and position indices are consistent.

Usage:
  python experiments/addition/operand_plots_by_position.py \
      --graph_json    runs/addition/graph/focus_36_59.json \
      --model         Qwen/Qwen3-4B \
      --transcoder_set mwhanna/qwen3-4b-transcoders \
      --dtype         bfloat16 \
      --ctx_positions 5 8 9 \
      --top_k         15 \
      --out_dir       runs/addition/operand_plots_by_pos
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger(__name__)

# Token positions (0-indexed) for "calc: {a}+{b}= " with 2-digit a, b
# "calc" ":" " " "3" "6" "+" "5" "9" "=" " "
#  idx0   1   2   3   4   5   6   7   8   9
POS_LABEL = {
    3: "tens(a)",
    4: "ones(a)",
    5: "+",
    6: "tens(b)",
    7: "ones(b)",
    8: "=",
    9: "trailing_space",
}


def load_graph_features(
    graph_json: Path,
    ctx_positions: list[int],
    top_k: int,
) -> dict[int, list[tuple[int, int]]]:
    """Extract top-k features per ctx position from the attribution graph.

    Returns dict mapping ctx_idx (1-indexed, as in the graph) to list of
    (layer, feat_idx) pairs sorted by total edge weight.
    """
    with open(graph_json) as f:
        g = json.load(f)

    nodes = g["nodes"]
    links = g["links"]

    edge_weight: dict[str, float] = defaultdict(float)
    for lk in links:
        edge_weight[lk["source"]] += abs(lk["weight"])
        edge_weight[lk["target"]] += abs(lk["weight"])

    result: dict[int, list[tuple[int, int]]] = {}
    for ctx in ctx_positions:
        group = [
            n for n in nodes
            if n["feature_type"] == "CLT"
            and not n["is_target_logit"]
            and n["ctx_idx"] == ctx
            and (n["activation"] or 0) > 0.3
        ]
        group.sort(key=lambda n: edge_weight[n["node_id"]], reverse=True)
        feats = []
        for n in group[:top_k]:
            parts = n["node_id"].split("_")
            feats.append((int(parts[0]), int(parts[1])))
        result[ctx] = feats
        log.info("ctx%d: %d features selected", ctx, len(feats))
    return result


@torch.no_grad()
def sweep_at_position(
    model,
    feature_ids: list[tuple[int, int]],
    token_pos: int,       # 0-indexed token position to capture
    a_range: range = range(10, 100),
    b_range: range = range(10, 100),
    batch_size: int = 32,
) -> dict[tuple[int, int], np.ndarray]:
    """Sweep (a, b) pairs and record feature activations at a fixed token position.

    Only 2-digit a, b are supported (consistent 10-token sequences).

    Returns dict (layer, feat_idx) → (90, 90) float32 array (a axis, b axis).
    """
    n_a, n_b = len(a_range), len(b_range)
    matrices = {lf: np.zeros((n_a, n_b), dtype=np.float32) for lf in feature_ids}

    prompts = [
        (ai, bi, f"calc: {a}+{b}= ")
        for ai, a in enumerate(a_range)
        for bi, b in enumerate(b_range)
    ]

    for start in tqdm(range(0, len(prompts), batch_size), desc=f"sweep@pos{token_pos}"):
        batch = prompts[start:start + batch_size]
        for ai, bi, prompt in batch:
            _, acts = model.get_activations(prompt)
            # acts: (n_layers, seq_len, d_transcoder)
            if token_pos >= acts.shape[1]:
                continue
            for layer, feat_idx in feature_ids:
                val = float(acts[layer, token_pos, feat_idx].item())
                matrices[(layer, feat_idx)][ai, bi] = val

    return matrices


def plot_combined(
    matrices: dict[tuple[int, int], np.ndarray],
    out_dir: Path,
    pos_label: str,
    a_range: range,
    b_range: range,
    reference_prompt: str,
) -> None:
    import math
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(_REPO_ROOT))
    from experiments.plot_style import apply
    apply()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw matrices
    for (layer, feat), mat in matrices.items():
        npy_path = out_dir / f"pos{pos_label}_L{layer:02d}_F{feat:06d}.npy"
        np.save(str(npy_path), mat)

    if not matrices:
        return

    items = list(matrices.items())
    n = len(items)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    cell = 3.2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * cell, nrows * cell + 1.0),
        squeeze=False,
    )

    extent = [a_range[0], a_range[-1], b_range[0], b_range[-1]]

    for idx, ((layer, feat), mat) in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        im = ax.imshow(
            mat.T, origin="lower", aspect="auto", cmap="viridis", extent=extent,
        )
        ax.set_title(f"L{layer:02d} F{feat:06d}", fontsize=8, pad=3)
        ax.set_xlabel("a", fontsize=7, labelpad=2)
        ax.set_ylabel("b", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"Prompt: \"{reference_prompt}\"    token position: {pos_label}",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()

    fname = out_dir / f"combined_{pos_label}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved combined figure (%d subplots) to %s", n, fname)


def score_single_operand(mat: np.ndarray, axis: int) -> tuple[float, int]:
    """Score how well the matrix is explained by a single ones-digit.

    For vertical stripes (axis=0), compute how much variance is explained
    by the column index mod 10.  Returns (score in [0,1], best digit).
    """
    n = mat.shape[axis]
    total_var = float(np.var(mat))
    if total_var < 1e-9:
        return 0.0, 0

    # Average along the OTHER axis
    profile = mat.mean(axis=1 - axis)   # shape (n,)
    mod10_means = np.zeros(10)
    for d in range(10):
        idx = np.arange(d, n, 10)
        if len(idx):
            mod10_means[d] = profile[idx].mean()
    best_digit = int(np.argmax(mod10_means))
    # Explained variance: compare profile to its mod-10 periodic reconstruction
    reconstructed = np.array([mod10_means[i % 10] for i in range(n)])
    explained = 1 - float(np.mean((profile - reconstructed) ** 2)) / (float(np.var(profile)) + 1e-9)
    return max(0.0, float(explained)), best_digit


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--graph_json", default="runs/addition/graph/focus_36_59.json")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--transcoder_set", default="mwhanna/qwen3-4b-transcoders")
    p.add_argument("--dtype", default="bfloat16", choices=["float32","bfloat16","float16"])
    p.add_argument("--ctx_positions", nargs="+", type=int, default=[5, 8, 9],
                   help="ctx_idx values (1-indexed) from the graph to sweep")
    p.add_argument("--top_k", type=int, default=20,
                   help="Top-k features per position by graph edge weight")
    p.add_argument("--out_dir", default="runs/addition/operand_plots_by_pos")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--reference_prompt", default="calc: 36+59=",
                   help="Prompt shown in figure title to identify the graph source")
    args = p.parse_args()

    from mechinterp_qwen3.attribution_model import AttributionModel
    from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub

    dtype = getattr(torch, args.dtype)
    transcoder, config = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_decoder=True
    )
    model_name = args.model or config.get("model_name", "Qwen/Qwen3-4B")
    model = AttributionModel.from_pretrained_and_transcoders(model_name, transcoder, dtype=dtype)
    model.eval()

    graph_features = load_graph_features(
        Path(args.graph_json), args.ctx_positions, args.top_k
    )

    a_range = range(10, 100)
    b_range = range(10, 100)

    summary = {}
    for ctx_idx, feat_ids in graph_features.items():
        if not feat_ids:
            continue
        # ctx_idx is 1-indexed; token_pos is 0-indexed
        token_pos = ctx_idx - 1
        pos_label = POS_LABEL.get(token_pos, f"pos{token_pos}")
        log.info("Sweeping %d features at ctx%d (%s) ...", len(feat_ids), ctx_idx, pos_label)

        matrices = sweep_at_position(
            model, feat_ids, token_pos,
            a_range=a_range, b_range=b_range,
            batch_size=args.batch_size,
        )

        out_dir = Path(args.out_dir) / f"ctx{ctx_idx}_{pos_label}"
        plot_combined(matrices, out_dir, pos_label, a_range, b_range, args.reference_prompt)

        # Score each matrix (ctx9 = sum/lookup features; axis=1 is arbitrary there)
        axis = 0 if ctx_idx == 5 else 1  # ctx5=ones(a)→vertical, ctx8=ones(b)→horizontal
        for (layer, feat), mat in matrices.items():
            sc, best_d = score_single_operand(mat, axis=axis)
            summary[f"L{layer:02d}_F{feat:06d}_ctx{ctx_idx}"] = {
                "layer": layer, "feat": feat, "ctx_idx": ctx_idx,
                "pos_label": pos_label,
                "mod10_score": round(sc, 4),
                "best_digit": best_d,
                "peak": round(float(mat.max()), 4),
            }

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {out_root}/summary.json")
    rows = sorted(summary.values(), key=lambda x: x["mod10_score"], reverse=True)
    print(f"\n{'Feature':18s}  {'ctx':5s}  {'mod10':6s}  {'best_d':6s}  peak")
    for r in rows[:20]:
        print(f"L{r['layer']:02d} F{r['feat']:06d}  ctx{r['ctx_idx']}   {r['mod10_score']:6.4f}   {r['best_digit']:6d}  {r['peak']:.2f}")


if __name__ == "__main__":
    main()
