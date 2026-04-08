"""Visualise steering-vector results.

Two figures are produced:

1. Sweep plot  — first-token accuracy, full-answer accuracy, and
                 mean P(true first answer token) vs alpha
2. Grid plot   — addition grid coloured by outcome under a
                 correctness metric (first-token or full-answer)

Usage
-----
    python experiments/knowledge_editing/steer_plot.py
    python experiments/knowledge_editing/steer_plot.py --grid_metric full_answer
    python experiments/knowledge_editing/steer_plot.py --best_by delta_first_token_acc
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from steer import (
    FIRST_TOKEN_METRIC,
    FULL_ANSWER_METRIC,
    STEERING_LABEL_METRICS,
    _digit_count,
    _eq_pos,
    _metric_label,
    _predict_sample,
    collect_steering_vecs,
    precompute_baseline,
)

from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("steer_plot")

_RESULTS_PATH = "runs/knowledge_editing/steer_results.json"
_DATASET = "data/addition_grid.jsonl"
_OUT_DIR = "runs/knowledge_editing/plots"
_LARGE_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"

_METRIC_SPECS = [
    (
        "first_token_acc",
        "delta_first_token_acc",
        "First-token accuracy (%)",
        "First-token accuracy vs α",
    ),
    (
        "full_answer_acc",
        "delta_full_answer_acc",
        "Full-answer accuracy (%)",
        "Full-answer accuracy vs α",
    ),
    (
        "mean_true_first_token_prob",
        "delta_mean_true_first_token_prob",
        "Mean P(true first answer token)",
        "Mean P(true first answer token) vs α",
    ),
]
_BEST_BY_CHOICES = tuple(delta_key for _, delta_key, _, _ in _METRIC_SPECS)


# ---------------------------------------------------------------------------
# Plot 1 — sweep
# ---------------------------------------------------------------------------


def plot_sweep(results: list[dict], out_dir: Path) -> None:
    layers = sorted(set(row["layer"] for row in results))
    bucket_metric = results[0].get("bucket_metric", FIRST_TOKEN_METRIC)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colours = plt.cm.tab10(np.linspace(0, 0.6, len(layers)))

    for ax, (metric, delta_metric, ylabel, title) in zip(axes, _METRIC_SPECS, strict=False):
        for layer, color in zip(layers, colours, strict=False):
            layer_rows = sorted(
                [row for row in results if row["layer"] == layer],
                key=lambda row: row["alpha"],
            )
            xs = [row["alpha"] for row in layer_rows]
            ys = [row[metric] for row in layer_rows]
            ax.plot(xs, ys, marker="o", label=f"L={layer}", color=color)

        base = results[0][metric] - results[0][delta_metric]
        ax.axhline(base, linestyle="--", color="grey", linewidth=1, label="baseline")
        ax.set_xlabel("α (steering scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Steering vectors — live eval metrics" f"  (SV buckets: {_metric_label(bucket_metric)})",
        fontsize=12,
    )
    fig.tight_layout()
    path = out_dir / "steer_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved sweep plot → %s", path)


# ---------------------------------------------------------------------------
# Plot 2 — grid
# ---------------------------------------------------------------------------


def run_grid_eval(
    model: AttributionModel,
    samples: list[dict],
    layer: int,
    alpha: float,
    sv: torch.Tensor,
    eq_token_id: int,
    device: torch.device,
) -> list[dict]:
    """Return per-sample dicts with both correctness definitions before/after."""
    sv = sv.to(device)
    per_sample = []

    model.eval()
    with torch.no_grad():
        for sample in tqdm(samples, desc=f"Grid eval L={layer} α={alpha}"):
            token_ids: list[int] = sample["prompt_token_ids"]
            eq_pos = _eq_pos(token_ids, eq_token_id)
            if eq_pos is None:
                continue

            before = _predict_sample(model, sample, device)

            def _steer(act, hook, _sv=sv, _pos=eq_pos, _a=alpha):
                act = act.clone()
                act[0, _pos, :] = act[0, _pos, :] + _a * _sv
                return act

            after = _predict_sample(
                model,
                sample,
                device,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post", _steer)],
            )

            per_sample.append(
                {
                    "a": sample["a"],
                    "b": sample["b"],
                    "first_token_correct_before": bool(before["first_token_correct"]),
                    "first_token_correct_after": bool(after["first_token_correct"]),
                    "full_answer_correct_before": bool(before["full_answer_correct"]),
                    "full_answer_correct_after": bool(after["full_answer_correct"]),
                }
            )

    return per_sample


def plot_grid(
    per_sample: list[dict],
    layer: int,
    alpha: float,
    out_dir: Path,
    grid_metric: str,
    bucket_metric: str,
    best_by: str,
) -> None:
    max_a = max(sample["a"] for sample in per_sample)
    max_b = max(sample["b"] for sample in per_sample)
    before_key = f"{grid_metric}_correct_before"
    after_key = f"{grid_metric}_correct_after"

    colors = {
        0: "#E84040",
        1: "#4CAF50",
        2: "#FF9800",
        3: "#1B7E35",
    }

    grid = np.full((max_a + 1, max_b + 1), -1, dtype=int)
    for sample in per_sample:
        before = sample[before_key]
        after = sample[after_key]
        grid[sample["a"], sample["b"]] = before * 2 + after

    cmap = matplotlib.colors.ListedColormap([colors[i] for i in range(4)])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(
        grid,
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=[-0.5, max_b + 0.5, -0.5, max_a + 0.5],
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(-0.5, max_b + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, max_a + 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(np.arange(0, max_b + 1, 5))
    ax.set_yticks(np.arange(0, max_a + 1, 5))
    ax.tick_params(labelsize=10)
    ax.set_xlabel("b", fontsize=12)
    ax.set_ylabel("a", fontsize=12)

    n_rescued = sum(1 for sample in per_sample if not sample[before_key] and sample[after_key])
    n_broken = sum(1 for sample in per_sample if sample[before_key] and not sample[after_key])
    n_total = len(per_sample)
    acc_before = 100 * sum(sample[before_key] for sample in per_sample) / n_total
    acc_after = 100 * sum(sample[after_key] for sample in per_sample) / n_total

    ax.set_title(
        f"Steering vector  L={layer}  α={alpha}\n"
        f"grid metric: {_metric_label(grid_metric)}  |  SV buckets: {_metric_label(bucket_metric)}\n"
        f"selected by {best_by}  |  acc {acc_before:.1f}% → {acc_after:.1f}%  |  rescued {n_rescued}  broken {n_broken}",
        fontsize=11,
        pad=10,
    )

    patches = [
        mpatches.Patch(color=colors[1], label="wrong → correct  (rescued)"),
        mpatches.Patch(color=colors[3], label="correct → correct"),
        mpatches.Patch(color=colors[2], label="correct → wrong  (broken)"),
        mpatches.Patch(color=colors[0], label="wrong → wrong"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.85, edgecolor="grey")

    fig.tight_layout()
    path = out_dir / f"steer_grid_{grid_metric}_L{layer}_a{alpha}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved grid plot → %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results", default=_RESULTS_PATH)
    parser.add_argument("--dataset", default=_DATASET)
    parser.add_argument("--out_dir", default=_OUT_DIR)
    parser.add_argument("--model", default=_LARGE_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layer", type=int, default=None, help="Layer for grid plot")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha for grid plot")
    parser.add_argument("--collect_n", type=int, default=300)
    parser.add_argument(
        "--bucket_metric",
        choices=STEERING_LABEL_METRICS,
        default=None,
        help="SV bucket metric to filter results by. Defaults to the metric stored in the results file.",
    )
    parser.add_argument(
        "--grid_metric",
        choices=STEERING_LABEL_METRICS,
        default=FULL_ANSWER_METRIC,
        help="Correctness definition shown in the grid plot.",
    )
    parser.add_argument(
        "--best_by",
        choices=_BEST_BY_CHOICES,
        default="delta_full_answer_acc",
        help="Metric used to choose the default (layer, alpha) pair.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        all_results = json.load(f)

    required_keys = {
        "first_token_acc",
        "delta_first_token_acc",
        "full_answer_acc",
        "delta_full_answer_acc",
        "mean_true_first_token_prob",
        "delta_mean_true_first_token_prob",
    }
    missing = sorted(required_keys - set(all_results[0]))
    if missing:
        raise RuntimeError(
            "Results file is missing the new steering metrics: "
            + ", ".join(missing)
            + ". Re-run experiments/knowledge_editing/steer.py first."
        )

    bucket_metric = args.bucket_metric or all_results[0].get("bucket_metric", FIRST_TOKEN_METRIC)
    results = [
        row for row in all_results if row.get("bucket_metric", FIRST_TOKEN_METRIC) == bucket_metric
    ]
    if not results:
        raise RuntimeError(f"No results found for bucket metric {bucket_metric!r}")

    plot_sweep(results, out_dir)

    best = max(results, key=lambda row: row[args.best_by])
    layer = args.layer if args.layer is not None else best["layer"]
    alpha = args.alpha if args.alpha is not None else best["alpha"]
    log.info(
        "Grid plot: layer=%d  alpha=%.1f  grid_metric=%s  best_by=%s",
        layer,
        alpha,
        args.grid_metric,
        args.best_by,
    )

    with open(args.dataset) as f:
        all_samples = [json.loads(line) for line in f]

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

    eq_token_id: int = model.tokenizer("=", add_special_tokens=False).input_ids[-1]

    hard = [sample for sample in all_samples if _digit_count(sample) >= 2]
    collect_pool = hard[: len(hard) // 2]
    collect_baseline = precompute_baseline(model, collect_pool, device)
    svecs = collect_steering_vecs(
        model,
        collect_pool,
        collect_baseline,
        [layer],
        eq_token_id,
        args.collect_n,
        device,
        dtype,
        bucket_metric,
    )
    if layer not in svecs:
        raise RuntimeError(f"Could not recompute steering vector for layer {layer}")
    sv = svecs[layer]

    per_sample = run_grid_eval(model, all_samples, layer, alpha, sv, eq_token_id, device)
    plot_grid(per_sample, layer, alpha, out_dir, args.grid_metric, bucket_metric, args.best_by)


if __name__ == "__main__":
    main()
