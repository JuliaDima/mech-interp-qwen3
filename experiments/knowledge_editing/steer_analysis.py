"""Deep analysis of a steering vector.

Three outputs
-------------
1. Logit lens       — which tokens the steering vector promotes / suppresses
                      (sv @ W_U, top-k positive and negative)
2. Feature maps     — for the top-k transcoder features aligned with sv,
                      plot a 31×31 heatmap of their activation across the
                      addition grid (a vs b)
3. Improved outputs — sample prompts where steering improves either
                      first-token or full-answer correctness

Usage
-----
    python experiments/knowledge_editing/steer_analysis.py
    python experiments/knowledge_editing/steer_analysis.py --jsonl_path data/addition_grid.jsonl
    python experiments/knowledge_editing/steer_analysis.py --jsonl_path data/addition_grid.jsonl --out_dir runs/knowledge_editing/plots
    python experiments/knowledge_editing/steer_analysis.py --layer 16 --alpha 5 --bucket_metric full_answer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open
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
    _metric_key,
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
log = logging.getLogger("steer_analysis")


# ---------------------------------------------------------------------------
# Recompute steering vector
# ---------------------------------------------------------------------------


def recompute_sv(
    model: AttributionModel,
    samples: list[dict],
    layer: int,
    eq_token_id: int,
    collect_n: int,
    device: torch.device,
    dtype: torch.dtype,
    bucket_metric: str,
) -> torch.Tensor:
    baseline = precompute_baseline(model, samples, device)
    svecs = collect_steering_vecs(
        model,
        samples,
        baseline,
        [layer],
        eq_token_id,
        collect_n,
        device,
        dtype,
        bucket_metric,
    )
    if layer not in svecs:
        raise RuntimeError(
            f"Could not compute steering vector for layer {layer} using {_metric_label(bucket_metric)}"
        )
    sv = svecs[layer]
    log.info(
        "SV recomputed for %s buckets: norm=%.3f", _metric_label(bucket_metric), sv.norm().item()
    )
    return sv


# ---------------------------------------------------------------------------
# Collect per-sample residual stream activations
# ---------------------------------------------------------------------------


def collect_residuals(
    model: AttributionModel,
    samples: list[dict],
    layer: int,
    eq_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[dict]]:
    """Returns (acts, meta) where acts[i] = h[eq_pos]."""
    acts, meta = [], []
    model.eval()
    with torch.no_grad():
        for sample in tqdm(samples, desc="Capturing residuals"):
            token_ids = sample["prompt_token_ids"]
            eq_pos = _eq_pos(token_ids, eq_token_id)
            if eq_pos is None:
                continue
            cache = {}
            model.run_with_hooks(
                torch.tensor([token_ids], dtype=torch.long, device=device),
                fwd_hooks=[
                    (
                        f"blocks.{layer}.hook_resid_post",
                        lambda act, hook, _p=eq_pos: (
                            cache.update({"h": act[0, _p, :].detach().clone()}) or act
                        ),
                    )
                ],
            )
            if "h" in cache:
                acts.append(cache["h"].to(dtype=torch.float32))
                meta.append({"a": sample["a"], "b": sample["b"]})
    return torch.stack(acts), meta


# ---------------------------------------------------------------------------
# 1. Logit lens
# ---------------------------------------------------------------------------


def logit_lens(
    sv: torch.Tensor, model: AttributionModel, layer: int, topk: int, out_dir: Path
) -> None:
    w_unembed = model.W_U.float()
    scores = sv.float() @ w_unembed
    top_pos = scores.topk(topk)
    top_neg = (-scores).topk(topk)

    log.info("=== Logit lens — top promoted tokens ===")
    for rank, (val, idx) in enumerate(zip(top_pos.values, top_pos.indices, strict=False)):
        tok = model.tokenizer.decode([idx.item()])
        log.info("  +%d  %+.3f  %r", rank + 1, val.item(), tok)

    log.info("=== Logit lens — top suppressed tokens ===")
    for rank, (val, idx) in enumerate(zip(top_neg.values, top_neg.indices, strict=False)):
        tok = model.tokenizer.decode([idx.item()])
        log.info("  -%d  %+.3f  %r", rank + 1, val.item(), tok)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, vals, idxs, color, title in [
        (axes[0], top_pos.values, top_pos.indices, "#2ca02c", f"Top-{topk} promoted"),
        (axes[1], top_neg.values, top_neg.indices, "#d62728", f"Top-{topk} suppressed"),
    ]:
        labels = [repr(model.tokenizer.decode([i.item()])) for i in idxs]
        ax.barh(range(topk), vals.cpu().float().numpy(), color=color)
        ax.set_yticks(range(topk))
        ax.set_yticklabels(labels[::-1] if color == "#d62728" else labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Logit change")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Logit lens of steering vector (layer {layer})", fontsize=12)
    fig.tight_layout()
    path = out_dir / "steer_logit_lens.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# 2. Feature maps
# ---------------------------------------------------------------------------


def feature_maps(
    sv: torch.Tensor,
    acts: torch.Tensor,
    meta: list[dict],
    layer: int,
    topk: int,
    out_dir: Path,
    tc_snap_path: str,
) -> None:
    tc_path = Path(tc_snap_path) / f"layer_{layer}.safetensors"
    log.info("Loading W_enc from %s", tc_path)
    with safe_open(str(tc_path), framework="pt", device="cpu") as f:
        w_enc = f.get_tensor("W_enc").float()
        b_enc = f.get_tensor("b_enc").float()

    sv_f = sv.float().cpu()
    sv_scores = w_enc @ sv_f
    top_feats = sv_scores.topk(topk)

    log.info("Top-%d transcoder features aligned with sv:", topk)
    for val, idx in zip(top_feats.values, top_feats.indices, strict=False):
        log.info("  feat %6d  dot=%.3f", idx.item(), val.item())

    acts_cpu = acts.cpu()
    max_a = max(m["a"] for m in meta)
    max_b = max(m["b"] for m in meta)

    ncols = min(topk, 3)
    nrows = (topk + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for plot_i, (feat_val, feat_idx) in enumerate(
        zip(top_feats.values, top_feats.indices, strict=False)
    ):
        feat_idx = feat_idx.item()
        w = w_enc[feat_idx]
        b = b_enc[feat_idx].item()
        feat_acts = (acts_cpu @ w + b).numpy()

        grid = np.full((max_a + 1, max_b + 1), np.nan)
        for i, row in enumerate(meta):
            grid[row["a"], row["b"]] = feat_acts[i]

        ax = axes[plot_i]
        im = ax.imshow(
            grid,
            origin="lower",
            extent=[-0.5, max_b + 0.5, -0.5, max_a + 0.5],
            aspect="equal",
            interpolation="nearest",
            cmap="RdYlGn",
        )
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Feature {feat_idx}\ndot={feat_val:.2f}", fontsize=9)
        ax.set_xlabel("b", fontsize=8)
        ax.set_ylabel("a", fontsize=8)
        ax.set_xticks(np.arange(0, max_b + 1, 5))
        ax.set_yticks(np.arange(0, max_a + 1, 5))

    for ax in axes[topk:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Top-{topk} transcoder features aligned with steering vector (L={layer})", fontsize=11
    )
    fig.tight_layout()
    path = out_dir / f"steer_feature_maps_L{layer}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# 3. Improved output examples
# ---------------------------------------------------------------------------


def print_improved_examples(
    model: AttributionModel,
    samples: list[dict],
    sv: torch.Tensor,
    layer: int,
    alpha: float,
    eq_token_id: int,
    n_examples: int,
    device: torch.device,
    metric: str,
) -> None:
    metric_key = _metric_key(metric)
    log.info(
        "=== Improved examples by %s (wrong → correct after steering) ===", _metric_label(metric)
    )

    sv = sv.to(device)
    shown = 0
    model.eval()
    with torch.no_grad():
        for sample in samples:
            if shown >= n_examples:
                break

            token_ids = sample["prompt_token_ids"]
            eq_pos = _eq_pos(token_ids, eq_token_id)
            if eq_pos is None:
                continue

            before = _predict_sample(model, sample, device)
            if before[metric_key]:
                continue

            def _steer(act, hook, _sv=sv, _pos=eq_pos, _a=alpha):
                act = act.clone()
                act[0, _pos, :] += _a * _sv
                return act

            after = _predict_sample(
                model,
                sample,
                device,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post", _steer)],
            )
            if not after[metric_key]:
                continue

            print(
                f"  {sample['prompt_str'].strip()!r:30s}  "
                f"target={before['target_answer_str']!r:5s}  "
                f"before={before['generated_answer_str']!r:5s}  "
                f"after={after['generated_answer_str']!r:5s}  "
                f"first-token {int(before['first_token_correct'])}->{int(after['first_token_correct'])}  "
                f"full-answer {int(before['full_answer_correct'])}->{int(after['full_answer_correct'])}  "
                f"P(true first token) {before['true_first_token_prob']:.3f}->{after['true_first_token_prob']:.3f}"
            )
            shown += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--jsonl_path", "--dataset", dest="jsonl_path", default="data/addition_grid.jsonl"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--transcoder_set", default="mwhanna/qwen3-4b-transcoders")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--collect_n", type=int, default=300)
    parser.add_argument("--n_examples", type=int, default=15)
    parser.add_argument(
        "--bucket_metric",
        choices=STEERING_LABEL_METRICS,
        default=FIRST_TOKEN_METRIC,
        help="Correctness label used to build the steering vector.",
    )
    parser.add_argument("--out_dir", default="runs/knowledge_editing/plots")
    parser.add_argument(
        "--tc_snap_path",
        default=(
            "/local/eid23/hf/hub/models--mwhanna--qwen3-4b-transcoders"
            "/snapshots/94d176260ac39ce2f882b8b09aba8c118df29bb3"
        ),
        help="Local snapshot directory containing transcoder layer safetensors for feature maps.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.jsonl_path) as f:
        all_samples = [json.loads(line) for line in f]
    hard = [sample for sample in all_samples if _digit_count(sample) >= 2]
    collect_pool = hard[: len(hard) // 2]

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
    sv = recompute_sv(
        model,
        collect_pool,
        args.layer,
        eq_token_id,
        args.collect_n,
        device,
        dtype,
        args.bucket_metric,
    )

    logit_lens(sv, model, args.layer, args.topk * 2, out_dir)
    acts, meta = collect_residuals(model, all_samples, args.layer, eq_token_id, device, dtype)
    feature_maps(sv, acts, meta, args.layer, args.topk, out_dir, args.tc_snap_path)

    eval_samples = hard[len(hard) // 2 :]
    print_improved_examples(
        model,
        eval_samples,
        sv,
        args.layer,
        args.alpha,
        eq_token_id,
        args.n_examples,
        device,
        FULL_ANSWER_METRIC,
    )
    print_improved_examples(
        model,
        eval_samples,
        sv,
        args.layer,
        args.alpha,
        eq_token_id,
        args.n_examples,
        device,
        FIRST_TOKEN_METRIC,
    )


if __name__ == "__main__":
    main()
