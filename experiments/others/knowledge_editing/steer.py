"""Steering vectors for addition prompts.

Strategy
--------
1. Load a dataset of prompts and gold answer-token IDs.
2. Filter to harder samples (max digits >= --min_digits).
3. Run the model live to measure both first-token correctness and full-answer
   correctness.
4. Capture h[eq_pos] at each sweep layer and bucket activations by the chosen
   steering-label metric (--bucket_metric).
5. steering_vec[L] = mean(correct_acts[L]) - mean(incorrect_acts[L])
6. Eval: inject alpha * sv at eq_pos, then report both correctness metrics plus
   the mean probability assigned to the true first answer token.

Usage
-----
    python experiments/knowledge_editing/steer.py
    python experiments/knowledge_editing/steer.py --min_digits 5 --layers 20 24 28 32
    python experiments/knowledge_editing/steer.py --alphas 0.5 1 2 5 --collect_n 400
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from scripts.model_config import default_model, default_transcoder_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("steer")

_LARGE_MODEL = default_model()
_TRANSCODER_SET = default_transcoder_set()
_DATASET = "data/addition_3digit.jsonl"
_SWEEP_LAYERS = [16, 20, 24, 28, 32]
_SWEEP_ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0]
_MIN_DIGITS = 3
_COLLECT_N = 300
_EVAL_N = 400

FIRST_TOKEN_METRIC = "first_token"
FULL_ANSWER_METRIC = "full_answer"
STEERING_LABEL_METRICS = (FIRST_TOKEN_METRIC, FULL_ANSWER_METRIC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eq_pos(token_ids: list[int], eq_token_id: int) -> int | None:
    """Return last position of the '=' token in the sequence."""
    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i] == eq_token_id:
            return i
    return None


def _digit_count(sample: dict) -> int:
    return max(len(str(sample["a"])), len(str(sample["b"])))


def _metric_key(metric: str) -> str:
    return f"{metric}_correct"


def _metric_label(metric: str) -> str:
    if metric == FIRST_TOKEN_METRIC:
        return "first-token correctness"
    if metric == FULL_ANSWER_METRIC:
        return "full-answer correctness"
    raise ValueError(f"Unknown metric: {metric}")


def _run_model(
    model: AttributionModel,
    input_ids: torch.Tensor,
    fwd_hooks: list[tuple[str, Callable]] | None = None,
) -> torch.Tensor:
    if fwd_hooks:
        return model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)
    return model(input_ids)


def _predict_sample(
    model: AttributionModel,
    sample: dict,
    device: torch.device,
    fwd_hooks: list[tuple[str, Callable]] | None = None,
) -> dict:
    """Run greedy decoding live and return both correctness definitions."""
    prompt_token_ids: list[int] = sample["prompt_token_ids"]
    target_token_ids: list[int] = sample["answer_token_ids"]
    generated_token_ids: list[int] = []

    true_first_token_prob = 0.0
    true_first_token_logp = float("-inf")

    for pos, target_id in enumerate(target_token_ids):
        input_ids = torch.tensor(
            [prompt_token_ids + generated_token_ids], dtype=torch.long, device=device
        )
        logits = _run_model(model, input_ids, fwd_hooks=fwd_hooks)[0, -1, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        pred_id = int(logits.argmax())

        if pos == 0:
            true_first_token_logp = log_probs[target_id].item()
            true_first_token_prob = log_probs[target_id].exp().item()

        generated_token_ids.append(pred_id)

    first_token_pred_id = generated_token_ids[0]
    generated_answer_str = model.tokenizer.decode(generated_token_ids).strip()
    target_answer_str = model.tokenizer.decode(target_token_ids).strip()

    return {
        "first_token_correct": first_token_pred_id == target_token_ids[0],
        "full_answer_correct": generated_token_ids == target_token_ids,
        "true_first_token_prob": true_first_token_prob,
        "true_first_token_logp": true_first_token_logp,
        "first_token_pred_id": first_token_pred_id,
        "generated_answer_ids": generated_token_ids,
        "generated_answer_str": generated_answer_str,
        "target_answer_str": target_answer_str,
    }


# ---------------------------------------------------------------------------
# Live baseline — run model once on all samples, record predictions
# ---------------------------------------------------------------------------


def precompute_baseline(
    model: AttributionModel,
    samples: list[dict],
    device: torch.device,
) -> list[dict]:
    """Run the model live on every sample; return per-sample prediction info."""
    results = []
    model.eval()
    with torch.no_grad():
        for sample in tqdm(samples, desc="Baseline pass"):
            results.append(_predict_sample(model, sample, device))
    return results


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect_steering_vecs(
    model: AttributionModel,
    samples: list[dict],
    baseline: list[dict],
    layers: list[int],
    eq_token_id: int,
    collect_n: int,
    device: torch.device,
    dtype: torch.dtype,
    bucket_metric: str = FIRST_TOKEN_METRIC,
) -> dict[int, torch.Tensor]:
    """Return {layer: steering_vec} for each sweep layer."""

    acts: dict[int, dict[str, list[torch.Tensor]]] = {
        layer: {"correct": [], "wrong": []} for layer in layers
    }
    needed = collect_n
    bucket_key = _metric_key(bucket_metric)

    model.eval()
    with torch.no_grad():
        for sample, base in tqdm(
            zip(samples, baseline, strict=False), desc="Collecting activations", total=len(samples)
        ):
            correct = bool(base[bucket_key])
            bucket = "correct" if correct else "wrong"

            if all(len(acts[layer][bucket]) >= needed for layer in layers):
                continue

            token_ids: list[int] = sample["prompt_token_ids"]
            eq_pos = _eq_pos(token_ids, eq_token_id)
            if eq_pos is None:
                continue

            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            cache: dict[int, torch.Tensor] = {}

            hooks = [
                (
                    f"blocks.{layer}.hook_resid_post",
                    lambda act, hook, _layer=layer, _pos=eq_pos: (
                        cache.update({_layer: act[0, _pos, :].detach().clone()}) or act
                    ),
                )
                for layer in layers
                if len(acts[layer][bucket]) < needed
            ]
            if not hooks:
                continue

            model.run_with_hooks(input_ids, fwd_hooks=hooks)

            for layer in layers:
                if layer in cache and len(acts[layer][bucket]) < needed:
                    acts[layer][bucket].append(cache[layer].to(dtype=dtype))

            if all(
                len(acts[layer]["correct"]) >= needed and len(acts[layer]["wrong"]) >= needed
                for layer in layers
            ):
                break

    svecs: dict[int, torch.Tensor] = {}
    for layer in layers:
        correct_acts = acts[layer]["correct"]
        wrong_acts = acts[layer]["wrong"]
        if not correct_acts or not wrong_acts:
            log.warning(
                "Layer %d: insufficient %s data (correct=%d, wrong=%d)",
                layer,
                _metric_label(bucket_metric),
                len(correct_acts),
                len(wrong_acts),
            )
            continue

        n = min(len(correct_acts), len(wrong_acts))
        sv = torch.stack(correct_acts[:n]).mean(0) - torch.stack(wrong_acts[:n]).mean(0)
        svecs[layer] = sv

        if len(correct_acts) < needed or len(wrong_acts) < needed:
            log.warning(
                "Layer %2d  using %d/%d requested samples per %s bucket",
                layer,
                n,
                needed,
                bucket_metric,
            )

        log.info(
            "Layer %2d  sv_norm=%.3f  (%s; from %d correct / %d wrong samples)",
            layer,
            sv.norm().item(),
            _metric_label(bucket_metric),
            len(correct_acts),
            len(wrong_acts),
        )
    return svecs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: AttributionModel,
    samples: list[dict],
    baseline: list[dict],
    eq_token_id: int,
    svecs: dict[int, torch.Tensor],
    alphas: list[float],
    device: torch.device,
    bucket_metric: str,
) -> list[dict]:
    """Evaluate each (layer, alpha) combination. Returns result dicts."""

    results = []
    model.eval()

    first_token_acc_base = 100.0 * sum(b["first_token_correct"] for b in baseline) / len(baseline)
    full_answer_acc_base = 100.0 * sum(b["full_answer_correct"] for b in baseline) / len(baseline)
    mean_true_first_token_prob_base = sum(b["true_first_token_prob"] for b in baseline) / len(
        baseline
    )
    geom_mean_true_first_token_prob_base = float(
        torch.tensor([b["true_first_token_logp"] for b in baseline]).mean().exp()
    )
    log.info(
        "Baseline (live, SV buckets=%s)  first-token acc=%.2f%%  full-answer acc=%.2f%%  "
        "mean P(true first token)=%.4f  geo-mean P(true first token)=%.4f",
        _metric_label(bucket_metric),
        first_token_acc_base,
        full_answer_acc_base,
        mean_true_first_token_prob_base,
        geom_mean_true_first_token_prob_base,
    )

    with torch.no_grad():
        for layer, sv in svecs.items():
            sv = sv.to(device)
            for alpha in alphas:
                n_eval = 0
                n_first_token_correct = 0
                n_full_answer_correct = 0
                sum_true_first_token_prob = 0.0
                sum_true_first_token_logp = 0.0

                for sample in tqdm(samples, desc=f"Eval L={layer} α={alpha}", leave=False):
                    token_ids: list[int] = sample["prompt_token_ids"]
                    eq_pos = _eq_pos(token_ids, eq_token_id)
                    if eq_pos is None:
                        continue

                    def _steer(act, hook, _sv=sv, _pos=eq_pos, _a=alpha):
                        act = act.clone()
                        act[0, _pos, :] = act[0, _pos, :] + _a * _sv
                        return act

                    pred = _predict_sample(
                        model,
                        sample,
                        device,
                        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", _steer)],
                    )

                    n_eval += 1
                    n_first_token_correct += int(pred["first_token_correct"])
                    n_full_answer_correct += int(pred["full_answer_correct"])
                    sum_true_first_token_prob += pred["true_first_token_prob"]
                    sum_true_first_token_logp += pred["true_first_token_logp"]

                first_token_acc = 100.0 * n_first_token_correct / n_eval
                full_answer_acc = 100.0 * n_full_answer_correct / n_eval
                mean_true_first_token_prob = sum_true_first_token_prob / n_eval
                geom_mean_true_first_token_prob = float(
                    torch.tensor(sum_true_first_token_logp / n_eval).exp()
                )
                delta_first_token_acc = first_token_acc - first_token_acc_base
                delta_full_answer_acc = full_answer_acc - full_answer_acc_base
                delta_mean_true_first_token_prob = (
                    mean_true_first_token_prob - mean_true_first_token_prob_base
                )
                delta_geom_mean_true_first_token_prob = (
                    geom_mean_true_first_token_prob - geom_mean_true_first_token_prob_base
                )

                log.info(
                    "L=%2d  α=%5.1f  first-token acc=%5.2f%%  (%+.2f%%)  "
                    "full-answer acc=%5.2f%%  (%+.2f%%)  "
                    "mean P(true first token)=%.4f  (%+.4f)",
                    layer,
                    alpha,
                    first_token_acc,
                    delta_first_token_acc,
                    full_answer_acc,
                    delta_full_answer_acc,
                    mean_true_first_token_prob,
                    delta_mean_true_first_token_prob,
                )
                results.append(
                    {
                        "bucket_metric": bucket_metric,
                        "layer": layer,
                        "alpha": alpha,
                        "n_eval": n_eval,
                        "first_token_acc": first_token_acc,
                        "delta_first_token_acc": delta_first_token_acc,
                        "full_answer_acc": full_answer_acc,
                        "delta_full_answer_acc": delta_full_answer_acc,
                        "mean_true_first_token_prob": mean_true_first_token_prob,
                        "delta_mean_true_first_token_prob": delta_mean_true_first_token_prob,
                        "geom_mean_true_first_token_prob": geom_mean_true_first_token_prob,
                        "delta_geom_mean_true_first_token_prob": (
                            delta_geom_mean_true_first_token_prob
                        ),
                        # Backward-compatible aliases for older readers.
                        "acc": first_token_acc,
                        "delta_acc": delta_first_token_acc,
                        "mean_p": geom_mean_true_first_token_prob,
                        "delta_p": delta_geom_mean_true_first_token_prob,
                    }
                )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default=_DATASET)
    parser.add_argument("--model", default=_LARGE_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--min_digits",
        type=int,
        default=_MIN_DIGITS,
        help="Filter to samples where max(digits(a), digits(b)) >= N",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=_SWEEP_LAYERS)
    parser.add_argument("--alphas", type=float, nargs="+", default=_SWEEP_ALPHAS)
    parser.add_argument(
        "--collect_n",
        type=int,
        default=_COLLECT_N,
        help="Samples per bucket (correct/wrong) for sv estimation",
    )
    parser.add_argument("--eval_n", type=int, default=_EVAL_N)
    parser.add_argument(
        "--bucket_metric",
        choices=STEERING_LABEL_METRICS,
        default=FIRST_TOKEN_METRIC,
        help="Correctness label used to split activations into steering buckets.",
    )
    parser.add_argument("--out", default="runs/knowledge_editing/steer_results.json")
    args = parser.parse_args()

    device = get_default_device()
    dtype = parse_dtype(args.dtype)
    log.info("Device: %s  dtype: %s", device, dtype)

    with open(args.dataset) as f:
        all_samples = [json.loads(line) for line in f]

    hard = [sample for sample in all_samples if _digit_count(sample) >= args.min_digits]
    log.info(
        "Hard samples (>= %d digits): %d / %d",
        args.min_digits,
        len(hard),
        len(all_samples),
    )

    collect_pool = hard[: len(hard) // 2]
    eval_samples = hard[len(hard) // 2 :][: args.eval_n]

    log.info("Loading model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()

    eq_token_id: int = model.tokenizer("=", add_special_tokens=False).input_ids[-1]
    log.info("'=' token id: %d", eq_token_id)

    log.info("Computing live baseline on collect + eval pools...")
    collect_baseline = precompute_baseline(model, collect_pool, device)
    eval_baseline = precompute_baseline(model, eval_samples, device)

    collect_first_token_acc = (
        100
        * sum(b["first_token_correct"] for b in collect_baseline)
        / max(len(collect_baseline), 1)
    )
    collect_full_answer_acc = (
        100
        * sum(b["full_answer_correct"] for b in collect_baseline)
        / max(len(collect_baseline), 1)
    )
    eval_first_token_acc = (
        100 * sum(b["first_token_correct"] for b in eval_baseline) / max(len(eval_baseline), 1)
    )
    eval_full_answer_acc = (
        100 * sum(b["full_answer_correct"] for b in eval_baseline) / max(len(eval_baseline), 1)
    )
    log.info(
        "Live baseline — collect: first-token %.1f%%, full-answer %.1f%%  eval: first-token %.1f%%, full-answer %.1f%%",
        collect_first_token_acc,
        collect_full_answer_acc,
        eval_first_token_acc,
        eval_full_answer_acc,
    )

    svecs = collect_steering_vecs(
        model,
        collect_pool,
        collect_baseline,
        args.layers,
        eq_token_id,
        args.collect_n,
        device,
        dtype,
        args.bucket_metric,
    )

    if not svecs:
        log.error("No steering vectors computed — check dataset, bucket metric, and digit filter")
        return

    results = evaluate(
        model,
        eval_samples,
        eval_baseline,
        eq_token_id,
        svecs,
        args.alphas,
        device,
        args.bucket_metric,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved to %s", args.out)

    best_full = max(results, key=lambda row: row["delta_full_answer_acc"])
    best_first = max(results, key=lambda row: row["delta_first_token_acc"])
    best_prob = max(results, key=lambda row: row["delta_mean_true_first_token_prob"])
    log.info(
        "Best by Δfull-answer acc: layer=%d  α=%.1f  Δ=%+.2f%%",
        best_full["layer"],
        best_full["alpha"],
        best_full["delta_full_answer_acc"],
    )
    log.info(
        "Best by Δfirst-token acc: layer=%d  α=%.1f  Δ=%+.2f%%",
        best_first["layer"],
        best_first["alpha"],
        best_first["delta_first_token_acc"],
    )
    log.info(
        "Best by Δmean P(true first token): layer=%d  α=%.1f  Δ=%+.4f",
        best_prob["layer"],
        best_prob["alpha"],
        best_prob["delta_mean_true_first_token_prob"],
    )


if __name__ == "__main__":
    main()
