"""Diagnostic: isolate whether transcoder accuracy improvement comes from features or structural terms.

Three conditions at the specified layers (default: 13, 14, 16, 21 — the carry-feature layers):
  1. raw_model          — plain model forward, no transcoder
  2. full_tc            — full transcoder reconstruction (all features active)
  3. bias_skip_only     — all features zeroed: output = b_dec + h_in @ W_skip.T only

If accuracy under condition 3 matches condition 2, the improvement is structural (biases/skip).
If accuracy under condition 3 matches condition 1, the improvement requires feature activations.

Usage:
    python -m experiments.concept_localization.pipeline.run_bias_skip_diagnostic \\
        --concept carry --layers 13 14 16 21 --sample_per_class 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.pipeline.run_concept import (
    CONCEPTS,
    _MODEL,
    _TRANSCODER_SET,
    _load_concept,
)
from experiments.concept_localization.pipeline.run_feature_ablation import (
    EvalMetrics,
    SplitMetrics,
    select_pairs,
)
from mechinterp_qwen3.interventions import inhibit_features, make_capture_hook
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bias_skip_diagnostic")


def _split_metrics(rows: list[dict], split: str) -> SplitMetrics:
    items = rows if split == "all" else [r for r in rows if r["split"] == split]
    if not items:
        return SplitMetrics(n=0, accuracy=0.0, mean_correct_prob=0.0)
    return SplitMetrics(
        n=len(items),
        accuracy=sum(1.0 for r in items if r["correct"]) / len(items),
        mean_correct_prob=sum(float(r["correct_prob"]) for r in items) / len(items),
    )


def _metrics(rows: list[dict], skipped: int) -> EvalMetrics:
    return EvalMetrics(
        all=_split_metrics(rows, "all"),
        pos=_split_metrics(rows, "pos"),
        neg=_split_metrics(rows, "neg"),
        skipped=skipped,
    )


@torch.no_grad()
def evaluate_condition(
    model,
    pairs: list,
    feature_map: dict[int, list[int]] | None,
    batch_size: int,
    desc: str,
) -> EvalMetrics:
    rows: list[dict] = []
    skipped = 0

    examples_by_len: dict[int, list] = {}
    for pair in pairs:
        pred_pos = pair.predict_pos if pair.predict_pos else pair.label_pos
        pred_neg = pair.predict_neg if pair.predict_neg else pair.label_neg
        for split, prompt, answer_str in [
            ("pos", pair.prompt_pos, pred_pos),
            ("neg", pair.prompt_neg, pred_neg),
        ]:
            prompt_ids = model.tokenizer(prompt, add_special_tokens=False).input_ids
            answer_ids = model.tokenizer(answer_str, add_special_tokens=False).input_ids
            if not answer_ids:
                skipped += 1
                continue
            examples_by_len.setdefault(len(prompt_ids) + len(answer_ids), []).append(
                (split, prompt_ids, answer_ids)
            )

    total = sum(len(v) for v in examples_by_len.values())
    with tqdm(total=total, desc=desc) as pbar:
        for items in examples_by_len.values():
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                tokens = torch.stack(
                    [
                        tokenize_qwen_input(
                            prompt_ids + answer_ids, model.tokenizer, model.cfg.device
                        )
                        for _, prompt_ids, answer_ids in batch
                    ],
                    dim=0,
                )
                if feature_map is None:
                    logits = model(tokens)
                else:
                    logits = inhibit_features(model, tokens, feature_map, alpha=0.0)

                for i, (split, prompt_ids, answer_ids) in enumerate(batch):
                    n = len(prompt_ids)
                    correct_prob = torch.softmax(logits[i, n], dim=-1)[answer_ids[0]].item()
                    all_correct = all(
                        int(logits[i, n + j].argmax()) == tok_id
                        for j, tok_id in enumerate(answer_ids)
                    )
                    rows.append({"split": split, "correct": all_correct, "correct_prob": correct_prob})
                pbar.update(len(batch))

    return _metrics(rows, skipped)


def _make_bias_skip_hook(model, layer: int) -> tuple[str, object]:
    """Hook that replaces MLP output with b_dec + h_in @ W_skip.T (all features zeroed)."""
    transcoder = model.transcoders[layer]

    def _hook(mlp_out: torch.Tensor, hook) -> torch.Tensor:
        h_in = _hook._last_mlp_in
        if h_in is None:
            return mlp_out
        with torch.no_grad():
            b_dec = transcoder.b_dec
            reconstructed = b_dec.expand_as(mlp_out.to(b_dec.dtype))
            if transcoder.W_skip is not None:
                reconstructed = reconstructed + h_in.to(reconstructed.dtype) @ transcoder.W_skip.T
        return reconstructed.to(mlp_out.dtype)

    _hook._last_mlp_in = None
    hook_name = f"blocks.{layer}.{model.original_feature_output_hook}"
    return hook_name, _hook


@torch.no_grad()
def evaluate_bias_skip_only(
    model,
    pairs: list,
    layers: list[int],
    batch_size: int,
) -> EvalMetrics:
    """Run forward pass with all feature activations zeroed; only b_dec + W_skip remain."""
    rows: list[dict] = []
    skipped = 0

    examples_by_len: dict[int, list] = {}
    for pair in pairs:
        pred_pos = pair.predict_pos if pair.predict_pos else pair.label_pos
        pred_neg = pair.predict_neg if pair.predict_neg else pair.label_neg
        for split, prompt, answer_str in [
            ("pos", pair.prompt_pos, pred_pos),
            ("neg", pair.prompt_neg, pred_neg),
        ]:
            prompt_ids = model.tokenizer(prompt, add_special_tokens=False).input_ids
            answer_ids = model.tokenizer(answer_str, add_special_tokens=False).input_ids
            if not answer_ids:
                skipped += 1
                continue
            examples_by_len.setdefault(len(prompt_ids) + len(answer_ids), []).append(
                (split, prompt_ids, answer_ids)
            )

    # Build hooks once
    hooks: list[tuple[str, object]] = []
    for layer in layers:
        out_name, inhibit_fn = _make_bias_skip_hook(model, layer)
        cap_name, cap_fn = make_capture_hook(model, layer, inhibit_fn)
        hooks.append((cap_name, cap_fn))
        hooks.append((out_name, inhibit_fn))

    total = sum(len(v) for v in examples_by_len.values())
    with tqdm(total=total, desc="3_bias_skip_only") as pbar:
        for items in examples_by_len.values():
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                tokens = torch.stack(
                    [
                        tokenize_qwen_input(
                            prompt_ids + answer_ids, model.tokenizer, model.cfg.device
                        )
                        for _, prompt_ids, answer_ids in batch
                    ],
                    dim=0,
                )
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                for i, (split, prompt_ids, answer_ids) in enumerate(batch):
                    n = len(prompt_ids)
                    correct_prob = torch.softmax(logits[i, n], dim=-1)[answer_ids[0]].item()
                    all_correct = all(
                        int(logits[i, n + j].argmax()) == tok_id
                        for j, tok_id in enumerate(answer_ids)
                    )
                    rows.append({"split": split, "correct": all_correct, "correct_prob": correct_prob})
                pbar.update(len(batch))

    return _metrics(rows, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", required=True, choices=CONCEPTS)
    parser.add_argument("--layers", nargs="+", type=int, default=[13, 14, 16, 21],
                        help="Layers at which to apply/zero transcoder")
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--sample_per_class", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    device = get_default_device()
    dtype = parse_dtype(args.dtype)
    log.info("Loading model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    from mechinterp_qwen3.attribution_model import AttributionModel
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()

    layers = args.layers
    log.info("Target layers: %s", layers)

    # Build feature maps
    full_tc_map: dict[int, list[int]] = {L: [] for L in layers}
    for L in layers:
        n_features = model.transcoders[L].W_enc.shape[0]
        log.info("Layer %d: %d features", L, n_features)

    log.info("Generating %d pairs/template for concept '%s'", args.n, args.concept)
    all_pairs = _load_concept(args.concept, args.n, args.seed)
    if args.template:
        all_pairs = [p for p in all_pairs if p.template == args.template]
        log.info("Filtered to template '%s': %d pairs", args.template, len(all_pairs))

    pairs = select_pairs(all_pairs, args.sample_per_class, args.seed)
    log.info("Evaluating %d pairs (%d pos + %d neg)", len(pairs), len(pairs), len(pairs))

    raw = evaluate_condition(model, pairs, None, args.batch_size, "1_raw_model")
    full_tc = evaluate_condition(model, pairs, full_tc_map, args.batch_size, "2_full_tc")
    bias_skip = evaluate_bias_skip_only(model, pairs, layers, args.batch_size)

    results = {
        "config": {
            "concept": args.concept,
            "layers": layers,
            "sample_per_class": args.sample_per_class,
            "seed": args.seed,
            "template": args.template,
        },
        "conditions": {
            "raw_model": {
                "accuracy": raw.all.accuracy,
                "mean_correct_prob": raw.all.mean_correct_prob,
                "pos_accuracy": raw.pos.accuracy,
                "neg_accuracy": raw.neg.accuracy,
            },
            "full_tc": {
                "accuracy": full_tc.all.accuracy,
                "mean_correct_prob": full_tc.all.mean_correct_prob,
                "pos_accuracy": full_tc.pos.accuracy,
                "neg_accuracy": full_tc.neg.accuracy,
            },
            "bias_skip_only": {
                "accuracy": bias_skip.all.accuracy,
                "mean_correct_prob": bias_skip.all.mean_correct_prob,
                "pos_accuracy": bias_skip.pos.accuracy,
                "neg_accuracy": bias_skip.neg.accuracy,
            },
        },
    }

    out_dir = Path(args.out_dir or f"runs/concept_localization/{args.concept}/bias_skip_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Saved → %s", out_path)

    # Print summary
    print("\n=== Bias+Skip Diagnostic ===")
    print(f"Layers: {layers}")
    print(f"{'Condition':<20} {'Accuracy':>10} {'P(correct)':>12} {'Pos acc':>10} {'Neg acc':>10}")
    print("-" * 64)
    for name, cond in results["conditions"].items():
        print(
            f"{name:<20} {cond['accuracy']:>10.1%} {cond['mean_correct_prob']:>12.4f}"
            f" {cond['pos_accuracy']:>10.1%} {cond['neg_accuracy']:>10.1%}"
        )


if __name__ == "__main__":
    main()
