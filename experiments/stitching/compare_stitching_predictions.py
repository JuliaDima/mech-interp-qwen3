#!/usr/bin/env python3
"""Compare large model predictions before and after stitching using teacher forcing.

This script:
1. Loads small model (QuantaMaths) and generates autoregressive predictions
2. Loads large model (Qwen3-4B) and evaluates with teacher forcing
3. Compares before/after stitching using teacher forcing on Template T0

Usage:
    python experiments/stitching/compare_stitching_predictions.py --n-examples 10
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add repo to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.addition.dataset_generation.generate_dataset_with_predictions import (  # noqa: E402
    TemplateID,
    build_prompt,
)
from experiments.stitching.run import (  # noqa: E402
    SmallAdditionTransformer,
    _qm_make_sample,
    _qm_tokenize,
    load_quanta_maths_model,
)
from mechinterp_qwen3.attribution_model import AttributionModel  # noqa: E402
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub  # noqa: E402
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype  # noqa: E402
from mechinterp_qwen3.utils_seed import seed_everything  # noqa: E402


def analyze_small_model_autoregressive(
    model: SmallAdditionTransformer,
    samples: list[dict],
    tokenize_fn,
    vocab: list[str],
) -> list[dict]:
    """Analyze small model using autoregressive generation."""
    model.model.eval()
    device = next(model.model.parameters()).device

    results = []
    for sample in tqdm(samples, desc="Small model (autoregressive)"):
        prompt = sample["prompt"]
        a, b = sample["a"], sample["b"]
        true_answer_numeric = a + b

        # Get expected answer format from sample
        full_text = sample.get("full", "")
        if "=" in full_text:
            true_answer_str = full_text.split("=")[1]
        else:
            true_answer_str = "+" + str(true_answer_numeric).zfill(6)

        # Autoregressive generation
        prompt_tokens = tokenize_fn(prompt)
        current_tokens = prompt_tokens[:]
        max_answer_tokens = len(true_answer_str)

        generated_tokens = []
        with torch.no_grad():
            for _step in range(max_answer_tokens):
                input_ids = torch.tensor([current_tokens], device=device, dtype=torch.long)
                logits = model.model(input_ids)
                next_token_logits = logits[0, -1, :]
                pred_token_id = int(next_token_logits.argmax())
                generated_tokens.append(vocab[pred_token_id] if pred_token_id < len(vocab) else "?")
                current_tokens.append(pred_token_id)

        predicted_answer = "".join(generated_tokens)
        correct = predicted_answer == true_answer_str

        results.append(
            {
                "a": a,
                "b": b,
                "sum": true_answer_numeric,
                "true_answer": true_answer_str,
                "predicted_answer": predicted_answer,
                "correct": correct,
            }
        )

    return results


def analyze_large_model_teacher_forcing(
    large_model: AttributionModel,
    samples: list[dict],
    template_id: TemplateID = TemplateID.T0,
) -> list[dict]:
    """Analyze large model using teacher forcing (no stitching)."""
    large_model.eval()

    results = []
    for sample in tqdm(samples, desc="Large model (teacher forcing - before)"):
        a, b = sample["a"], sample["b"]
        true_answer = str(a + b)

        # Build prompt using Template T0 by default (with trailing space)
        prompt = build_prompt(template_id, a, b)

        # Teacher forcing
        full_text = prompt + true_answer
        tokens = large_model.to_tokens(full_text)

        with torch.no_grad():
            logits = large_model(tokens)  # (1, seq_len, vocab_size)

        # Analyze first answer token (position after prompt)
        prompt_tokens = large_model.to_tokens(prompt)
        first_answer_pos = prompt_tokens.shape[1] - 1  # Position that predicts first answer token

        # Get prediction at first answer position
        first_answer_logits = logits[0, first_answer_pos, :]
        probs = F.softmax(first_answer_logits, dim=-1)

        # Top-5 predictions
        top5_probs, top5_ids = torch.topk(probs, min(5, probs.shape[0]))
        top5 = [
            {
                "token_id": int(tid),
                "token": large_model.tokenizer.decode([int(tid)]),
                "probability": float(prob),
            }
            for tid, prob in zip(top5_ids, top5_probs, strict=False)
        ]

        # True first token
        first_true_token_str = true_answer[0]
        first_true_token_ids = large_model.tokenizer.encode(
            first_true_token_str, add_special_tokens=False
        )
        first_true_token_id = first_true_token_ids[0] if first_true_token_ids else -1
        first_true_prob = float(probs[first_true_token_id]) if first_true_token_id >= 0 else 0.0

        # Predicted token
        pred_token_id = int(first_answer_logits.argmax())
        pred_token_str = large_model.tokenizer.decode([pred_token_id])
        pred_prob = float(probs[pred_token_id])

        results.append(
            {
                "a": a,
                "b": b,
                "sum": a + b,
                "prompt": prompt,
                "true_answer": true_answer,
                "first_true_token": first_true_token_str,
                "first_true_token_id": first_true_token_id,
                "first_true_probability": first_true_prob,
                "first_predicted_token": pred_token_str,
                "first_predicted_token_id": pred_token_id,
                "first_predicted_probability": pred_prob,
                "first_token_correct": pred_token_id in first_true_token_ids
                if first_true_token_ids
                else False,
                "top5_predictions": top5,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare predictions before/after stitching")
    parser.add_argument("--hub-model", default="PhilipQuirke/QuantaMaths_add_d5_l1_h3_t15K_s372001")
    parser.add_argument("--large-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--transcoder-set", default="Qwen2.5-3B-Instruct-ablation")
    parser.add_argument("--n-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-file", type=Path, default=None)

    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else get_default_device()
    dtype = parse_dtype(args.dtype)

    print(f"Device: {device}, dtype: {dtype}", file=sys.stderr)

    print(f"\n=== Loading small model: {args.hub_model} ===", file=sys.stderr)
    small_model, n_digits = load_quanta_maths_model(args.hub_model, device)
    vocab = [str(i) for i in range(10)] + ["+", "-", "=", "P", "M"]
    tokenize = _qm_tokenize

    import random

    random.seed(args.seed)
    max_val = 10**n_digits - 1
    samples = []
    for _ in range(args.n_examples):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        sample_dict = _qm_make_sample(a, b, n_digits)
        samples.append(sample_dict)

    print(f"Generated {len(samples)} test samples", file=sys.stderr)

    small_results = analyze_small_model_autoregressive(small_model, samples, tokenize, vocab)

    print(f"\n=== Loading large model: {args.large_model} ===", file=sys.stderr)
    print(f"Loading transcoders: {args.transcoder_set}", file=sys.stderr)

    transcoder, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=True
    )
    large_model = AttributionModel.from_pretrained_and_transcoders(
        args.large_model, transcoder, dtype=dtype, device=device
    )

    # Analyze large model (teacher forcing, no stitching)
    large_samples = [{"a": s["a"], "b": s["b"]} for s in samples]
    large_results = analyze_large_model_teacher_forcing(large_model, large_samples)

    output = {
        "config": {
            "small_model": args.hub_model,
            "large_model": args.large_model,
            "transcoder_set": args.transcoder_set,
            "n_examples": len(samples),
            "template": "T0 (calc: {a}+{b}= )",
            "seed": args.seed,
        },
        "small_model_results": {
            "method": "autoregressive_generation",
            "accuracy": sum(1 for r in small_results if r["correct"]) / len(small_results),
            "predictions": small_results,
        },
        "large_model_results": {
            "method": "teacher_forcing_first_token",
            "accuracy": sum(1 for r in large_results if r["first_token_correct"])
            / len(large_results),
            "predictions": large_results,
        },
    }

    print("\n=== SUMMARY ===", file=sys.stderr)
    print(
        f"Small model (autoregressive): {output['small_model_results']['accuracy']:.2%} accuracy",
        file=sys.stderr,
    )
    print(
        f"Large model (teacher forcing): {output['large_model_results']['accuracy']:.2%} first token accuracy",
        file=sys.stderr,
    )

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n=== JSON output written to {args.output_file} ===", file=sys.stderr)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
