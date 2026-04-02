#!/usr/bin/env python3
"""Analyze small addition model predictions on a grid of examples.

This script loads or trains a small addition model and outputs JSON with:
1. Input prompt example (tokenization)
2. 10 output token predictions with probabilities

Usage:
    python experiments/stitching/visualize_small_model_predictions.py --hub-model PhilipQuirke/QuantaMaths_add_d5_l1_h3_t15K_s372001
    python experiments/stitching/visualize_small_model_predictions.py --train-from-scratch
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

from experiments.stitching.run import (  # noqa: E402
    SmallAdditionTransformer,
    _qm_make_sample,
    _qm_tokenize,
    get_small_model_tokenizer,
    load_quanta_maths_model,
    train_small_model,
)
from mechinterp_qwen3.utils.model_utils import get_default_device  # noqa: E402
from mechinterp_qwen3.utils_seed import seed_everything  # noqa: E402


def analyze_tokenization(text: str, tokens: list[int], vocab: list[str]) -> dict:
    """Analyze how a prompt is tokenized."""
    return {
        "original_text": text,
        "tokens": [
            {
                "position": i,
                "token_id": token_id,
                "character": vocab[token_id] if token_id < len(vocab) else "?",
            }
            for i, token_id in enumerate(tokens)
        ],
        "total_tokens": len(tokens),
    }


def analyze_predictions(
    model: SmallAdditionTransformer,
    samples: list[dict],
    tokenize_fn,
    vocab: list[str],
    n_examples: int = 10,
) -> list[dict]:
    """Analyze model predictions for multiple examples using autoregressive generation."""
    model.model.eval()
    device = next(model.model.parameters()).device

    results = []
    n_examples = min(n_examples, len(samples))

    for sample in tqdm(samples[:n_examples], desc="Analyzing predictions"):
        # Get prompt and answer
        prompt = sample["prompt"]
        a, b = sample["a"], sample["b"]
        true_answer_numeric = a + b

        # For QuantaMaths: answer format is "+NNNNNN" (sign + zero-padded digits)
        # Extract expected format from sample if available
        full_text = sample.get("full", "")
        if "=" in full_text:
            true_answer_str = full_text.split("=")[1]
        else:
            # Fallback: assume answer is sign + digits
            true_answer_str = "+" + str(true_answer_numeric).zfill(6)

        # Tokenize prompt only
        prompt_tokens = tokenize_fn(prompt)
        max_answer_tokens = len(true_answer_str)

        # Autoregressive generation
        current_tokens = prompt_tokens[:]
        predictions = []

        with torch.no_grad():
            for step in range(max_answer_tokens):
                input_ids = torch.tensor([current_tokens], device=device, dtype=torch.long)
                logits = model.model(input_ids)  # (1, seq_len, vocab_size)

                # Get logits at the last position
                next_token_logits = logits[0, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)

                # Top prediction
                pred_token_id = int(next_token_logits.argmax())
                pred_prob = float(probs[pred_token_id])

                # True token at this position
                true_token_char = true_answer_str[step] if step < len(true_answer_str) else ""
                # Map character to token ID
                if true_token_char.isdigit():
                    true_token_id = int(true_token_char)
                elif true_token_char in ["+", "-", "=", "P", "M"]:
                    try:
                        true_token_id = vocab.index(true_token_char)
                    except ValueError:
                        true_token_id = -1
                else:
                    true_token_id = -1

                true_prob = float(probs[true_token_id]) if 0 <= true_token_id < len(vocab) else 0.0

                # Top-5 predictions
                top5_probs, top5_ids = torch.topk(probs, min(5, len(probs)))
                top5 = [
                    {
                        "token_id": int(tid),
                        "character": vocab[int(tid)] if int(tid) < len(vocab) else "?",
                        "probability": float(prob),
                    }
                    for tid, prob in zip(top5_ids, top5_probs, strict=False)
                ]

                predictions.append(
                    {
                        "position": step,
                        "true_token_id": true_token_id,
                        "true_character": true_token_char,
                        "true_probability": true_prob,
                        "predicted_token_id": pred_token_id,
                        "predicted_character": vocab[pred_token_id]
                        if pred_token_id < len(vocab)
                        else "?",
                        "predicted_probability": pred_prob,
                        "correct": pred_token_id == true_token_id,
                        "top5_predictions": top5,
                    }
                )

                # Append predicted token for next iteration
                current_tokens.append(pred_token_id)

        # Calculate accuracy
        n_correct = sum(1 for p in predictions if p["correct"])
        accuracy = n_correct / len(predictions) if predictions else 0.0

        predicted_answer = "".join(p["predicted_character"] for p in predictions)

        results.append(
            {
                "problem": {
                    "a": a,
                    "b": b,
                    "sum": true_answer_numeric,
                    "expression": f"{a}+{b}={true_answer_numeric}",
                },
                "prompt": prompt,
                "true_answer": true_answer_str,
                "predicted_answer": predicted_answer,
                "accuracy": accuracy,
                "correct_tokens": n_correct,
                "total_tokens": len(predictions),
                "predictions_by_position": predictions,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze small model predictions (JSON output)")
    parser.add_argument(
        "--hub-model",
        default="PhilipQuirke/QuantaMaths_add_d5_l1_h3_t15K_s372001",
        help="HuggingFace model ID (empty for training from scratch)",
    )
    parser.add_argument(
        "--train-from-scratch", action="store_true", help="Train model from scratch"
    )
    parser.add_argument("--num-digits", type=int, default=5, help="Number of digits for addition")
    parser.add_argument("--n-examples", type=int, default=10, help="Number of examples to analyze")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-file", type=Path, default=None, help="Output JSON file (default: stdout)"
    )

    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else get_default_device()

    # Load or train model
    if args.train_from_scratch or not args.hub_model:
        print("=== Training small model from scratch ===", file=sys.stderr)
        small_model, train_samples, _, _ = train_small_model(
            n_layers=2,
            n_heads=3,
            d_model=256,
            epochs=100,
            lr=1e-3,
            device=device,
            dtype=torch.float32,
            num_digits=args.num_digits,
            dry_run=False,
        )

        # Create vocab
        vocab = ["<PAD>", "<BOS>", "<EOS>"] + [str(i) for i in range(10)] + ["+", "=", " "]
        tokenize = get_small_model_tokenizer(small_model)

        # Generate samples
        import random

        random.seed(args.seed)
        max_val = 10**args.num_digits - 1
        samples = []
        for _ in range(args.n_examples):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            samples.append(
                {
                    "prompt": f"{a}+{b}=",
                    "full": f"{a}+{b}={a+b}",
                    "a": a,
                    "b": b,
                    "answer": str(a + b),
                }
            )

    else:
        print(f"=== Loading pretrained model: {args.hub_model} ===", file=sys.stderr)
        small_model, n_digits = load_quanta_maths_model(args.hub_model, device)

        # QuantaMaths vocab
        vocab = [str(i) for i in range(10)] + ["+", "-", "=", "P", "M"]
        tokenize = _qm_tokenize

        # Generate samples
        import random

        random.seed(args.seed)
        max_val = 10**n_digits - 1
        samples = []
        for _ in range(args.n_examples):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            sample_dict = _qm_make_sample(a, b, n_digits)
            samples.append(sample_dict)

    print(f"=== Generated {len(samples)} test samples ===", file=sys.stderr)

    first_sample = samples[0]
    prompt_text = first_sample["prompt"]
    prompt_tokens = tokenize(prompt_text)
    tokenization_analysis = analyze_tokenization(prompt_text, prompt_tokens, vocab)

    prediction_results = analyze_predictions(small_model, samples, tokenize, vocab, args.n_examples)

    output = {
        "model_info": {
            "hub_model": args.hub_model if not args.train_from_scratch else "trained_from_scratch",
            "num_digits": n_digits if not args.train_from_scratch else args.num_digits,
            "vocab_size": len(vocab),
            "vocab": vocab,
        },
        "tokenization_example": tokenization_analysis,
        "predictions": prediction_results,
        "summary": {
            "total_examples": len(prediction_results),
            "average_accuracy": sum(r["accuracy"] for r in prediction_results)
            / len(prediction_results),
            "perfect_predictions": sum(1 for r in prediction_results if r["accuracy"] == 1.0),
        },
    }

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"=== JSON output written to {args.output_file} ===", file=sys.stderr)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
