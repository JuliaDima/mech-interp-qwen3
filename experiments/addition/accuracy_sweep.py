"""Accuracy sweep for addition prompts before circuit discovery.

This module verifies that the model actually solves the addition task before
attempting circuit discovery. Following best practices:

1. Check tokenization of answers (single vs multi-token)
2. Run greedy decoding sweep over all calc: a+b= prompts
3. Filter to only prompts where model's argmax matches ground truth
4. Optionally filter by probability margin for high-confidence cases
5. Generate a "verified prompts" list for downstream analysis

If accuracy is low, this script helps identify which prompt formats work better.

Usage:
    python experiments/addition/accuracy_sweep.py --config experiments/addition/config.yaml
    python experiments/addition/accuracy_sweep.py --all  # Run all checks
    python experiments/addition/accuracy_sweep.py --all --quick  # Run all checks quickly
    python experiments/addition/accuracy_sweep.py --all --batch_size 64  # Run all checks quickly with batch size 64
    python experiments/addition/accuracy_sweep.py --tokenization  # Just check tokenization
    python experiments/addition/accuracy_sweep.py --sweep  # Run accuracy sweep only
    python experiments/addition/accuracy_sweep.py --filter --min_prob 0.7  # Filter high-confidence
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

# Ensure repo root is on sys.path so relative imports work when run as script
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.addition.expected_answers import EXPECTED_ANSWERS, get_target_tokens  # noqa: E402
from experiments.addition.prompts import CALC_GRID  # noqa: E402
from mechinterp_qwen3.utils.config_utils import (  # noqa: E402
    add_config_args,
    load_config,
    print_config,
    set_parser_defaults_from_config,
)
from mechinterp_qwen3.utils.inference_utils import tokenize_and_pad  # noqa: E402

if TYPE_CHECKING:
    from mechinterp_qwen3.attribution_model import AttributionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("addition.accuracy_sweep")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TokenizationInfo:
    """Information about how an answer tokenizes."""

    answer_str: str
    token_ids: list[int]
    token_strs: list[str]
    n_tokens: int
    is_single_token: bool


@dataclass
class AccuracyResult:
    """Result of accuracy check for a single prompt."""

    prompt: str
    answer_str: str
    predicted_str: str
    is_correct: bool
    target_token_id: int
    predicted_token_id: int
    target_prob: float
    predicted_prob: float
    margin: float  # predicted_prob - second_best_prob
    logit_true: float
    logit_predicted: float


# ---------------------------------------------------------------------------
# Phase 1: Tokenization verification
# ---------------------------------------------------------------------------


def check_tokenization(
    model: AttributionModel,
    out_dir: Path,
) -> dict[str, TokenizationInfo]:
    """Check how all answers tokenize.

    Args:
        model: AttributionModel with tokenizer
        out_dir: Output directory for JSON results

    Returns:
        Dict mapping answer_str → TokenizationInfo
    """
    log.info("Phase 1: Checking tokenization of all unique answers...")

    unique_answers = set(EXPECTED_ANSWERS.values())
    tokenizer = model.tokenizer

    tokenization_map: dict[str, TokenizationInfo] = {}

    for answer_str in sorted(unique_answers):
        token_ids: list[int] = tokenizer(answer_str, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]

        info = TokenizationInfo(
            answer_str=answer_str,
            token_ids=token_ids,
            token_strs=token_strs,
            n_tokens=len(token_ids),
            is_single_token=len(token_ids) == 1,
        )
        tokenization_map[answer_str] = info

    # Print summary
    single_token_count = sum(1 for info in tokenization_map.values() if info.is_single_token)
    multi_token_count = len(tokenization_map) - single_token_count

    log.info(
        "Tokenization summary: %d unique answers, %d single-token, %d multi-token",
        len(tokenization_map),
        single_token_count,
        multi_token_count,
    )

    # Show examples
    print("\n" + "=" * 60)
    print("TOKENIZATION EXAMPLES")
    print("=" * 60)

    for answer_str in sorted(tokenization_map.keys())[:10]:
        info = tokenization_map[answer_str]
        print(f"{answer_str:>4} → {info.token_strs}  (ids: {info.token_ids})")

    if multi_token_count > 0:
        print("\nMulti-token answers (first 10):")
        multi_token_answers = [
            (ans, info)
            for ans, info in sorted(tokenization_map.items())
            if not info.is_single_token
        ]
        for answer_str, info in multi_token_answers[:10]:
            print(f"  {answer_str:>4} → {info.token_strs}")

        if len(multi_token_answers) > 10:
            print(f"  ... and {len(multi_token_answers) - 10} more (see tokenization.json)")

    # Write to JSON
    out_path = out_dir / "tokenization.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                answer_str: {
                    "token_ids": info.token_ids,
                    "token_strs": info.token_strs,
                    "n_tokens": info.n_tokens,
                }
                for answer_str, info in tokenization_map.items()
            },
            f,
            indent=2,
        )
    log.info("Tokenization info written to %s", out_path)

    return tokenization_map


# ---------------------------------------------------------------------------
# Phase 2: Accuracy sweep
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_accuracy_sweep(
    model: AttributionModel,
    out_dir: Path,
    *,
    prompt_subset: list[str] | None = None,
    batch_size: int = 32,
) -> list[AccuracyResult]:
    """Run greedy decoding accuracy sweep with batched inference.

    Args:
        model: AttributionModel
        out_dir: Output directory for results
        prompt_subset: Optional list of specific prompts to test (default: all CALC_GRID)
        batch_size: Number of prompts to process in parallel (default: 32)

    Returns:
        List of AccuracyResult for each prompt
    """
    log.info("Phase 2: Running accuracy sweep with greedy decoding (batched)...")

    if prompt_subset is None:
        prompts_to_test = [entry["prompt"] for entry in CALC_GRID]
    else:
        prompts_to_test = prompt_subset

    log.info("Testing %d prompts with batch_size=%d", len(prompts_to_test), batch_size)

    results: list[AccuracyResult] = []

    # Process in batches for efficiency
    for batch_start in tqdm(
        range(0, len(prompts_to_test), batch_size),
        desc="Accuracy sweep (batches)",
        unit="batch",
    ):
        batch_prompts = prompts_to_test[batch_start : batch_start + batch_size]

        tokens, mask, lengths = tokenize_and_pad(model, batch_prompts)
        logits = model(tokens)  # (batch_size, seq_len, vocab_size)

        # Process each item in batch
        for i, prompt in enumerate(batch_prompts):
            answer_str = EXPECTED_ANSWERS[prompt]
            target_token_ids = get_target_tokens(prompt, model)
            target_token_id = target_token_ids[0]

            # Get last non-padding position
            length = lengths[i]
            last_logits = logits[i, length - 1, :]  # (vocab_size,)

            # Get predicted token (argmax)
            predicted_token_id = torch.argmax(last_logits).item()
            predicted_str = model.tokenizer.decode([predicted_token_id])

            # Compute probabilities
            probs = torch.softmax(last_logits, dim=-1)
            target_prob = probs[target_token_id].item()
            predicted_prob = probs[predicted_token_id].item()

            # Compute margin
            top2_probs, _ = torch.topk(probs, k=2)
            margin = top2_probs[0].item() - top2_probs[1].item()

            is_correct = predicted_token_id == target_token_id

            result = AccuracyResult(
                prompt=prompt,
                answer_str=answer_str,
                predicted_str=predicted_str,
                is_correct=is_correct,
                target_token_id=target_token_id,
                predicted_token_id=predicted_token_id,
                target_prob=target_prob,
                predicted_prob=predicted_prob,
                margin=margin,
                logit_true=last_logits[target_token_id].item(),
                logit_predicted=last_logits[predicted_token_id].item(),
            )
            results.append(result)

    # Compute statistics
    n_correct = sum(1 for r in results if r.is_correct)
    accuracy = n_correct / len(results) if results else 0.0

    log.info(
        "Accuracy sweep complete: %d/%d correct (%.2f%%)",
        n_correct,
        len(results),
        accuracy * 100,
    )

    # Print detailed statistics
    print("\n" + "=" * 60)
    print("ACCURACY SWEEP RESULTS")
    print("=" * 60)
    print(f"Total prompts tested: {len(results)}")
    print(f"Correct predictions:  {n_correct} ({accuracy:.2%})")
    print(f"Incorrect predictions: {len(results) - n_correct}")

    if n_correct > 0:
        correct_margins = [r.margin for r in results if r.is_correct]
        print("\nMargin statistics (correct cases):")
        print(f"  Mean:   {sum(correct_margins) / len(correct_margins):.4f}")
        print(f"  Median: {sorted(correct_margins)[len(correct_margins) // 2]:.4f}")
        print(f"  Min:    {min(correct_margins):.4f}")
        print(f"  Max:    {max(correct_margins):.4f}")

    # Show failure cases
    if n_correct < len(results):
        incorrect = [r for r in results if not r.is_correct]
        print(f"\nFailure analysis ({len(incorrect)} cases):")
        print("  Sample failures:")
        for r in incorrect[:10]:
            print(
                f"    {r.prompt:20s} → expected '{r.answer_str}', "
                f"got '{r.predicted_str}' (P={r.predicted_prob:.3f})"
            )

    # Write detailed results
    out_path = out_dir / "accuracy_sweep.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "summary": {
                    "n_prompts": len(results),
                    "n_correct": n_correct,
                    "accuracy": accuracy,
                },
                "results": [
                    {
                        "prompt": r.prompt,
                        "answer_str": r.answer_str,
                        "predicted_str": r.predicted_str,
                        "is_correct": r.is_correct,
                        "target_prob": round(r.target_prob, 6),
                        "predicted_prob": round(r.predicted_prob, 6),
                        "margin": round(r.margin, 6),
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    log.info("Detailed results written to %s", out_path)

    return results


# ---------------------------------------------------------------------------
# Phase 3: Filter to verified prompts
# ---------------------------------------------------------------------------


def filter_verified_prompts(
    results: list[AccuracyResult],
    out_dir: Path,
    *,
    min_prob: float = 0.0,
    min_margin: float = 0.0,
) -> list[str]:
    """Filter to prompts where model is correct and confident.

    Args:
        results: AccuracyResult list from sweep
        out_dir: Output directory
        min_prob: Minimum probability for target token (default: 0.0 = accept all correct)
        min_margin: Minimum margin between top-2 predictions (default: 0.0)

    Returns:
        List of verified prompt strings
    """
    log.info(
        "Phase 3: Filtering verified prompts (min_prob=%.2f, min_margin=%.2f)...",
        min_prob,
        min_margin,
    )

    verified = [
        r.prompt
        for r in results
        if r.is_correct and r.target_prob >= min_prob and r.margin >= min_margin
    ]

    log.info(
        "Filtered to %d/%d prompts (%.1f%% retention)",
        len(verified),
        len(results),
        100 * len(verified) / len(results) if results else 0,
    )

    # Show which prompts were filtered out
    filtered_out = [r for r in results if r.prompt not in verified]
    if filtered_out:
        print("\n" + "=" * 60)
        print("FILTERED OUT PROMPTS")
        print("=" * 60)
        print(f"Removed {len(filtered_out)} prompts:")

        reasons = Counter()
        for r in filtered_out:
            if not r.is_correct:
                reasons["incorrect_prediction"] += 1
            elif r.target_prob < min_prob:
                reasons["low_probability"] += 1
            elif r.margin < min_margin:
                reasons["low_margin"] += 1

        for reason, count in reasons.most_common():
            print(f"  {reason}: {count}")

    # Write verified list
    out_path = out_dir / "verified_prompts.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "min_prob": min_prob,
                "min_margin": min_margin,
                "n_verified": len(verified),
                "prompts": verified,
            },
            f,
            indent=2,
        )
    log.info("Verified prompts written to %s", out_path)

    # Also write a simple text list for easy use
    txt_path = out_dir / "verified_prompts.txt"
    with open(txt_path, "w") as f:
        for prompt in verified:
            f.write(prompt + "\n")
    log.info("Text list written to %s", txt_path)

    return verified


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Accuracy sweep for addition circuit discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Phases
    phases = p.add_argument_group("Phases")
    phases.add_argument("--all", action="store_true", help="Run all phases")
    phases.add_argument("--tokenization", action="store_true", help="Check tokenization only")
    phases.add_argument("--sweep", action="store_true", help="Run accuracy sweep only")
    phases.add_argument("--filter", action="store_true", help="Filter verified prompts only")
    phases.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test random 1000 prompts instead of all 10k",
    )

    # Model
    model_args = p.add_argument_group("Model")
    model_args.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model name")
    model_args.add_argument(
        "--transcoder_set",
        default="mwhanna/qwen3-4b-transcoders",
        help="HuggingFace transcoder set",
    )
    model_args.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "bfloat16", "float16"],
        help="Model dtype (defaults to config.yaml value)",
    )

    # Config file
    add_config_args(p)

    # Performance
    perf_args = p.add_argument_group("Performance")
    perf_args.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (higher = faster, but more memory)",
    )

    # Filtering
    filter_args = p.add_argument_group("Filtering")
    filter_args.add_argument(
        "--min_prob",
        type=float,
        default=0.5,
        help="Minimum probability for target token (0.0 = accept all correct)",
    )
    filter_args.add_argument(
        "--min_margin",
        type=float,
        default=0.0,
        help="Minimum margin between top-2 predictions",
    )

    # Output
    out_args = p.add_argument_group("Output")
    out_args.add_argument(
        "--out_dir",
        default="runs/addition/accuracy_sweep",
        help="Output directory",
    )

    return p


def main() -> None:
    parser = build_parser()

    pre, _ = parser.parse_known_args()
    config = load_config(pre.config)

    # Apply config defaults
    set_parser_defaults_from_config(parser, config)

    args = parser.parse_args()

    # Standardized configuration printing
    print_config(args, title="Effective Accuracy Sweep Configuration")

    # Determine which phases to run
    run_all = args.all
    do_tokenization = run_all or args.tokenization
    do_sweep = run_all or args.sweep
    do_filter = run_all or args.filter

    if not any([do_tokenization, do_sweep, do_filter]):
        parser.print_help()
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # Load model (only if needed)
    model = None
    if do_tokenization or do_sweep:
        from mechinterp_qwen3.attribution_model import AttributionModel
        from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub

        # Use bfloat16 as default if none specified (to save RAM)
        dtype_str = args.dtype or config.get("dtype") or "bfloat16"
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        dtype = dtype_map[dtype_str]

        log.info("Loading transcoder (dtype=%s)...", dtype_str)
        transcoder, config_sae = load_transcoder_from_hub(
            args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=True
        )

        log.info("Loading model %s (dtype=%s)...", args.model, dtype_str)
        model = AttributionModel.from_pretrained_and_transcoders(
            args.model,
            transcoder,
            dtype=dtype,
            low_cpu_mem_usage=True,  # Critical for CPU-only nodes
        )

    # Run phases
    if do_tokenization:
        check_tokenization(model, out_dir)

    results = None
    if do_sweep:
        # Quick mode: sample random subset
        prompt_subset = None
        if args.quick:
            import random

            all_prompts = [entry["prompt"] for entry in CALC_GRID]
            random.seed(42)
            prompt_subset = random.sample(all_prompts, min(1000, len(all_prompts)))
            log.info("Quick mode: testing %d random prompts", len(prompt_subset))

        results = run_accuracy_sweep(
            model, out_dir, prompt_subset=prompt_subset, batch_size=args.batch_size
        )

    if do_filter:
        # Load results if not already computed
        if results is None:
            results_path = out_dir / "accuracy_sweep.json"
            if not results_path.exists():
                log.error(
                    "No sweep results found. Run --sweep first or use --all to run all phases."
                )
                return
            with open(results_path) as f:
                data = json.load(f)
                results = [
                    AccuracyResult(
                        prompt=r["prompt"],
                        answer_str=r["answer_str"],
                        predicted_str=r["predicted_str"],
                        is_correct=r["is_correct"],
                        target_token_id=0,  # Not saved in JSON
                        predicted_token_id=0,
                        target_prob=r["target_prob"],
                        predicted_prob=r["predicted_prob"],
                        margin=r["margin"],
                        logit_true=0.0,
                        logit_predicted=0.0,
                    )
                    for r in data["results"]
                ]

        filter_verified_prompts(
            results, out_dir, min_prob=args.min_prob, min_margin=args.min_margin
        )

    print(f"\n{'=' * 60}")
    print(f"All phases complete. Results in: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
