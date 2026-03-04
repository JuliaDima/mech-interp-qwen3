#!/usr/bin/env python3
"""

Example usage:
    # Grid sampling with all templates
    python -m mechinterp_qwen3.dataset_generation \
        --model_name Qwen/Qwen2.5-3B-Instruct \
        --output_path data/addition_dataset.jsonl \
        --sampling_strategy grid \
        --max_value 20 \
        --templates T0 T1 T2

    # Stratified sampling with greedy generation
    python -m mechinterp_qwen3.dataset_generation \
        --model_name Qwen/Qwen2.5-3B-Instruct \
        --output_path data/addition_stratified.jsonl \
        --sampling_strategy stratified \
        --max_value 100 \
        --templates T0 \
        --stratified_n_per_category 50 \
        --stratified_uniform_remainder 100 \
        --enable_greedy_generation \
        --seed 42

    # Random sampling
    python -m mechinterp_qwen3.dataset_generation \
        --model_name Qwen/Qwen2.5-3B-Instruct \
        --output_path data/addition_random.jsonl \
        --sampling_strategy random \
        --max_value 1000 \
        --n_samples 500 \
        --templates T1
"""

import argparse
from pathlib import Path

from ..utils.config_utils import print_config
from .generate_add_dataset import (
    DatasetConfig,
    SamplingStrategy,
    TemplateID,
    generate_dataset,
    write_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate addition dataset with baseline model statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If not specified, uses CUDA if available.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: float32)",
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to output JSONL file",
    )

    parser.add_argument(
        "--templates",
        nargs="+",
        type=str,
        default=["T0"],
        choices=["T0", "T1", "T2"],
        help="Templates to use (default: T0). Can specify multiple.",
    )

    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="grid",
        choices=["grid", "stratified", "random"],
        help="Sampling strategy for (a,b) pairs (default: grid)",
    )
    parser.add_argument(
        "--max_value",
        type=int,
        default=20,
        help="Maximum value for a and b (default: 20)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples (only for random strategy)",
    )
    parser.add_argument(
        "--stratified_n_per_category",
        type=int,
        default=100,
        help="Number of samples per carry category for stratified sampling (default: 100)",
    )
    parser.add_argument(
        "--stratified_uniform_remainder",
        type=int,
        default=100,
        help="Number of uniform random samples for stratified sampling (default: 100)",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top-k tokens to store per position (default: 10)",
    )

    parser.add_argument(
        "--enable_greedy_generation",
        action="store_true",
        help="Enable greedy generation for each prompt",
    )
    parser.add_argument(
        "--max_gen_tokens",
        type=int,
        default=10,
        help="Maximum tokens to generate in greedy mode (default: 10)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()

    templates = [TemplateID(t) for t in args.templates]

    config = DatasetConfig(
        model_name=args.model_name,
        output_path=args.output_path,
        templates=templates,
        sampling_strategy=SamplingStrategy(args.sampling_strategy),
        max_value=args.max_value,
        n_samples=args.n_samples,
        stratified_n_per_category=args.stratified_n_per_category,
        stratified_uniform_remainder=args.stratified_uniform_remainder,
        top_k=args.top_k,
        enable_greedy_generation=args.enable_greedy_generation,
        max_gen_tokens=args.max_gen_tokens,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )

    if config.sampling_strategy == SamplingStrategy.RANDOM and config.n_samples is None:
        raise ValueError("--n_samples is required when using random sampling strategy")

    # Standardized configuration printing
    print_config(args, title="Effective Dataset Generation Configuration")

    records, summary = generate_dataset(config)
    write_dataset(records, summary, config.output_path)

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
