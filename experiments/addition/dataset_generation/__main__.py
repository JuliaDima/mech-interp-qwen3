#!/usr/bin/env python3
"""
CLI entrypoint for dataset generation.

Example usage:
    # Grid sampling with all templates (loads defaults from config.yaml automatically)
    python -m mechinterp_qwen3.dataset_generation

    # Override defaults:
    python -m mechinterp_qwen3.dataset_generation --model Qwen/Qwen3-4B --max_value 100
"""

import argparse
from pathlib import Path

from mechinterp_qwen3.utils.config_utils import (
    add_config_args,
    load_config,
    print_config,
    set_parser_defaults_from_config,
)

from .generate_dataset_with_predictions import (
    DatasetConfig,
    SamplingStrategy,
    TemplateID,
    generate_dataset,
    write_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate addition dataset with baseline model statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model name (default: Qwen/Qwen3-4B)",
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
        default=None,
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
        "--max_gen_tokens",
        type=int,
        default=10,
        help="Maximum tokens to generate in greedy mode (default: 10)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation (default: 32)",
    )

    add_config_args(parser)

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_parser()

    # Pre-parse to get config path
    pre, _ = parser.parse_known_args()
    config_dict = load_config(pre.config)

    # Apply defaults from config.yaml
    set_parser_defaults_from_config(parser, config_dict, section="generate_dataset")

    args = parser.parse_args()

    # Shared validation
    if not args.output_path:
        parser.error(
            "the following arguments are required: --output_path (or define in config.yaml)"
        )

    templates = [TemplateID(t) for t in args.templates]

    config = DatasetConfig(
        model=args.model,
        output_path=args.output_path,
        templates=templates,
        sampling_strategy=SamplingStrategy(args.sampling_strategy),
        max_value=args.max_value,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        stratified_n_per_category=args.stratified_n_per_category,
        stratified_uniform_remainder=args.stratified_uniform_remainder,
        top_k=args.top_k,
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
