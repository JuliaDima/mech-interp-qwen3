#!/usr/bin/env python3
"""Train a carry-detection probe on transcoder activations.

This script trains a linear logistic probe to detect whether addition problems
require carry operations, using transcoder activations extracted online from
the model.

Example usage:
    # Train on single layer
    python scripts/train_carry_probe.py --layers 14 --max_value 99 --n_epochs 20

    # Train on multiple layers with regularization
    python scripts/train_carry_probe.py --layers 7 14 21 --l1_penalty 1e-5 --l2_penalty 1e-4

    # Use balanced sampling with validation split
    python scripts/train_carry_probe.py --layers 14 --strategy balanced --n_train 2000 --val_split 0.2

    # Full grid with caching
    python scripts/train_carry_probe.py --layers 14 --strategy grid --max_value 99 --cache_activations
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

# Note: sys.path modification must occur before project imports
# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
# Project imports must come after sys.path modification
from experiments.addition.dataset_generation.generate_add_dataset import (
    TemplateID,
    build_prompt,
)
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.probe import (
    CarryProbe,
    ProbeTrainer,
    generate_addition_examples,
)
from mechinterp_qwen3.probe.dataset_utils import ProbeDataset
from mechinterp_qwen3.probe.feature_analysis import (
    analyze_feature_importance,
    print_top_features,
)
from mechinterp_qwen3.utils.config_utils import (
    add_config_args,
    load_config,
    set_parser_defaults_from_config,
)
from mechinterp_qwen3.utils.model_utils import get_default_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_carry_probe")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser with all training options."""
    p = argparse.ArgumentParser(
        description="Train carry-detection probe on transcoder activations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    add_config_args(p)

    # Model arguments
    model_group = p.add_argument_group("Model")
    model_group.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model name")
    model_group.add_argument(
        "--transcoder_set",
        default="mwhanna/qwen3-4b-transcoders",
        help="Transcoder set identifier",
    )
    model_group.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Model dtype",
    )
    model_group.add_argument("--device", default=None, help="Device (cuda/cpu)")

    # Probe architecture
    probe_group = p.add_argument_group("Probe Architecture")
    probe_group.add_argument(
        "--layers",
        type=int,
        nargs="*",
        help="Transcoder layers to use (e.g., --layers 7 14 21). If not provided, all layers will be used.",
    )
    probe_group.add_argument(
        "--token_position",
        default="final",
        help="Token position: 'final', 'answer', or integer index",
    )

    # Dataset generation
    data_group = p.add_argument_group("Dataset")
    data_group.add_argument("--max_value", type=int, default=99, help="Maximum operand value")
    data_group.add_argument(
        "--strategy",
        default="grid",
        choices=["grid", "balanced", "random"],
        help="Sampling strategy",
    )
    data_group.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="Number of training samples (required for balanced/random)",
    )
    data_group.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    data_group.add_argument("--seed", type=int, default=42, help="Random seed")
    data_group.add_argument(
        "--template",
        default="T0",
        choices=["T0", "T1", "T2"],
        help="Prompt template (T0='calc: a+b= ')",
    )

    # Training hyperparameters
    train_group = p.add_argument_group("Training")
    train_group.add_argument("--n_epochs", type=int, default=20, help="Number of epochs")
    train_group.add_argument("--batch_size", type=int, default=8, help="Batch size")
    train_group.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate")
    train_group.add_argument(
        "--l1_penalty", type=float, default=0.0, help="L1 regularization coefficient"
    )
    train_group.add_argument(
        "--l2_penalty", type=float, default=0.0, help="L2 regularization coefficient"
    )
    train_group.add_argument(
        "--save_epochs", action="store_true", help="Save checkpoints at the end of every epoch"
    )
    train_group.add_argument(
        "--gradient_clip", type=float, default=None, help="Gradient clipping max norm"
    )
    train_group.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs)",
    )

    # Performance options
    perf_group = p.add_argument_group("Performance")
    perf_group.add_argument(
        "--cache_activations",
        action="store_true",
        help="Cache all activations upfront (faster but uses more memory)",
    )

    # Output
    output_group = p.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir",
        default="runs/carry_probe",
        help="Output directory for checkpoints and results",
    )
    output_group.add_argument("--run_id", default=None, help="Custom run identifier")
    output_group.add_argument(
        "--save_top_k",
        type=int,
        default=100,
        help="Number of top features to save per layer",
    )

    return p


def validate_layers(layers: list[int], n_layers: int):
    """Validate that layer indices are valid.

    Args:
        layers: List of layer indices
        n_layers: Total number of layers in model

    Raises:
        ValueError: If any layer index is invalid
    """
    if not layers:
        raise ValueError("Must specify at least one layer")

    if len(layers) != len(set(layers)):
        raise ValueError(f"Duplicate layers specified: {layers}")

    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} out of range [0, {n_layers})")


def main():
    """Main training script."""
    parser = build_parser()

    # Parse known args to get the config path if provided
    early_args, _ = parser.parse_known_args()

    # Load config and set defaults
    config = load_config(early_args.config)
    set_parser_defaults_from_config(parser, config)

    args = parser.parse_args()

    # Create run directory
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S") if args.run_id is None else args.run_id

    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Output directory: {output_dir}")

    # Save arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Set device
    device = get_default_device() if args.device is None else torch.device(args.device)

    log.info(f"Using device: {device}")

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    # Load model
    log.info(f"Loading model: {args.model}")
    log.info(f"Loading transcoders: {args.transcoder_set}")

    model = AttributionModel.from_pretrained(
        model_name=args.model,
        transcoder_set=args.transcoder_set,
        device=device,
        dtype=dtype,
    )

    log.info(f"Model loaded. Layers: {model.cfg.n_layers}")

    # Validate layers
    if not args.layers:
        args.layers = list(range(model.cfg.n_layers))
        log.info(f"No layers specified, defaulting to all {model.cfg.n_layers} layers.")
    validate_layers(args.layers, model.cfg.n_layers)
    log.info(f"Using transcoder layers: {args.layers}")

    # Get transcoder dimension
    # Inspect the first layer transcoder
    d_transcoder = model.transcoders.d_transcoder
    log.info(f"Transcoder dimension: {d_transcoder}")

    # Generate dataset
    log.info("Generating addition examples...")
    log.info(f"Strategy: {args.strategy}, Max value: {args.max_value}")

    operands_a, operands_b, labels = generate_addition_examples(
        max_value=args.max_value,
        n_samples=args.n_train,
        strategy=args.strategy,
        seed=args.seed,
    )

    log.info(f"Generated {len(operands_a)} examples")
    log.info(f"  Positive (carry=1): {sum(labels)}")
    log.info(f"  Negative (carry=0): {len(labels) - sum(labels)}")

    # Convert to template format
    template_id = getattr(TemplateID, args.template)
    prompts = [
        build_prompt(template_id, a, b) for a, b in zip(operands_a, operands_b, strict=False)
    ]

    # Train/val split
    n_samples = len(prompts)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val

    log.info(f"Splitting: {n_train} train, {n_val} val")

    train_prompts = prompts[:n_train]
    train_labels = labels[:n_train]
    val_prompts = prompts[n_train:]
    val_labels = labels[n_train:]

    # Parse token position
    try:
        token_position = int(args.token_position)
    except ValueError:
        token_position = args.token_position

    # Create datasets
    log.info("Creating probe datasets...")
    train_dataset = ProbeDataset(
        prompts=train_prompts,
        labels=train_labels,
        model=model,
        layers=args.layers,
        token_position=token_position,
        cache_activations=args.cache_activations,
    )

    val_dataset = None
    if n_val > 0:
        val_dataset = ProbeDataset(
            prompts=val_prompts,
            labels=val_labels,
            model=model,
            layers=args.layers,
            token_position=token_position,
            cache_activations=args.cache_activations,
        )

    # Get max_seq_len from dataset
    log.info("Determining max sequence length...")
    # Fetch batch of 1 to inspect the shape from the dataset
    sample_acts, _ = train_dataset.get_batch([0])
    # [batch, seq_len, d_transcoder]
    max_seq_len = sample_acts[args.layers[0]].shape[1]
    log.info(f"Max sequence length is {max_seq_len}")

    train_dataset.set_max_seq_len(max_seq_len)
    if val_dataset is not None:
        val_dataset.set_max_seq_len(max_seq_len)

    # Initialize probe
    log.info("Initializing carry probe...")
    probe = CarryProbe(
        layers=args.layers,
        d_transcoder=d_transcoder,
        max_seq_len=max_seq_len,
        device=device,
        dtype=dtype,
    )

    n_params = sum(p.numel() for p in probe.parameters())
    log.info(f"Probe parameters: {n_params:,}")

    # Initialize trainer
    log.info("Initializing trainer...")
    trainer = ProbeTrainer(
        probe=probe,
        learning_rate=args.learning_rate,
        l1_penalty=args.l1_penalty,
        l2_penalty=args.l2_penalty,
        gradient_clip=args.gradient_clip,
        device=device,
    )

    # Train
    log.info("Starting training...")
    log.info("=" * 70)

    history = trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=output_dir / "checkpoints",
        save_epochs=args.save_epochs,
        verbose=True,
    )

    log.info("=" * 70)
    log.info("Training complete!")

    # Final evaluation
    log.info("\nFinal Evaluation:")
    log.info("-" * 70)

    train_loss, train_metrics = trainer.evaluate(train_dataset, batch_size=args.batch_size)
    log.info("\nTrain Set:")
    log.info(str(train_metrics))

    if val_dataset is not None:
        val_loss, val_metrics = trainer.evaluate(val_dataset, batch_size=args.batch_size)
        log.info("\nValidation Set:")
        log.info(str(val_metrics))

    # Feature importance analysis
    log.info("\nAnalyzing feature importance...")
    analysis_dir = output_dir / "feature_analysis"
    analyze_feature_importance(probe=probe, top_k=args.save_top_k, save_dir=analysis_dir)
    log.info(f"Feature analysis saved to: {analysis_dir}")

    # Print top features
    for layer in args.layers:
        print_top_features(probe, layer, k=20)

    # Save final results summary
    summary = {
        "run_id": run_id,
        "layers": args.layers,
        "d_transcoder": d_transcoder,
        "n_train": n_train,
        "n_val": n_val,
        "train_metrics": train_metrics.to_dict(),
        "val_metrics": val_metrics.to_dict() if val_dataset is not None else None,
        "n_epochs": args.n_epochs,
        "best_epoch": history["val_loss"].index(min(history["val_loss"])) + 1
        if val_dataset is not None
        else args.n_epochs,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nResults saved to: {output_dir}")
    log.info("Done!")


if __name__ == "__main__":
    main()
