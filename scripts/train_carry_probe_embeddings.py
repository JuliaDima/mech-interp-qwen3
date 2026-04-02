#!/usr/bin/env python3
"""Train a carry-detection probe on embeddings only.

This script trains a linear logistic probe to detect whether addition problems
require carry operations, using only the token embeddings (before any transformer
processing). This helps determine if carry information is already present in the
embedding space vs. being computed by the transformer layers.

Key differences from train_carry_probe.py:
    - Uses hook_embed (token embeddings) instead of transcoder activations
    - No transcoder required - just the base model
    - Helps isolate whether carry is learned vs. embedded

Example usage:
    # Train on embeddings with final token position
    python scripts/train_carry_probe_embeddings.py --max_value 99 --n_epochs 20

    # Use answer token position
    python scripts/train_carry_probe_embeddings.py --token_position answer --max_value 99

    # Balanced sampling with validation split
    python scripts/train_carry_probe_embeddings.py --strategy balanced --n_train 2000 --val_split 0.2

    # Full grid
    python scripts/train_carry_probe_embeddings.py --strategy grid --max_value 99 --cache_activations
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from experiments.addition.dataset_generation.generate_dataset_with_predictions import (
    TemplateID,
    build_prompt,
)
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.probe import generate_addition_examples
from mechinterp_qwen3.probe.metrics import ProbeMetrics, compute_metrics
from mechinterp_qwen3.utils.config_utils import (
    add_config_args,
    load_config,
    set_parser_defaults_from_config,
)
from mechinterp_qwen3.utils.model_utils import get_default_device
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_carry_probe_embeddings")


class EmbeddingDataset(Dataset):
    """Dataset that extracts embeddings from a model."""

    def __init__(
        self,
        prompts: list[str],
        labels: list[int],
        model: AttributionModel,
        token_position: str | int = "final",
        cache_activations: bool = False,
    ):
        """Initialize embedding dataset.

        Args:
            prompts: List of prompts to extract embeddings from
            labels: Binary labels (0 or 1) for each prompt
            model: AttributionModel to extract embeddings from
            token_position: Which token to extract ('final', 'answer', or int index)
            cache_activations: If True, cache all embeddings upfront
        """
        self.prompts = prompts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.model = model
        self.token_position = token_position
        self.max_seq_len = None

        # Cache if requested
        self.cached_embeddings = None
        if cache_activations:
            log.info("Caching embeddings for all samples...")
            self._cache_all_embeddings()

    def _cache_all_embeddings(self):
        """Cache embeddings for all samples."""
        all_embeddings = []
        self.model.eval()
        with torch.no_grad():
            for prompt in self.prompts:
                emb = self._get_embedding(prompt)
                all_embeddings.append(emb.cpu())
        self.cached_embeddings = all_embeddings
        log.info(f"Cached {len(all_embeddings)} embeddings")

    def _get_embedding(self, prompt: str) -> torch.Tensor:
        """Extract embedding for a single prompt.

        Returns:
            Tensor of shape [seq_len, d_model]
        """
        tokens = tokenize_qwen_input(prompt, self.model.tokenizer, self.model.cfg.device).unsqueeze(
            0
        )  # [1, seq_len]

        # Hook to capture embeddings
        embeddings = {}

        def hook_fn(act, hook):
            embeddings["embed"] = act.clone()
            return act

        # Run forward pass with hook
        with self.model.hooks(fwd_hooks=[("hook_embed", hook_fn)]):
            self.model(tokens)

        # Extract embedding at specified position
        emb = embeddings["embed"][0]  # [seq_len, d_model]

        # Handle token position
        if isinstance(self.token_position, int):
            return emb[self.token_position : self.token_position + 1]  # [1, d_model]
        elif self.token_position == "final":
            return emb[-1:, :]  # [1, d_model]
        elif self.token_position == "answer":
            # Last token position
            return emb[-1:, :]  # [1, d_model]
        else:
            raise ValueError(f"Unknown token_position: {self.token_position}")

    def set_max_seq_len(self, max_seq_len: int):
        """Set maximum sequence length (for compatibility with trainer)."""
        self.max_seq_len = max_seq_len

    def get_batch(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of embeddings and labels.

        Args:
            indices: List of sample indices to retrieve

        Returns:
            embeddings: Tensor of shape [batch, 1, d_model]
            labels: Tensor of shape [batch]
        """
        batch_embeddings = []

        if self.cached_embeddings is not None:
            # Use cached embeddings
            for idx in indices:
                batch_embeddings.append(self.cached_embeddings[idx])
        else:
            # Extract embeddings on-the-fly
            self.model.eval()
            with torch.no_grad():
                for idx in indices:
                    emb = self._get_embedding(self.prompts[idx])
                    batch_embeddings.append(emb.cpu())

        embeddings = torch.stack(batch_embeddings).to(
            device=self.labels.device, dtype=self.model.cfg.dtype
        )  # [batch, 1, d_model]
        labels = self.labels[indices]  # [batch]

        return embeddings, labels

    def __len__(self) -> int:
        return len(self.prompts)


class EmbeddingProbe(nn.Module):
    """Linear probe on embeddings for carry detection."""

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize embedding probe.

        Args:
            d_model: Model embedding dimension
            max_seq_len: Maximum sequence length (usually 1 for single token)
            device: Device to place probe on
            dtype: Data type for probe parameters
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Linear classifier: [d_model] -> [1]
        self.weight = nn.Parameter(torch.zeros(d_model, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Tensor of shape [batch, seq_len, d_model]

        Returns:
            logits: Tensor of shape [batch]
        """
        # Average pool over sequence if needed
        embeddings = (
            embeddings.mean(dim=1) if embeddings.shape[1] > 1 else embeddings.squeeze(1)
        )  # [batch, d_model]

        # Linear classification
        logits = embeddings @ self.weight + self.bias  # [batch]
        return logits

    def get_feature_weights(self) -> torch.Tensor:
        """Get feature weights for analysis.

        Returns:
            weights: Tensor of shape [d_model]
        """
        return self.weight.detach()


class EmbeddingProbeTrainer:
    """Trainer for embedding probe."""

    def __init__(
        self,
        probe: EmbeddingProbe,
        learning_rate: float = 1e-3,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
        gradient_clip: float | None = None,
        device: torch.device | None = None,
    ):
        """Initialize trainer.

        Args:
            probe: EmbeddingProbe to train
            learning_rate: Learning rate for optimizer
            l1_penalty: L1 regularization coefficient
            l2_penalty: L2 regularization coefficient (weight decay)
            gradient_clip: Max gradient norm for clipping
            device: Device for training
        """
        self.probe = probe
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.gradient_clip = gradient_clip
        self.device = device or torch.device("cpu")

        self.probe.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=learning_rate, weight_decay=l2_penalty
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(
        self, dataset: EmbeddingDataset, batch_size: int = 32
    ) -> tuple[float, ProbeMetrics]:
        """Train for one epoch.

        Args:
            dataset: EmbeddingDataset to train on
            batch_size: Batch size

        Returns:
            avg_loss: Average loss over epoch
            metrics: Training metrics
        """
        self.probe.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        n_batches = 0

        # Generate random batch indices
        indices = torch.randperm(len(dataset)).tolist()

        for start_idx in range(0, len(dataset), batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]

            # Get batch
            embeddings, labels = dataset.get_batch(batch_indices)
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.probe(embeddings)  # [batch]

            # Compute loss
            loss = self.criterion(logits, labels)

            # Add L1 regularization if specified
            if self.l1_penalty > 0:
                l1_reg = self.l1_penalty * self.probe.weight.abs().sum()
                loss = loss + l1_reg

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.probe.parameters(), self.gradient_clip)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).detach().cpu())
            all_labels.append(labels.detach().cpu())
            n_batches += 1

        # Compute epoch metrics
        avg_loss = total_loss / n_batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        # Convert probabilities to binary predictions
        binary_preds = (all_preds > 0.5).float()
        metrics = compute_metrics(binary_preds, all_labels, probabilities=all_preds)

        return avg_loss, metrics

    def evaluate(
        self, dataset: EmbeddingDataset, batch_size: int = 32
    ) -> tuple[float, ProbeMetrics]:
        """Evaluate on a dataset.

        Args:
            dataset: EmbeddingDataset to evaluate on
            batch_size: Batch size

        Returns:
            avg_loss: Average loss
            metrics: Evaluation metrics
        """
        self.probe.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        n_batches = 0

        with torch.no_grad():
            for start_idx in range(0, len(dataset), batch_size):
                batch_indices = list(range(start_idx, min(start_idx + batch_size, len(dataset))))

                # Get batch
                embeddings, labels = dataset.get_batch(batch_indices)
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.probe(embeddings)

                # Compute loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                # Track predictions
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())
                n_batches += 1

        # Compute metrics
        avg_loss = total_loss / n_batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        # Convert probabilities to binary predictions
        binary_preds = (all_preds > 0.5).float()
        metrics = compute_metrics(binary_preds, all_labels, probabilities=all_preds)

        return avg_loss, metrics

    def fit(
        self,
        train_dataset: EmbeddingDataset,
        val_dataset: EmbeddingDataset | None = None,
        n_epochs: int = 20,
        batch_size: int = 32,
        early_stopping_patience: int | None = None,
        checkpoint_dir: Path | None = None,
        save_epochs: bool = False,
        verbose: bool = True,
    ) -> dict:
        """Train the probe.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            n_epochs: Number of epochs to train
            batch_size: Batch size
            early_stopping_patience: Early stopping patience (epochs)
            checkpoint_dir: Directory to save checkpoints
            save_epochs: Save checkpoint at end of every epoch
            verbose: Print progress

        Returns:
            history: Dictionary with training history
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(n_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_dataset, batch_size)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_metrics.accuracy)

            # Validate
            if val_dataset is not None:
                val_loss, val_metrics = self.evaluate(val_dataset, batch_size)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_metrics.accuracy)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    if checkpoint_dir is not None:
                        torch.save(self.probe.state_dict(), checkpoint_dir / "best_probe.pt")
                else:
                    patience_counter += 1

                if verbose:
                    log.info(
                        f"Epoch {epoch + 1}/{n_epochs}: "
                        f"train_loss={train_loss:.4f} train_acc={train_metrics.accuracy:.4f} "
                        f"val_loss={val_loss:.4f} val_acc={val_metrics.accuracy:.4f}"
                    )

                # Early stopping check
                if (
                    early_stopping_patience is not None
                    and patience_counter >= early_stopping_patience
                ):
                    log.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if verbose:
                    log.info(
                        f"Epoch {epoch + 1}/{n_epochs}: "
                        f"train_loss={train_loss:.4f} train_acc={train_metrics.accuracy:.4f}"
                    )

            # Save epoch checkpoint
            if save_epochs and checkpoint_dir is not None:
                torch.save(self.probe.state_dict(), checkpoint_dir / f"epoch_{epoch + 1}.pt")

        # Load best model if available
        if checkpoint_dir is not None and (checkpoint_dir / "best_probe.pt").exists():
            self.probe.load_state_dict(torch.load(checkpoint_dir / "best_probe.pt"))
            log.info("Loaded best probe from checkpoint")

        return history


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description="Train carry-detection probe on embeddings only",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    add_config_args(p)

    # Model arguments
    model_group = p.add_argument_group("Model")
    model_group.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model name")
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
        "--save_epochs", action="store_true", help="Save checkpoints at end of every epoch"
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
        help="Cache all embeddings upfront (faster but uses more memory)",
    )

    # Output
    output_group = p.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir",
        default="runs/carry_probe_embeddings",
        help="Output directory for checkpoints and results",
    )
    output_group.add_argument("--run_id", default=None, help="Custom run identifier")
    output_group.add_argument(
        "--save_top_k",
        type=int,
        default=100,
        help="Number of top embedding dimensions to save",
    )

    return p


def main():
    """Main training script."""
    parser = build_parser()

    # Parse known args to get config path
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

    # Load model (no transcoders needed for embeddings!)
    log.info(f"Loading model: {args.model}")
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        args.model,
        device=device,
        dtype=dtype,
    )

    d_model = model.cfg.d_model
    log.info(f"Model loaded. d_model: {d_model}")

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
    log.info("Creating embedding datasets...")
    train_dataset = EmbeddingDataset(
        prompts=train_prompts,
        labels=train_labels,
        model=model,
        token_position=token_position,
        cache_activations=args.cache_activations,
    )

    val_dataset = None
    if n_val > 0:
        val_dataset = EmbeddingDataset(
            prompts=val_prompts,
            labels=val_labels,
            model=model,
            token_position=token_position,
            cache_activations=args.cache_activations,
        )

    # Initialize probe
    log.info("Initializing embedding probe...")
    probe = EmbeddingProbe(
        d_model=d_model,
        max_seq_len=1,  # Single token position
        device=device,
        dtype=dtype,
    )

    n_params = sum(p.numel() for p in probe.parameters())
    log.info(f"Probe parameters: {n_params:,}")

    # Initialize trainer
    log.info("Initializing trainer...")
    trainer = EmbeddingProbeTrainer(
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

    # Analyze top embedding dimensions
    log.info("\nAnalyzing top embedding dimensions...")
    weights = probe.get_feature_weights()  # [d_model]
    top_indices = torch.argsort(weights.abs(), descending=True)[: args.save_top_k]

    analysis = {
        "top_embedding_dims": top_indices.tolist(),
        "top_weights": weights[top_indices].tolist(),
    }

    analysis_dir = output_dir / "feature_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis_dir / "top_embedding_dims.json", "w") as f:
        json.dump(analysis, f, indent=2)

    log.info(f"Top {args.save_top_k} embedding dimensions:")
    for i, (dim_idx, weight) in enumerate(
        zip(top_indices[:20], weights[top_indices[:20]], strict=False)
    ):
        log.info(f"  {i + 1}. Dimension {dim_idx.item()}: weight={weight.item():.4f}")

    # Save final results summary
    summary = {
        "run_id": run_id,
        "d_model": d_model,
        "n_train": n_train,
        "n_val": n_val,
        "token_position": token_position,
        "train_metrics": train_metrics.to_dict(),
        "val_metrics": val_metrics.to_dict() if val_dataset is not None else None,
        "n_epochs": args.n_epochs,
        "best_epoch": history["val_loss"].index(min(history["val_loss"])) + 1
        if val_dataset is not None and history["val_loss"]
        else args.n_epochs,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nResults saved to: {output_dir}")
    log.info("Done!")


if __name__ == "__main__":
    main()
