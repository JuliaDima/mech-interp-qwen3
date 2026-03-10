"""Training logic for the carry detection probe."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .carry_probe import CarryProbe
from .dataset_utils import ProbeDataset
from .metrics import ProbeMetrics, compute_metrics


class ProbeTrainer:
    """Trainer for the carry detection probe with regularization support.

    Supports:
    - L1 regularization (for sparse feature selection)
    - L2 regularization
    - Optional gradient clipping
    - Validation monitoring
    - Checkpoint saving
    """

    def __init__(
        self,
        probe: CarryProbe,
        learning_rate: float = 1e-3,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
        gradient_clip: float | None = None,
        device: torch.device | None = None,
    ):
        """Initialize probe trainer.

        Args:
            probe: CarryProbe instance to train
            learning_rate: Learning rate for Adam optimizer
            l1_penalty: L1 regularization coefficient (encourages sparse weights)
            l2_penalty: L2 regularization coefficient (weight decay)
            gradient_clip: If not None, clip gradients to this max norm
            device: Device to train on
        """
        self.probe = probe
        self.learning_rate = learning_rate
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.gradient_clip = gradient_clip
        self.device = device or torch.device("cpu")

        self.probe.to(self.device)

        # Optimizer with L2 regularization via weight_decay
        self.optimizer = torch.optim.Adam(
            self.probe.parameters(),
            lr=learning_rate,
            weight_decay=l2_penalty,
        )

        # Training history
        self.history = {
            "train_loss": [],
            "train_metrics": [],
            "val_loss": [],
            "val_metrics": [],
        }

    def compute_loss(
        self, activations: dict[int, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss with regularization.

        Args:
            activations: Dict mapping layer to activation tensor [batch, d_transcoder]
            labels: Binary labels [batch]

        Returns:
            Tuple of (total_loss, base_loss, probabilities)
        """
        # Forward pass returning [batch] logits
        logits = self.probe(activations, return_logits=True)
        probabilities = torch.sigmoid(logits)

        # labels is [batch], logits is [batch] - direct BCE
        import torch.nn.functional as F

        base_loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        # L1 regularization
        l1_loss = 0.0
        if self.l1_penalty > 0:
            for param in self.probe.parameters():
                l1_loss += torch.sum(torch.abs(param))
            l1_loss = self.l1_penalty * l1_loss

        # Total loss (L2 is handled by weight_decay in optimizer)
        total_loss = base_loss + l1_loss

        return total_loss, base_loss, probabilities

    def train_epoch(
        self,
        dataset: ProbeDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        show_progress: bool = True,
    ) -> tuple[float, ProbeMetrics]:
        """Train for one epoch.

        Args:
            dataset: ProbeDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (average_loss, metrics)
        """
        self.probe.train()

        # Create batches
        n_samples = len(dataset)
        indices = list(range(n_samples))
        if shuffle:
            import random

            random.shuffle(indices)

        n_batches = (n_samples + batch_size - 1) // batch_size
        total_loss = 0.0

        all_predictions = []
        all_probabilities = []
        all_labels = []

        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Training")

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            activations, labels = dataset.get_batch(batch_indices)
            activations = {k: v.to(self.device) for k, v in activations.items()}
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            loss, base_loss, probabilities = self.compute_loss(activations, labels)
            loss.backward()
            if self.gradient_clip is not None:
                nn.utils.clip_grad_norm_(self.probe.parameters(), self.gradient_clip)
            self.optimizer.step()

            total_loss += base_loss.item() * len(batch_indices)

            with torch.no_grad():
                predictions = (probabilities > 0.5).float()
                all_predictions.append(predictions.cpu())
                all_probabilities.append(probabilities.cpu())
                all_labels.append(labels.cpu())
        all_predictions = torch.cat(all_predictions)
        all_probabilities = torch.cat(all_probabilities)
        all_labels = torch.cat(all_labels)

        avg_loss = total_loss / n_samples
        metrics = compute_metrics(all_predictions, all_labels, all_probabilities)

        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(
        self,
        dataset: ProbeDataset,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> tuple[float, ProbeMetrics]:
        """Evaluate probe on a dataset.

        Args:
            dataset: ProbeDataset instance
            batch_size: Batch size
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (average_loss, metrics)
        """
        self.probe.eval()

        n_samples = len(dataset)
        indices = list(range(n_samples))
        n_batches = (n_samples + batch_size - 1) // batch_size

        total_loss = 0.0
        all_predictions = []
        all_probabilities = []
        all_labels = []

        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Evaluating")

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            activations, labels = dataset.get_batch(batch_indices)
            activations = {k: v.to(self.device) for k, v in activations.items()}
            labels = labels.to(self.device)

            logits = self.probe(activations, return_logits=True)
            probabilities = torch.sigmoid(logits)

            import torch.nn.functional as F

            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

            total_loss += loss.item() * len(batch_indices)

            predictions = (probabilities > 0.5).float()
            all_predictions.append(predictions.cpu())
            all_probabilities.append(probabilities.cpu())
            all_labels.append(labels.cpu())

        all_predictions = torch.cat(all_predictions)
        all_probabilities = torch.cat(all_probabilities)
        all_labels = torch.cat(all_labels)

        avg_loss = total_loss / n_samples
        metrics = compute_metrics(all_predictions, all_labels, all_probabilities)

        return avg_loss, metrics

    def fit(
        self,
        train_dataset: ProbeDataset,
        val_dataset: ProbeDataset | None = None,
        n_epochs: int = 10,
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
            n_epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: If not None, stop if validation loss doesn't
                improve for this many epochs
            checkpoint_dir: If not None, save checkpoints to this directory
            save_epochs: Whether to save checkpoints at the end of every epoch
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        best_val_loss = float("inf")
        patience_counter = 0

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(n_epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{n_epochs}")
                print("-" * 50)

            train_loss, train_metrics = self.train_epoch(
                train_dataset, batch_size=batch_size, show_progress=verbose
            )

            self.history["train_loss"].append(train_loss)
            self.history["train_metrics"].append(train_metrics.to_dict())

            if verbose:
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Train Accuracy: {train_metrics.accuracy:.4f}")
                print(f"Train F1: {train_metrics.f1:.4f}")

            if val_dataset is not None:
                val_loss, val_metrics = self.evaluate(
                    val_dataset, batch_size=batch_size, show_progress=verbose
                )

                self.history["val_loss"].append(val_loss)
                self.history["val_metrics"].append(val_metrics.to_dict())

                if verbose:
                    print(f"Val Loss: {val_loss:.4f}")
                    print(f"Val Accuracy: {val_metrics.accuracy:.4f}")
                    print(f"Val F1: {val_metrics.f1:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best checkpoint
                    if checkpoint_dir is not None:
                        self.save_checkpoint(checkpoint_dir / "best_probe.pt")
                else:
                    patience_counter += 1

                # Save epoch checkpoint if requested
                if save_epochs and checkpoint_dir is not None:
                    self.save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1}.pt")

                if (
                    early_stopping_patience is not None
                    and patience_counter >= early_stopping_patience
                ):
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        # Save final checkpoint
        if checkpoint_dir is not None:
            self.save_checkpoint(checkpoint_dir / "final_probe.pt")
            self.save_history(checkpoint_dir / "training_history.json")

        return self.history

    def save_checkpoint(self, path: Path):
        """Save probe checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "probe": self.probe.state_dict_with_metadata(),
            "optimizer": self.optimizer.state_dict(),
            "learning_rate": self.learning_rate,
            "l1_penalty": self.l1_penalty,
            "l2_penalty": self.l2_penalty,
            "gradient_clip": self.gradient_clip,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load probe checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.probe = CarryProbe.from_state_dict_with_metadata(
            checkpoint["probe"], device=self.device
        )
        self.optimizer = torch.optim.Adam(
            self.probe.parameters(),
            lr=checkpoint["learning_rate"],
            weight_decay=checkpoint["l2_penalty"],
        )
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.learning_rate = checkpoint["learning_rate"]
        self.l1_penalty = checkpoint["l1_penalty"]
        self.l2_penalty = checkpoint["l2_penalty"]
        self.gradient_clip = checkpoint.get("gradient_clip")

    def save_history(self, path: Path):
        """Save training history to JSON.

        Args:
            path: Path to save history
        """
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
