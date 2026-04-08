"""Evaluation metrics for the carry detection probe."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ProbeMetrics:
    """Container for probe evaluation metrics.

    Attributes:
        accuracy: Classification accuracy
        precision: Precision (TP / (TP + FP))
        recall: Recall (TP / (TP + FN))
        f1: F1 score (harmonic mean of precision and recall)
        roc_auc: Area under the ROC curve
        loss: Binary cross-entropy loss
        n_samples: Number of samples evaluated
        n_positive: Number of positive (carry=1) samples
        n_negative: Number of negative (carry=0) samples
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    loss: float
    n_samples: int
    n_positive: int
    n_negative: int

    def __str__(self) -> str:
        """Format metrics for display."""
        lines = [
            "Probe Evaluation Metrics",
            "=" * 50,
            f"Samples:     {self.n_samples:6d} (pos: {self.n_positive}, neg: {self.n_negative})",
            f"Accuracy:    {self.accuracy:.4f}",
            f"Precision:   {self.precision:.4f}",
            f"Recall:      {self.recall:.4f}",
            f"F1 Score:    {self.f1:.4f}",
            f"ROC-AUC:     {self.roc_auc:.4f}",
            f"BCE Loss:    {self.loss:.4f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "loss": self.loss,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    probabilities: torch.Tensor | None = None,
) -> ProbeMetrics:
    """Compute evaluation metrics for binary classification.

    Args:
        predictions: Binary predictions [batch]
        labels: Ground truth binary labels [batch]
        probabilities: Predicted probabilities [batch], optional.
            Required for ROC-AUC computation. If None, ROC-AUC will be NaN.

    Returns:
        ProbeMetrics object containing all evaluation metrics

    Raises:
        ValueError: If inputs have mismatched shapes
    """
    if predictions.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}"
        )

    if probabilities is not None and probabilities.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: probabilities {probabilities.shape} vs labels {labels.shape}"
        )

    preds_np = predictions.cpu().float().numpy()
    labels_np = labels.cpu().float().numpy()

    accuracy = accuracy_score(labels_np, preds_np)
    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall = recall_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)

    if probabilities is not None:
        probs_np = probabilities.detach().cpu().float().numpy()
        try:
            roc_auc = roc_auc_score(labels_np, probs_np)
        except ValueError:
            # Can happen if only one class present
            roc_auc = float("nan")
    else:
        roc_auc = float("nan")

    if probabilities is not None:
        eps = 1e-7
        probs_clamped = probabilities.clamp(eps, 1 - eps)
        bce = -(labels * torch.log(probs_clamped) + (1 - labels) * torch.log(1 - probs_clamped))
        loss = bce.mean().item()
    else:
        loss = float("nan")

    # Count samples
    n_samples = len(labels)
    n_positive = int(labels.sum().item())
    n_negative = n_samples - n_positive

    return ProbeMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        loss=loss,
        n_samples=n_samples,
        n_positive=n_positive,
        n_negative=n_negative,
    )


def binary_cross_entropy_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute binary cross-entropy loss with numerical stability.

    Args:
        probabilities: Predicted probabilities [batch]
        labels: Ground truth binary labels [batch]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss tensor (scalar if reduction != 'none')
    """
    eps = 1e-7
    probs_clamped = probabilities.clamp(eps, 1 - eps)
    bce = -(labels * torch.log(probs_clamped) + (1 - labels) * torch.log(1 - probs_clamped))

    if reduction == "mean":
        return bce.mean()
    elif reduction == "sum":
        return bce.sum()
    elif reduction == "none":
        return bce
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


@dataclass
class MulticlassMetrics:
    """Container for multiclass probe evaluation metrics.

    Attributes:
        accuracy: Top-1 classification accuracy
        loss: Cross-entropy loss
        n_samples: Number of samples evaluated
        n_classes: Number of classes
        per_class_accuracy: Per-class accuracy (list of length n_classes)
    """

    accuracy: float
    loss: float
    n_samples: int
    n_classes: int
    per_class_accuracy: list[float]

    def __str__(self) -> str:
        lines = [
            "Multiclass Probe Metrics",
            "=" * 50,
            f"Samples:     {self.n_samples:6d}",
            f"Classes:     {self.n_classes}",
            f"Accuracy:    {self.accuracy:.4f}",
            f"CE Loss:     {self.loss:.4f}",
        ]
        for i, acc in enumerate(self.per_class_accuracy):
            lines.append(f"  Class {i}:   {acc:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "loss": self.loss,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "per_class_accuracy": self.per_class_accuracy,
        }


def compute_metrics_multiclass(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor | None = None,
) -> MulticlassMetrics:
    """Compute evaluation metrics for multiclass classification.

    Args:
        predictions: Class predictions [batch] (integer class indices)
        labels: Ground truth class labels [batch] (integer class indices)
        logits: Raw logits [batch, n_classes], used for loss computation.

    Returns:
        MulticlassMetrics object
    """
    import torch.nn.functional as F

    preds_np = predictions.cpu().long().numpy()
    labels_np = labels.cpu().long().numpy()

    accuracy = accuracy_score(labels_np, preds_np)

    n_classes = int(labels_np.max()) + 1 if logits is None else logits.shape[-1]
    per_class_accuracy = []
    for c in range(n_classes):
        mask = labels_np == c
        if mask.sum() == 0:
            per_class_accuracy.append(float("nan"))
        else:
            per_class_accuracy.append(float((preds_np[mask] == c).mean()))

    if logits is not None:
        loss = F.cross_entropy(logits.cpu().float(), labels.cpu().long()).item()
    else:
        loss = float("nan")

    return MulticlassMetrics(
        accuracy=accuracy,
        loss=loss,
        n_samples=len(labels),
        n_classes=n_classes,
        per_class_accuracy=per_class_accuracy,
    )
