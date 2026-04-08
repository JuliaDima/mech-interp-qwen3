"""Carry-detection probe for transcoder activations.

This module implements a linear logistic probe to determine whether carry
information is linearly decodable from transcoder activations.
"""

from .carry_probe import CarryProbe
from .dataset_utils import ProbeDataset
from .label_utils import compute_carry_label, compute_unit_digit_label, generate_addition_examples
from .metrics import (
    MulticlassMetrics,
    ProbeMetrics,
    binary_cross_entropy_loss,
    compute_metrics,
    compute_metrics_multiclass,
)
from .probe_trainer import ProbeTrainer

__all__ = [
    "CarryProbe",
    "ProbeTrainer",
    "ProbeDataset",
    "ProbeMetrics",
    "MulticlassMetrics",
    "compute_metrics",
    "compute_metrics_multiclass",
    "binary_cross_entropy_loss",
    "compute_carry_label",
    "compute_unit_digit_label",
    "generate_addition_examples",
]
