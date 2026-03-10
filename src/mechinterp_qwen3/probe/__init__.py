"""Carry-detection probe for transcoder activations.

This module implements a linear logistic probe to determine whether carry
information is linearly decodable from transcoder activations.
"""

from .carry_probe import CarryProbe
from .label_utils import compute_carry_label, generate_addition_examples
from .metrics import ProbeMetrics, compute_metrics
from .probe_trainer import ProbeTrainer

__all__ = [
    "CarryProbe",
    "ProbeTrainer",
    "ProbeMetrics",
    "compute_metrics",
    "compute_carry_label",
    "generate_addition_examples",
]
