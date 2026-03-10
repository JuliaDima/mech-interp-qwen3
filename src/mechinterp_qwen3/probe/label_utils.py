"""Utilities for computing carry labels and generating addition examples."""

from __future__ import annotations

import random
from typing import Literal

import numpy as np


def compute_carry_label(a: int, b: int) -> int:
    """Compute binary carry label for addition a + b.

    Args:
        a: First operand
        b: Second operand

    Returns:
        1 if the addition requires at least one carry, 0 otherwise

    Examples:
        >>> compute_carry_label(5, 3)  # 5 + 3 = 8, no carry
        0
        >>> compute_carry_label(5, 7)  # 5 + 7 = 12, carry required
        1
        >>> compute_carry_label(36, 59)  # 36 + 59 = 95, carry required
        1
        >>> compute_carry_label(10, 20)  # 10 + 20 = 30, no carry
        0
    """
    a_str = str(a)
    b_str = str(b)

    max_len = max(len(a_str), len(b_str))
    a_str = a_str.zfill(max_len)
    b_str = b_str.zfill(max_len)

    carry = 0
    for i in range(max_len - 1, -1, -1):
        digit_sum = int(a_str[i]) + int(b_str[i]) + carry
        if digit_sum >= 10:
            return 1  # Carry detected
        carry = digit_sum // 10

    return 0


def has_carry_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized version of carry detection.

    Args:
        a: Array of first operands
        b: Array of second operands

    Returns:
        Boolean array indicating whether each addition requires carry
    """
    return np.array([compute_carry_label(int(x), int(y)) for x, y in zip(a, b, strict=False)])


def generate_addition_examples(
    max_value: int = 99,
    n_samples: int | None = None,
    strategy: Literal["grid", "balanced", "random"] = "grid",
    seed: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Generate addition examples with carry labels.

    Args:
        max_value: Maximum value for operands (inclusive)
        n_samples: Number of samples to generate. If None and strategy='grid',
            generates all combinations. Required for 'balanced' and 'random'.
        strategy: Sampling strategy:
            - 'grid': All combinations a, b in [0, max_value]
            - 'balanced': Equal numbers of carry and no-carry examples
            - 'random': Random uniform sampling
        seed: Random seed for reproducibility

    Returns:
        Tuple of (operands_a, operands_b, labels)

    Raises:
        ValueError: If n_samples is None for 'balanced' or 'random' strategies
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if strategy == "grid":
        # Generate all combinations
        operands_a = []
        operands_b = []
        for a in range(max_value + 1):
            for b in range(max_value + 1):
                operands_a.append(a)
                operands_b.append(b)

        labels = [compute_carry_label(a, b) for a, b in zip(operands_a, operands_b, strict=False)]

        # Shuffle to avoid train/val distribution shift
        indices = list(range(len(operands_a)))
        random.shuffle(indices)
        operands_a = [operands_a[i] for i in indices]
        operands_b = [operands_b[i] for i in indices]
        labels = [labels[i] for i in indices]

        # Subsample if requested
        if n_samples is not None and n_samples < len(operands_a):
            indices = np.random.choice(len(operands_a), n_samples, replace=False)
            operands_a = [operands_a[i] for i in indices]
            operands_b = [operands_b[i] for i in indices]
            labels = [labels[i] for i in indices]

        return operands_a, operands_b, labels

    elif strategy == "balanced":
        if n_samples is None:
            raise ValueError("n_samples must be specified for balanced strategy")

        n_per_class = n_samples // 2
        operands_a = []
        operands_b = []
        labels = []

        # Generate carry examples
        carry_count = 0
        while carry_count < n_per_class:
            a = random.randint(0, max_value)
            b = random.randint(0, max_value)
            if compute_carry_label(a, b) == 1:
                operands_a.append(a)
                operands_b.append(b)
                labels.append(1)
                carry_count += 1

        # Generate no-carry examples
        no_carry_count = 0
        while no_carry_count < n_per_class:
            a = random.randint(0, max_value)
            b = random.randint(0, max_value)
            if compute_carry_label(a, b) == 0:
                operands_a.append(a)
                operands_b.append(b)
                labels.append(0)
                no_carry_count += 1

        # Shuffle
        indices = list(range(len(operands_a)))
        random.shuffle(indices)
        operands_a = [operands_a[i] for i in indices]
        operands_b = [operands_b[i] for i in indices]
        labels = [labels[i] for i in indices]

        return operands_a, operands_b, labels

    elif strategy == "random":
        if n_samples is None:
            raise ValueError("n_samples must be specified for random strategy")

        operands_a = [random.randint(0, max_value) for _ in range(n_samples)]
        operands_b = [random.randint(0, max_value) for _ in range(n_samples)]
        labels = [compute_carry_label(a, b) for a, b in zip(operands_a, operands_b, strict=False)]

        return operands_a, operands_b, labels

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
