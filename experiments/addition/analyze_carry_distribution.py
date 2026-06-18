#!/usr/bin/env python3
"""Analyze the distribution of carry labels in addition problems."""

# Add repo root to path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from mechinterp_qwen3.probe import compute_carry_label


def analyze_carry_distribution(max_value=99):
    """Analyze carry distribution for all pairs (a, b) in [0, max_value]."""

    total = 0
    carry_count = 0

    # Count by digit patterns
    digit_patterns = {}

    for a in range(max_value + 1):
        for b in range(max_value + 1):
            total += 1
            has_carry = compute_carry_label(a, b)
            carry_count += has_carry

            # Categorize by last digits (ones place)
            last_a = a % 10
            last_b = b % 10
            key = (last_a, last_b)

            if key not in digit_patterns:
                digit_patterns[key] = {"total": 0, "carry": 0}

            digit_patterns[key]["total"] += 1
            digit_patterns[key]["carry"] += has_carry

    print(f"Total pairs: {total}")
    print(f"Carry pairs: {carry_count} ({100 * carry_count / total:.2f}%)")
    print(f"No-carry pairs: {total - carry_count} ({100 * (total - carry_count) / total:.2f}%)")
    print()

    # Analyze by last digit sum
    print("Distribution by ones-place digit sum:")
    print(f"{'Sum':<6} {'Carry?':<10} {'Count':<10} {'%':<10}")
    print("-" * 40)

    by_sum = {}
    for a in range(10):
        for b in range(10):
            digit_sum = a + b
            if digit_sum not in by_sum:
                by_sum[digit_sum] = {"total": 0, "carry": 0}

            # Count how many pairs have this digit sum
            count = (max_value // 10 + (1 if a <= max_value % 10 else 0)) * (
                max_value // 10 + (1 if b <= max_value % 10 else 0)
            )

            by_sum[digit_sum]["total"] += count
            if digit_sum >= 10:
                by_sum[digit_sum]["carry"] += count

    for s in sorted(by_sum.keys()):
        requires_carry = "Yes" if s >= 10 else "No"
        count = by_sum[s]["total"]
        pct = 100 * count / total
        print(f"{s:<6} {requires_carry:<10} {count:<10} {pct:.2f}%")

    print()
    print("Key insight: Any pair where last_digit(a) + last_digit(b) >= 10 ALWAYS has carry")
    print("This makes carry trivially predictable from embeddings!")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Heatmap of carry by last digits
    heatmap = np.zeros((10, 10))
    for a in range(10):
        for b in range(10):
            # Carry happens when sum >= 10
            heatmap[a, b] = 1 if (a + b) >= 10 else 0

    im = axes[0].imshow(heatmap, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[0].set_xlabel("Last digit of B")
    axes[0].set_ylabel("Last digit of A")
    axes[0].set_title("Carry Required by Last Digits\n(Red = Carry, Green = No Carry)")
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(10))

    # Add text annotations
    for i in range(10):
        for j in range(10):
            digit_sum = i + j
            axes[0].text(j, i, f"{digit_sum}", ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=axes[0])

    # Plot 2: Distribution
    categories = ["No Carry\n(30.25%)", "Carry\n(69.75%)"]
    counts = [total - carry_count, carry_count]
    colors = ["green", "red"]

    axes[1].bar(categories, counts, color=colors, alpha=0.7, edgecolor="black")
    axes[1].set_ylabel("Number of Examples")
    axes[1].set_title("Carry Distribution (0-99 addition)")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add count labels on bars
    for i, (_cat, count) in enumerate(zip(categories, counts, strict=False)):
        axes[1].text(i, count + 100, f"{count:,}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("carry_distribution_analysis.png", dpi=300, bbox_inches="tight")
    print("\nSaved visualization to: carry_distribution_analysis.png")
    plt.close()


if __name__ == "__main__":
    analyze_carry_distribution(max_value=99)
