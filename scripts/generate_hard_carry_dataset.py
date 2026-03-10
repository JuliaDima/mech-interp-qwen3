#!/usr/bin/env python3
"""Generate a harder carry detection dataset that can't be solved from embeddings alone.

Key ideas:
1. Multi-position carries (not just ones place)
2. Test compositional understanding
3. Out-of-distribution digit combinations for test set
"""

from __future__ import annotations

import argparse
import json

# Add repo root to path
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from mechinterp_qwen3.probe import compute_carry_label


def get_carry_positions(a: int, b: int) -> list[bool]:
    """Get which digit positions require carry.

    Returns:
        List of booleans, [ones_place, tens_place, hundreds_place, ...]
    """
    a_str = str(a).zfill(3)  # Pad to 3 digits
    b_str = str(b).zfill(3)

    carries = []
    carry = 0

    for i in range(2, -1, -1):  # Right to left
        digit_sum = int(a_str[i]) + int(b_str[i]) + carry
        carries.append(digit_sum >= 10)
        carry = digit_sum // 10

    return list(reversed(carries))  # Return left to right


def categorize_carry_type(a: int, b: int) -> str:
    """Categorize the type of carry operation.

    Returns:
        - "no_carry": No carry at all
        - "ones_only": Carry only in ones place
        - "tens_only": Carry only in tens place
        - "propagating": Carry propagates (e.g., 199 + 1)
        - "multiple": Multiple independent carries
    """
    carry_pos = get_carry_positions(a, b)

    if not any(carry_pos):
        return "no_carry"

    if carry_pos[2] and not carry_pos[1] and not carry_pos[0]:
        return "ones_only"

    if carry_pos[1] and not carry_pos[2]:
        return "tens_only"

    # Check for propagating carry (consecutive)
    if carry_pos[1] and carry_pos[2]:
        return "propagating"

    # Multiple non-consecutive carries
    if sum(carry_pos) > 1:
        return "multiple"

    return "other"


def generate_stratified_dataset(
    max_value: int = 99, n_per_category: int = 200, seed: int = 42
) -> dict:
    """Generate dataset stratified by carry type.

    This ensures we have examples that require looking beyond the ones place.
    """
    np.random.seed(seed)

    categories = {
        "no_carry": [],
        "ones_only": [],
        "tens_only": [],
        "propagating": [],
        "multiple": [],
    }

    # Generate all pairs and categorize
    all_pairs = []
    for a in range(max_value + 1):
        for b in range(max_value + 1):
            cat = categorize_carry_type(a, b)
            if cat in categories:
                all_pairs.append((a, b, cat))

    print(f"Total pairs analyzed: {len(all_pairs)}")

    # Shuffle and sample
    np.random.shuffle(all_pairs)

    dataset = {
        "train": {"a": [], "b": [], "labels": [], "categories": []},
        "val": {"a": [], "b": [], "labels": [], "categories": []},
        "test": {"a": [], "b": [], "labels": [], "categories": []},
    }

    category_counts = {cat: {"total": 0, "train": 0, "val": 0, "test": 0} for cat in categories}

    # Sample n_per_category from each category
    for cat in categories:
        cat_pairs = [(a, b) for a, b, c in all_pairs if c == cat]

        if len(cat_pairs) < n_per_category:
            print(f"WARNING: Only {len(cat_pairs)} examples for {cat}, using all")
            n_sample = len(cat_pairs)
        else:
            n_sample = n_per_category

        selected = cat_pairs[:n_sample]

        # Split: 60% train, 20% val, 20% test
        n_train = int(0.6 * n_sample)
        n_val = int(0.2 * n_sample)

        train_pairs = selected[:n_train]
        val_pairs = selected[n_train : n_train + n_val]
        test_pairs = selected[n_train + n_val :]

        # Add to dataset
        for split, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
            for a, b in pairs:
                dataset[split]["a"].append(a)
                dataset[split]["b"].append(b)
                dataset[split]["labels"].append(compute_carry_label(a, b))
                dataset[split]["categories"].append(cat)
                category_counts[cat][split] += 1
                category_counts[cat]["total"] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        n = len(dataset[split]["a"])
        n_carry = sum(dataset[split]["labels"])
        print(f"\n{split.upper()}:")
        print(f"  Total: {n}")
        print(f"  Carry: {n_carry} ({100*n_carry/n:.1f}%)")
        print(f"  No carry: {n - n_carry} ({100*(n-n_carry)/n:.1f}%)")

    print("\n" + "-" * 70)
    print("By Category:")
    print(f"{'Category':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 70)

    for cat in categories:
        print(
            f"{cat:<20} {category_counts[cat]['train']:<10} "
            f"{category_counts[cat]['val']:<10} {category_counts[cat]['test']:<10} "
            f"{category_counts[cat]['total']:<10}"
        )

    print("=" * 70)

    return dataset


def generate_ood_test_set(
    train_dataset: dict, n_samples: int = 500, max_value: int = 99, seed: int = 43
) -> dict:
    """Generate out-of-distribution test set.

    Use digit combinations NOT seen in training for the ones place.
    This tests if the model uses computational layers vs just embeddings.
    """
    np.random.seed(seed)

    # Get ones-place digit pairs from training
    train_ones_pairs = set()
    for a, b in zip(train_dataset["train"]["a"], train_dataset["train"]["b"], strict=False):
        train_ones_pairs.add((a % 10, b % 10))

    print(f"\nTraining set uses {len(train_ones_pairs)} unique ones-place digit pairs")

    # Find unseen ones-place pairs
    all_ones_pairs = [(i, j) for i in range(10) for j in range(10)]
    unseen_pairs = [p for p in all_ones_pairs if p not in train_ones_pairs]

    print(f"Unseen ones-place pairs: {len(unseen_pairs)}")
    print(f"Examples: {unseen_pairs[:10]}")

    if len(unseen_pairs) == 0:
        print("WARNING: All ones-place pairs seen in training!")
        return None

    # Generate OOD examples
    ood_dataset = {"a": [], "b": [], "labels": [], "categories": []}

    attempts = 0
    max_attempts = n_samples * 100

    while len(ood_dataset["a"]) < n_samples and attempts < max_attempts:
        attempts += 1

        # Pick an unseen ones-place pair
        ones_a, ones_b = unseen_pairs[np.random.randint(len(unseen_pairs))]

        # Generate full number with these ones digits
        tens = np.random.randint(0, 10)
        a = tens * 10 + ones_a
        b = tens * 10 + ones_b

        if a <= max_value and b <= max_value:
            ood_dataset["a"].append(a)
            ood_dataset["b"].append(b)
            ood_dataset["labels"].append(compute_carry_label(a, b))
            ood_dataset["categories"].append(categorize_carry_type(a, b))

    print(f"\nGenerated {len(ood_dataset['a'])} OOD examples")
    n_carry = sum(ood_dataset["labels"])
    print(f"  Carry: {n_carry} ({100*n_carry/len(ood_dataset['a']):.1f}%)")

    return ood_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate harder carry detection dataset")
    parser.add_argument("--max_value", type=int, default=99)
    parser.add_argument("--n_per_category", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="data/carry_hard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate_ood", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate stratified dataset
    print("Generating stratified dataset by carry type...")
    dataset = generate_stratified_dataset(
        max_value=args.max_value, n_per_category=args.n_per_category, seed=args.seed
    )

    # Save
    for split in ["train", "val", "test"]:
        output_file = output_dir / f"{split}.json"
        with open(output_file, "w") as f:
            json.dump(dataset[split], f, indent=2)
        print(f"\nSaved {split} to {output_file}")

    # Generate OOD test set if requested
    if args.generate_ood:
        print("\n" + "=" * 70)
        print("GENERATING OUT-OF-DISTRIBUTION TEST SET")
        print("=" * 70)

        ood_dataset = generate_ood_test_set(
            dataset, n_samples=500, max_value=args.max_value, seed=args.seed + 1
        )

        if ood_dataset:
            output_file = output_dir / "test_ood.json"
            with open(output_file, "w") as f:
                json.dump(ood_dataset, f, indent=2)
            print(f"Saved OOD test set to {output_file}")

    print("\n" + "=" * 70)
    print(f"All datasets saved to: {output_dir}")
    print("=" * 70)

    print("\nNext steps:")
    print("1. Train probe with: python scripts/train_carry_probe.py --custom_data data/carry_hard")
    print(
        "2. Exclude layer 0: python scripts/train_carry_probe.py --layers 1-35 --custom_data data/carry_hard"
    )
    print("3. Compare performance on standard test vs OOD test")


if __name__ == "__main__":
    main()
