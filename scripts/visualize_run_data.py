#!/usr/bin/env python3
"""Visualize the training and validation dataset distribution for a specific run."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from mechinterp_qwen3.probe import generate_addition_examples


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset distribution for a run")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    args_file = run_dir / "args.json"

    if not args_file.exists():
        print(f"Error: {args_file} not found")
        return

    with open(args_file) as f:
        run_args = json.load(f)

    print(
        f"Generating dataset with strategy={run_args['strategy']}, max_value={run_args['max_value']}, seed={run_args['seed']}"
    )

    operands_a, operands_b, labels = generate_addition_examples(
        max_value=run_args["max_value"],
        n_samples=run_args.get("n_train"),
        strategy=run_args["strategy"],
        seed=run_args["seed"],
    )

    n_samples = len(operands_a)
    val_split = run_args["val_split"]
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    # Split (mimicking train_carry_probe.py logic)
    train_a = operands_a[:n_train]
    train_b = operands_b[:n_train]
    val_a = operands_a[n_train:]
    val_b = operands_b[n_train:]

    print(f"Total: {n_samples}, Train: {n_train}, Val: {n_val}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Train distribution
    axes[0].scatter(train_a, train_b, s=1, alpha=0.5, c="blue")
    axes[0].set_title(
        f"Train Set Distribution (n={n_train})\nSeed={run_args['seed']}, Strategy={run_args['strategy']}"
    )
    axes[0].set_xlabel("Operand A")
    axes[0].set_ylabel("Operand B")
    axes[0].set_xlim(-1, run_args["max_value"] + 1)
    axes[0].set_ylim(-1, run_args["max_value"] + 1)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Val distribution
    axes[1].scatter(val_a, val_b, s=5, alpha=0.5, c="red")
    axes[1].set_title(f"Validation Set Distribution (n={n_val})")
    axes[1].set_xlabel("Operand A")
    axes[1].set_ylabel("Operand B")
    axes[1].set_xlim(-1, run_args["max_value"] + 1)
    axes[1].set_ylim(-1, run_args["max_value"] + 1)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path = run_dir / "dataset_distribution.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved distribution plot to: {output_path}")


if __name__ == "__main__":
    main()
