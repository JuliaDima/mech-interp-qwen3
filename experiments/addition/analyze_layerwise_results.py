#!/usr/bin/env python3
"""Analyze layer-wise probe results and generate visualizations.

This script:
1. Collects results from all layer-specific probe runs
2. Plots accuracy vs layer depth
3. Identifies which layers contain carry information
4. Generates summary statistics

Usage:
    python scripts/analyze_layerwise_results.py --scan_dir runs/carry_probe/layerwise_scan_20260309_120000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def collect_results(scan_dir: Path) -> dict[int, dict]:
    """Collect summary.json from all layer subdirectories.

    Args:
        scan_dir: Directory containing layer_X subdirectories

    Returns:
        Dictionary mapping layer index to metrics
    """
    results = {}

    for layer_dir in sorted(scan_dir.glob("layer_*")):
        summary_file = layer_dir / "summary.json"

        if not summary_file.exists():
            print(f"WARNING: No summary.json found for {layer_dir.name}")
            continue

        layer_idx = int(layer_dir.name.split("_")[1])

        with open(summary_file) as f:
            data = json.load(f)
            results[layer_idx] = data

    return results


def plot_accuracy_vs_layer(results: dict[int, dict], output_dir: Path):
    """Plot train and validation accuracy vs layer depth.

    Args:
        results: Dictionary mapping layer to metrics
        output_dir: Directory to save plots
    """
    layers = sorted(results.keys())
    train_acc = [results[layer]["train_metrics"]["accuracy"] for layer in layers]
    val_acc = [results[layer]["val_metrics"]["accuracy"] for layer in layers]

    plt.figure(figsize=(12, 6))
    plt.plot(layers, train_acc, "o-", label="Train Accuracy", linewidth=2, markersize=6)
    plt.plot(layers, val_acc, "s-", label="Val Accuracy", linewidth=2, markersize=6)
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random Baseline")

    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Carry Detection Accuracy vs Layer Depth", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.4, 1.0])

    # Annotate best layer
    best_layer = max(layers, key=lambda layer: results[layer]["val_metrics"]["accuracy"])
    best_acc = results[best_layer]["val_metrics"]["accuracy"]
    plt.annotate(
        f"Best: Layer {best_layer}\n({best_acc:.2%})",
        xy=(best_layer, best_acc),
        xytext=(best_layer + 2, best_acc - 0.05),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontsize=10,
        color="red",
        fontweight="bold",
    )

    plt.tight_layout()
    output_path = output_dir / "accuracy_vs_layer.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_f1_vs_layer(results: dict[int, dict], output_dir: Path):
    """Plot F1 score vs layer depth.

    Args:
        results: Dictionary mapping layer to metrics
        output_dir: Directory to save plots
    """
    layers = sorted(results.keys())
    train_f1 = [results[layer]["train_metrics"]["f1"] for layer in layers]
    val_f1 = [results[layer]["val_metrics"]["f1"] for layer in layers]

    plt.figure(figsize=(12, 6))
    plt.plot(layers, train_f1, "o-", label="Train F1", linewidth=2, markersize=6)
    plt.plot(layers, val_f1, "s-", label="Val F1", linewidth=2, markersize=6)

    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("Carry Detection F1 Score vs Layer Depth", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.4, 1.0])

    plt.tight_layout()
    output_path = output_dir / "f1_vs_layer.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def print_summary(results: dict[int, dict]):
    """Print summary statistics.

    Args:
        results: Dictionary mapping layer to metrics
    """
    print("\n" + "=" * 70)
    print("LAYER-WISE CARRY DETECTION SUMMARY")
    print("=" * 70)

    layers = sorted(results.keys())
    val_accs = [(layer, results[layer]["val_metrics"]["accuracy"]) for layer in layers]
    val_accs_sorted = sorted(val_accs, key=lambda x: x[1], reverse=True)

    print(f"\nTotal layers analyzed: {len(layers)}")
    print(f"Layers tested: {min(layers)} to {max(layers)}")

    print("\n" + "-" * 70)
    print("TOP 10 LAYERS BY VALIDATION ACCURACY")
    print("-" * 70)
    print(f"{'Rank':<6} {'Layer':<8} {'Val Acc':<12} {'Train Acc':<12} {'F1 Score':<12}")
    print("-" * 70)

    for rank, (layer, val_acc) in enumerate(val_accs_sorted[:10], 1):
        train_acc = results[layer]["train_metrics"]["accuracy"]
        f1 = results[layer]["val_metrics"]["f1"]
        print(f"{rank:<6} {layer:<8} {val_acc:<12.4f} {train_acc:<12.4f} {f1:<12.4f}")

    print("\n" + "-" * 70)
    print("LAYER GROUPS")
    print("-" * 70)

    early_layers = [layer for layer in layers if layer < 12]
    middle_layers = [layer for layer in layers if 12 <= layer < 24]
    late_layers = [layer for layer in layers if layer >= 24]

    if early_layers:
        early_acc = np.mean([results[layer]["val_metrics"]["accuracy"] for layer in early_layers])
        print(f"Early layers (0-11):   Mean accuracy = {early_acc:.4f}")

    if middle_layers:
        middle_acc = np.mean([results[layer]["val_metrics"]["accuracy"] for layer in middle_layers])
        print(f"Middle layers (12-23): Mean accuracy = {middle_acc:.4f}")

    if late_layers:
        late_acc = np.mean([results[layer]["val_metrics"]["accuracy"] for layer in late_layers])
        print(f"Late layers (24-35):   Mean accuracy = {late_acc:.4f}")

    print("\n" + "-" * 70)
    print("INSIGHTS")
    print("-" * 70)

    best_layer, best_acc = val_accs_sorted[0]
    print(f"• Best single layer: Layer {best_layer} with {best_acc:.2%} accuracy")

    high_acc_layers = [layer for layer, acc in val_accs if acc > 0.85]
    if high_acc_layers:
        print(f"• Layers with >85% accuracy: {high_acc_layers}")

    if len(early_layers) > 0 and len(late_layers) > 0:
        early_best = max([results[layer]["val_metrics"]["accuracy"] for layer in early_layers])
        late_best = max([results[layer]["val_metrics"]["accuracy"] for layer in late_layers])
        if late_best > early_best + 0.1:
            print(
                f"• Carry computation appears in later layers (late: {late_best:.2%} vs early: {early_best:.2%})"
            )
        elif early_best > 0.8:
            print(
                f"• WARNING: High early-layer accuracy ({early_best:.2%}) suggests probe may be computing from embeddings"
            )

    print("=" * 70 + "\n")


def save_summary_csv(results: dict[int, dict], output_dir: Path):
    """Save results to CSV file.

    Args:
        results: Dictionary mapping layer to metrics
        output_dir: Directory to save CSV
    """
    import csv

    output_path = output_dir / "layerwise_summary.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Layer",
                "Train_Acc",
                "Val_Acc",
                "Train_F1",
                "Val_F1",
                "Train_Precision",
                "Val_Precision",
                "Train_Recall",
                "Val_Recall",
            ]
        )

        for layer in sorted(results.keys()):
            train = results[layer]["train_metrics"]
            val = results[layer]["val_metrics"]
            writer.writerow(
                [
                    layer,
                    train["accuracy"],
                    val["accuracy"],
                    train["f1"],
                    val["f1"],
                    train["precision"],
                    val["precision"],
                    train["recall"],
                    val["recall"],
                ]
            )

    print(f"Saved CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze layer-wise probe results")
    parser.add_argument(
        "--scan_dir", required=True, help="Directory containing layer_X subdirectories"
    )
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)

    if not scan_dir.exists():
        print(f"ERROR: Directory not found: {scan_dir}")
        return

    print(f"Analyzing results from: {scan_dir}")

    # Collect results
    results = collect_results(scan_dir)

    if not results:
        print("ERROR: No results found!")
        return

    print(f"Found results for {len(results)} layers")

    # Create analysis directory
    analysis_dir = scan_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_accuracy_vs_layer(results, analysis_dir)
    plot_f1_vs_layer(results, analysis_dir)

    # Save CSV
    save_summary_csv(results, analysis_dir)

    # Print summary
    print_summary(results)

    print(f"\nAll analysis saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
