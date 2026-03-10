#!/usr/bin/env python3
"""Analyze which layers contribute most to carry detection.

This script loads a trained probe and analyzes the feature importance
per layer to determine which layers are most important for detecting carries.
"""

from __future__ import annotations

import argparse

# Add repo root to path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from mechinterp_qwen3.probe import CarryProbe


def load_probe(checkpoint_path: Path, device: str = "cpu") -> CarryProbe:
    """Load a trained probe from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the probe on

    Returns:
        Loaded CarryProbe instance
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract probe information from checkpoint
    probe_data = checkpoint.get("probe", checkpoint)
    probe_state = probe_data.get("state_dict", probe_data)

    # Get configuration
    layers = probe_data.get("layers")
    d_transcoder = probe_data.get("d_transcoder")
    max_seq_len = probe_data.get("max_seq_len", 0)

    # Infer from state dict if not available
    if "linear.weight" in probe_state:
        linear_weight = probe_state["linear.weight"]
        total_features = linear_weight.shape[1]

        if d_transcoder is None:
            # Assume d_transcoder = 163840 for Qwen3-4B
            d_transcoder = 163840

        n_layers = total_features // d_transcoder

        if layers is None:
            layers = list(range(n_layers))

    print("Detected probe configuration:")
    print(f"  Total features: {total_features:,}")
    print(f"  Transcoder dim: {d_transcoder:,}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Layers: {layers}")

    # Create probe instance
    probe = CarryProbe(
        layers=layers,
        d_transcoder=d_transcoder,
        max_seq_len=max_seq_len,
        device=torch.device(device),
        dtype=linear_weight.dtype,
    )

    # Load weights
    probe.load_state_dict(probe_state)

    return probe


def analyze_layer_importance(probe: CarryProbe, top_k: int = 1000) -> dict:
    """Analyze feature importance per layer.

    Args:
        probe: Trained CarryProbe instance
        top_k: Number of top features to consider per layer

    Returns:
        Dictionary with analysis results per layer
    """
    results = {}

    for layer_idx in probe.layers:
        # Get weights for this layer
        layer_weights = probe.get_layer_weights(layer_idx)  # [d_transcoder]

        # Convert to numpy for analysis (convert to float32 first to avoid bfloat16 issues)
        weights_np = layer_weights.detach().cpu().float().numpy()

        # Compute various metrics
        abs_weights = np.abs(weights_np)

        results[layer_idx] = {
            "mean_abs_weight": float(np.mean(abs_weights)),
            "max_abs_weight": float(np.max(abs_weights)),
            "std_abs_weight": float(np.std(abs_weights)),
            "l1_norm": float(np.sum(abs_weights)),
            "l2_norm": float(np.linalg.norm(weights_np)),
            "num_nonzero": int(np.sum(abs_weights > 1e-6)),
            "top_k_sum": float(np.sum(np.sort(abs_weights)[-top_k:])),
            "top_k_mean": float(np.mean(np.sort(abs_weights)[-top_k:])),
        }

    return results


def plot_layer_importance(results: dict, output_dir: Path, metric: str = "l1_norm"):
    """Plot layer importance based on a specific metric.

    Args:
        results: Dictionary from analyze_layer_importance
        output_dir: Directory to save plots
        metric: Metric to plot (e.g., 'l1_norm', 'mean_abs_weight', 'top_k_sum')
    """
    layers = sorted(results.keys())
    values = [results[layer][metric] for layer in layers]

    plt.figure(figsize=(14, 6))
    plt.bar(layers, values, color="steelblue", alpha=0.7, edgecolor="black")
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
    plt.title(f"Layer Importance for Carry Detection ({metric})", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    output_path = output_dir / f"layer_importance_{metric}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_all_metrics(results: dict, output_dir: Path):
    """Create a comprehensive plot showing multiple metrics.

    Args:
        results: Dictionary from analyze_layer_importance
        output_dir: Directory to save plots
    """
    layers = sorted(results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics = [
        ("l1_norm", "L1 Norm", "steelblue"),
        ("l2_norm", "L2 Norm", "coral"),
        ("mean_abs_weight", "Mean Absolute Weight", "seagreen"),
        ("max_abs_weight", "Max Absolute Weight", "tomato"),
        ("top_k_sum", "Top-1000 Sum", "purple"),
        ("top_k_mean", "Top-1000 Mean", "orange"),
    ]

    for idx, (metric, title, color) in enumerate(metrics):
        values = [results[layer][metric] for layer in layers]

        axes[idx].bar(layers, values, color=color, alpha=0.7, edgecolor="black")
        axes[idx].set_xlabel("Layer Index", fontsize=10)
        axes[idx].set_ylabel(title, fontsize=10)
        axes[idx].set_title(title, fontsize=11, fontweight="bold")
        axes[idx].grid(True, alpha=0.3, axis="y")

        # Highlight top 5 layers
        top_5_indices = np.argsort(values)[-5:]
        for top_idx in top_5_indices:
            layer = layers[top_idx]
            axes[idx].bar(
                layer, values[top_idx], color="red", alpha=0.8, edgecolor="darkred", linewidth=2
            )

    plt.suptitle(
        "Layer-wise Feature Importance for Carry Detection", fontsize=16, fontweight="bold", y=1.00
    )
    plt.tight_layout()

    output_path = output_dir / "layer_importance_comprehensive.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comprehensive plot: {output_path}")
    plt.close()


def print_summary(results: dict):
    """Print summary statistics.

    Args:
        results: Dictionary from analyze_layer_importance
    """
    print("\n" + "=" * 80)
    print("LAYER-WISE FEATURE IMPORTANCE SUMMARY")
    print("=" * 80)

    layers = sorted(results.keys())

    # Rank by different metrics
    metrics = ["l1_norm", "l2_norm", "mean_abs_weight", "top_k_sum"]

    for metric in metrics:
        print(f"\n{'─' * 80}")
        print(f"TOP 10 LAYERS BY {metric.upper().replace('_', ' ')}")
        print(f"{'─' * 80}")

        ranked = sorted(layers, key=lambda layer: results[layer][metric], reverse=True)[:10]

        print(f"{'Rank':<6} {'Layer':<8} {metric:<20} {'% of Total':<12}")
        print(f"{'─' * 80}")

        total = sum(results[layer][metric] for layer in layers)

        for rank, layer in enumerate(ranked, 1):
            value = results[layer][metric]
            pct = 100 * value / total if total > 0 else 0
            print(f"{rank:<6} {layer:<8} {value:<20.6f} {pct:<12.2f}%")

    print("\n" + "=" * 80)
    print("LAYER GROUPS")
    print("=" * 80)

    # Divide into early, middle, late layers
    n_layers = len(layers)
    early = [layer for layer in layers if layer < n_layers // 3]
    middle = [layer for layer in layers if n_layers // 3 <= layer < 2 * n_layers // 3]
    late = [layer for layer in layers if layer >= 2 * n_layers // 3]

    for group_name, group_layers in [("Early", early), ("Middle", middle), ("Late", late)]:
        if not group_layers:
            continue

        print(f"\n{group_name} layers ({min(group_layers)}-{max(group_layers)}):")

        for metric in ["l1_norm", "mean_abs_weight"]:
            group_total = sum(results[layer][metric] for layer in group_layers)
            total = sum(results[layer][metric] for layer in layers)
            pct = 100 * group_total / total if total > 0 else 0
            print(f"  {metric}: {group_total:.6f} ({pct:.1f}% of total)")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze layer-wise feature importance in trained carry probe"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/carry_probe/2026-03-10_120616/checkpoints/best_probe.pt",
        help="Path to probe checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as checkpoint dir)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load probe on",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of top features to consider per layer",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent.parent / "layer_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading probe from: {checkpoint_path}")
    probe = load_probe(checkpoint_path, device=args.device)

    print(f"\nAnalyzing layer importance (top_k={args.top_k})...")
    results = analyze_layer_importance(probe, top_k=args.top_k)

    print("\nGenerating plots...")
    plot_all_metrics(results, output_dir)

    for metric in ["l1_norm", "l2_norm", "mean_abs_weight", "top_k_sum"]:
        plot_layer_importance(results, output_dir, metric=metric)

    print_summary(results)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
