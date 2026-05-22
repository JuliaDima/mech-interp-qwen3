"""Feature importance analysis utilities for interpreting probe weights."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .carry_probe import CarryProbe


def analyze_feature_importance(
    probe: CarryProbe,
    top_k: int = 50,
    save_dir: Path | None = None,
) -> dict[int, pd.DataFrame]:
    """Analyze feature importance across all probe layers.

    Args:
        probe: Trained CarryProbe instance
        top_k: Number of top features to extract per layer
        save_dir: If not None, save analysis results to this directory

    Returns:
        Dictionary mapping layer index to DataFrame with top features
    """
    results = {}

    for layer in probe.layers:
        # Get full feature rankings
        indices, weights, abs_weights = probe.get_feature_rankings(layer)

        # Create DataFrame for top-k features
        df = pd.DataFrame(
            {
                "feature_id": indices[:top_k].cpu().numpy(),
                "weight": weights[:top_k].cpu().detach().float().numpy(),
                "abs_weight": abs_weights[:top_k].cpu().detach().float().numpy(),
                "rank": range(1, top_k + 1),
            }
        )

        results[layer] = df

        # Save if requested
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save as CSV
            csv_path = save_dir / f"layer_{layer}_top_features.csv"
            df.to_csv(csv_path, index=False)

            # Also save top positive and negative features separately
            pos_indices, pos_weights = probe.get_top_features(layer, k=top_k, by="positive")
            neg_indices, neg_weights = probe.get_top_features(layer, k=top_k, by="negative")

            pos_df = pd.DataFrame(
                {
                    "feature_id": pos_indices.cpu().numpy(),
                    "weight": pos_weights.cpu().detach().float().numpy(),
                }
            )
            neg_df = pd.DataFrame(
                {
                    "feature_id": neg_indices.cpu().numpy(),
                    "weight": neg_weights.cpu().detach().float().numpy(),
                }
            )

            pos_df.to_csv(save_dir / f"layer_{layer}_positive_features.csv", index=False)
            neg_df.to_csv(save_dir / f"layer_{layer}_negative_features.csv", index=False)

    # Save summary
    if save_dir is not None:
        summary = {
            "layers": probe.layers,
            "d_transcoder": probe.d_transcoder,
            "top_k": top_k,
            "n_nonzero_weights_per_layer": {
                layer: int((probe.get_layer_weights(layer) != 0).sum().item())
                for layer in probe.layers
            },
        }

        with open(save_dir / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return results


def print_top_features(probe: CarryProbe, layer: int, k: int = 20):
    """Print top features for a layer in a readable format.

    Args:
        probe: Trained CarryProbe instance
        layer: Layer index
        k: Number of features to print
    """
    print(f"\nLayer {layer} - Top {k} Features by Absolute Weight")
    print("=" * 70)

    indices, weights, abs_weights = probe.get_feature_rankings(layer)

    for i in range(min(k, len(indices))):
        feat_id = indices[i].item()
        weight = weights[i].item()
        abs_weight = abs_weights[i].item()
        sign = "+" if weight > 0 else "-"

        print(f"Rank {i + 1:3d}: Feature {feat_id:6d}  Weight: {sign}{abs_weight:8.4f}")

    # Also show most positive and most negative
    print(f"\nTop {min(k // 2, 10)} Most Positive Features (pro-carry):")
    print("-" * 70)
    pos_indices, pos_weights = probe.get_top_features(layer, k=min(k // 2, 10), by="positive")
    for i, (feat_id, weight) in enumerate(zip(pos_indices, pos_weights, strict=False)):
        print(f"  {i + 1:2d}. Feature {feat_id.item():6d}  Weight: +{weight.item():.4f}")

    print(f"\nTop {min(k // 2, 10)} Most Negative Features (anti-carry):")
    print("-" * 70)
    neg_indices, neg_weights = probe.get_top_features(layer, k=min(k // 2, 10), by="negative")
    for i, (feat_id, weight) in enumerate(zip(neg_indices, neg_weights, strict=False)):
        print(f"  {i + 1:2d}. Feature {feat_id.item():6d}  Weight: {weight.item():.4f}")


def compute_layer_ablation_impact(
    probe: CarryProbe,
    activations: dict[int, torch.Tensor],
    labels: torch.Tensor,
) -> dict[int, float]:
    """Compute impact of each layer on probe predictions via ablation.

    For each layer, compute how much the probe's predictions change when
    that layer's contribution is zeroed out.

    Args:
        probe: Trained CarryProbe instance
        activations: Dictionary mapping layer to activations [batch, d_transcoder]
        labels: Ground truth labels [batch]

    Returns:
        Dictionary mapping layer index to ablation impact (change in loss)
    """
    probe.eval()

    # Compute baseline predictions
    with torch.no_grad():
        baseline_probs = probe(activations, return_logits=False)
        baseline_loss = torch.nn.functional.binary_cross_entropy(baseline_probs, labels).item()

    impacts = {}

    for layer in probe.layers:
        # Create ablated activations (zero out this layer)
        ablated_activations = {
            L: (torch.zeros_like(activations[L]) if layer == L else activations[L])
            for L in probe.layers
        }

        # Compute predictions with ablation
        with torch.no_grad():
            ablated_probs = probe(ablated_activations, return_logits=False)
            ablated_loss = torch.nn.functional.binary_cross_entropy(ablated_probs, labels).item()

        # Impact is change in loss
        impact = ablated_loss - baseline_loss
        impacts[layer] = impact

    return impacts


def export_feature_rankings_for_all_layers(
    probe: CarryProbe, output_path: Path, format: str = "csv"
):
    """Export complete feature rankings for all layers.

    Args:
        probe: Trained CarryProbe instance
        output_path: Path to save rankings
        format: 'csv', 'json', or 'txt'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rankings = {}

    for layer in probe.layers:
        indices, weights, abs_weights = probe.get_feature_rankings(layer)

        all_rankings[f"layer_{layer}"] = {
            "feature_ids": indices.cpu().numpy().tolist(),
            "weights": weights.cpu().detach().float().numpy().tolist(),
            "abs_weights": abs_weights.cpu().detach().float().numpy().tolist(),
        }

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(all_rankings, f, indent=2)

    elif format == "csv":
        # Save as one CSV with layer column
        rows = []
        for layer in probe.layers:
            indices, weights, abs_weights = probe.get_feature_rankings(layer)
            for i, (idx, w, aw) in enumerate(zip(indices, weights, abs_weights, strict=False)):
                rows.append(
                    {
                        "layer": layer,
                        "rank": i + 1,
                        "feature_id": idx.item(),
                        "weight": w.item(),
                        "abs_weight": aw.item(),
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    elif format == "txt":
        with open(output_path, "w") as f:
            for layer in probe.layers:
                f.write(f"\n{'=' * 70}\n")
                f.write(f"Layer {layer}\n")
                f.write(f"{'=' * 70}\n\n")

                indices, weights, abs_weights = probe.get_feature_rankings(layer)

                f.write(f"{'Rank':<6} {'Feature ID':<12} {'Weight':<12} {'|Weight|':<12}\n")
                f.write(f"{'-' * 50}\n")

                for i, (idx, w, aw) in enumerate(zip(indices, weights, abs_weights, strict=False)):
                    f.write(f"{i + 1:<6} {idx.item():<12} {w.item():<12.6f} {aw.item():<12.6f}\n")

    else:
        raise ValueError(f"Unknown format: {format}")
