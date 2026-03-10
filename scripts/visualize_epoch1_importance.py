#!/usr/bin/env python3
"""Visualize layer importance from a specific probe checkpoint."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from mechinterp_qwen3.probe import CarryProbe


def load_probe(checkpoint_path: Path, device: str = "cpu") -> CarryProbe:
    """Load a trained probe from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    probe_data = checkpoint.get("probe", checkpoint)
    probe_state = probe_data.get("state_dict", probe_data)
    layers = probe_data.get("layers")
    d_transcoder = probe_data.get("d_transcoder", 163840)
    max_seq_len = probe_data.get("max_seq_len", 0)

    if (linear_weight := probe_state.get("linear.weight")) is not None:
        total_features = linear_weight.shape[1]
        n_layers = total_features // d_transcoder
        if layers is None:
            layers = list(range(n_layers))
    else:
        raise ValueError("Could not find weights in checkpoint")

    probe = CarryProbe(
        layers=layers,
        d_transcoder=d_transcoder,
        max_seq_len=max_seq_len,
        device=torch.device(device),
        dtype=linear_weight.dtype,
    )
    probe.load_state_dict(probe_state)
    return probe


def main():
    parser = argparse.ArgumentParser(description="Visualize layer importance for a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found")
        return

    probe = load_probe(checkpoint_path)

    importance = {}
    for layer in probe.layers:
        weights = probe.get_layer_weights(layer)
        importance[layer] = torch.norm(weights, p=1).item()

    layers = sorted(importance.keys())
    values = [importance[layer] for layer in layers]

    plt.figure(figsize=(12, 6))
    plt.bar(layers, values, color="skyblue", edgecolor="navy")
    plt.xlabel("Layer Index")
    plt.ylabel("Weight L1 Norm")
    plt.title(f"Layer-wise Importance (Checkpoint: {checkpoint_path.name})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Highlight highest importance layer
    max_val = max(values)
    max_layer = layers[values.index(max_val)]
    plt.annotate(
        f"Max: Layer {max_layer}",
        xy=(max_layer, max_val),
        xytext=(max_layer, max_val * 1.05),
        ha="center",
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    plt.tight_layout()
    output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_importance.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved importance plot to: {output_path}")


if __name__ == "__main__":
    main()
