"""Load and manage Qwen3-4B transcoders for SAE feature extraction."""

from __future__ import annotations

from pathlib import Path

import torch
from circuit_tracer.transcoder import Transcoder

DEFAULT_TRANSCODER_REPO = "mwhanna/qwen3-4b-transcoders"


def load_transcoder(
    transcoder_repo: str = DEFAULT_TRANSCODER_REPO,
    layer_id: int = 0,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> Transcoder:
    """
    Load a single transcoder for a specific layer.

    Args:
        transcoder_repo: HuggingFace repo containing transcoders
        layer_id: Layer number to load transcoder for
        device: Device to load transcoder on ('cpu' or 'cuda')
        cache_dir: Optional cache directory for transcoder weights

    Returns:
        Loaded transcoder instance
    """
    transcoder = Transcoder.from_pretrained(
        transcoder_repo,
        layer=layer_id,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    transcoder = transcoder.to(device)
    transcoder.eval()

    return transcoder


def load_transcoders_for_layers(
    layer_ids: list[int],
    transcoder_repo: str = DEFAULT_TRANSCODER_REPO,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> dict[int, Transcoder]:
    """
    Load transcoders for multiple layers.

    Args:
        layer_ids: List of layer IDs to load transcoders for
        transcoder_repo: HuggingFace repo containing transcoders
        device: Device to load transcoders on
        cache_dir: Optional cache directory

    Returns:
        Dictionary mapping layer_id -> Transcoder
    """
    transcoders = {}

    for layer_id in layer_ids:
        print(f"Loading transcoder for layer {layer_id}...")
        transcoder = load_transcoder(
            transcoder_repo=transcoder_repo,
            layer_id=layer_id,
            device=device,
            cache_dir=cache_dir,
        )
        transcoders[layer_id] = transcoder

    return transcoders


@torch.no_grad()
def extract_sae_features(
    activations: torch.Tensor,
    transcoder: Transcoder,
) -> torch.Tensor:
    """
    Extract SAE features from MLP activations using a transcoder.

    Args:
        activations: MLP activations [seq_len, d_model]
        transcoder: Loaded transcoder instance

    Returns:
        SAE feature activations [seq_len, n_features]
    """
    # Ensure activations are on same device as transcoder
    device = next(transcoder.parameters()).device
    activations = activations.to(device)

    # Add batch dimension if needed: [seq_len, d_model] -> [1, seq_len, d_model]
    if activations.dim() == 2:
        activations = activations.unsqueeze(0)

    # Extract features using transcoder's encoder
    # Transcoders typically have an encoder that maps activations -> SAE features
    sae_features = transcoder.encode(activations)

    # Remove batch dimension: [1, seq_len, n_features] -> [seq_len, n_features]
    if sae_features.dim() == 3 and sae_features.shape[0] == 1:
        sae_features = sae_features.squeeze(0)

    return sae_features.cpu()


def get_transcoder_info(transcoder: Transcoder) -> dict[str, any]:
    """
    Get information about a transcoder.

    Args:
        transcoder: Loaded transcoder instance

    Returns:
        Dictionary with transcoder metadata
    """
    info = {
        "n_features": transcoder.n_dict_components
        if hasattr(transcoder, "n_dict_components")
        else None,
        "d_model": transcoder.d_model if hasattr(transcoder, "d_model") else None,
        "device": str(next(transcoder.parameters()).device),
    }

    return info
