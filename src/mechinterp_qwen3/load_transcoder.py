"""Load and manage Qwen3-4B transcoders for SAE feature extraction."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .transcoder import SingleLayerTranscoder

DEFAULT_TRANSCODER_REPO = "mwhanna/qwen3-4b-transcoders"


def load_transcoder(
    transcoder_repo: str = DEFAULT_TRANSCODER_REPO,
    layer_id: int = 0,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> SingleLayerTranscoder:
    """
    Load a single transcoder for a specific layer.

    Args:
        transcoder_repo: HuggingFace repo containing transcoders
        layer_id: Layer number to load transcoder for
        device: Device to load transcoder on ('cpu' or 'cuda')\n        cache_dir: Optional cache directory for transcoder weights

    Returns:
        Loaded transcoder instance
    """
    transcoder_path = hf_hub_download(
        repo_id=transcoder_repo,
        filename=f"layer_{layer_id}.safetensors",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    state_dict = load_file(transcoder_path, device=device)

    # Extract dimensions from the state dict
    # W_enc shape: [d_transcoder, d_model], W_dec shape: [d_transcoder, d_model]
    d_transcoder, d_model = state_dict["W_enc"].shape

    # Create a SingleLayerTranscoder instance with the correct parameters
    # activation_function is typically ReLU for transcoders
    import torch.nn as nn

    transcoder = SingleLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=nn.ReLU(),  # Anthropic (March 2025) uses JumpReLU, however mwhanna/qwen3-4b-transcoders uses ReLU
        layer_idx=layer_id,
        device=device,
    )

    # Load the state dict into the transcoder
    transcoder.load_state_dict(state_dict)

    # Set to eval mode
    transcoder.eval()

    return transcoder


def load_transcoders_for_layers(
    layer_ids: list[int],
    transcoder_repo: str = DEFAULT_TRANSCODER_REPO,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> dict[int, SingleLayerTranscoder]:
    """
    Load transcoders for multiple layers.

    Args:
        layer_ids: List of layer IDs to load transcoders for
        transcoder_repo: HuggingFace repo containing transcoders
        device: Device to load transcoders on
        cache_dir: Optional cache directory

    Returns:
        Dictionary mapping layer_id -> SingleLayerTranscoder
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
