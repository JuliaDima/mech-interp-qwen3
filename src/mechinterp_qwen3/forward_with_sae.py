"""Forward pass with SAE feature extraction.

Runs model forward pass while capturing MLP activations and extracting
SAE features using transcoders.
"""

from __future__ import annotations

from typing import Any

import torch
from circuit_tracer.transcoder import SingleLayerTranscoder as Transcoder
from transformers import PreTrainedModel, PreTrainedTokenizer

from .hooks import LayerActs, MLPHookManager


@torch.no_grad()
def forward_with_sae_features(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    transcoders: dict[int, Transcoder],
    prompt: str,
    layers_to_analyze: list[int],
) -> dict[str, Any]:
    """
    Run forward pass and extract SAE features at specified layers.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        transcoders: Dictionary mapping layer_id -> Transcoder
        prompt: Input prompt text
        layers_to_analyze: List of layer IDs to extract features from

    Returns:
        Dictionary containing:
            - 'input_ids': Input token IDs [seq_len]
            - 'tokens': List of token strings
            - 'logits': Output logits [seq_len, vocab_size]
            - 'sae_features': {layer_id: feature_activations [seq_len, n_features]}
            - 'mlp_activations': {layer_id: mlp_acts [seq_len, d_model]}
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    tokens = [tokenizer.decode([tok_id]) for tok_id in input_ids[0]]

    # Install hooks to capture MLP activations
    hooker = MLPHookManager(model, layer_ids=layers_to_analyze)
    hooker.install()

    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits[0]  # [seq_len, vocab_size]

    # Extract MLP activations from hooks
    mlp_activations = {}
    for layer_id in layers_to_analyze:
        layer_acts: LayerActs = hooker.cache[layer_id]
        if layer_acts.mlp_out is None:
            raise RuntimeError(f"No MLP activations captured for layer {layer_id}")
        # mlp_out shape: [batch=1, seq_len, d_model] -> [seq_len, d_model]
        mlp_activations[layer_id] = layer_acts.mlp_out.squeeze(0)

    hooker.remove()

    # Extract SAE features using transcoders
    sae_features = {}
    for layer_id in layers_to_analyze:
        transcoder = transcoders[layer_id]
        mlp_acts = mlp_activations[layer_id]  # [seq_len, d_model]

        # Add batch dimension for transcoder
        mlp_acts_batched = mlp_acts.unsqueeze(0)  # [1, seq_len, d_model]

        # Encode to get SAE features
        features = transcoder.encode(mlp_acts_batched)  # [1, seq_len, n_features]
        features = features.squeeze(0)  # [seq_len, n_features]

        sae_features[layer_id] = features

    return {
        "input_ids": input_ids[0],  # [seq_len]
        "tokens": tokens,
        "logits": logits,  # [seq_len, vocab_size]
        "sae_features": sae_features,  # {layer_id: [seq_len, n_features]}
        "mlp_activations": mlp_activations,  # {layer_id: [seq_len, d_model]}
    }


def forward_with_sae_features_grad(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    transcoders: dict[int, Transcoder],
    prompt: str,
    layers_to_analyze: list[int],
) -> dict[str, Any]:
    """
    Forward pass with SAE features, keeping gradients enabled.

    Same as forward_with_sae_features but with gradients enabled for attribution.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        transcoders: Dictionary mapping layer_id -> Transcoder
        prompt: Input prompt text
        layers_to_analyze: List of layer IDs to extract features from

    Returns:
        Same as forward_with_sae_features, but tensors have gradients
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    tokens = [tokenizer.decode([tok_id]) for tok_id in input_ids[0]]

    # Install hooks to capture MLP activations (with gradients!)
    hooker = MLPHookManager(model, layer_ids=layers_to_analyze, detach=False)
    hooker.install()

    # Forward pass with gradients explicitly enabled
    # This ensures intermediate activations have requires_grad=True
    with torch.set_grad_enabled(True):
        outputs = model(**inputs)
        logits = outputs.logits[0]  # [seq_len, vocab_size]

    # Extract MLP activations from hooks
    mlp_activations = {}
    for layer_id in layers_to_analyze:
        layer_acts: LayerActs = hooker.cache[layer_id]
        if layer_acts.mlp_out is None:
            raise RuntimeError(f"No MLP activations captured for layer {layer_id}")
        mlp_activations[layer_id] = layer_acts.mlp_out

        # Retain gradients on the original tensor used in computation
        mlp_activations[layer_id].retain_grad()

    hooker.remove()

    # Extract SAE features using transcoders
    sae_features = {}
    for layer_id in layers_to_analyze:
        transcoder = transcoders[layer_id]
        mlp_acts = mlp_activations[layer_id]

        # Already batched [1, seq_len, d_model]
        mlp_acts_batched = mlp_acts

        # Encode to get SAE features (with gradients)
        features = transcoder.encode(mlp_acts_batched)
        features = features.squeeze(0)

        # Retain gradients for non-leaf tensors (critical for attribution!)
        features.retain_grad()

        sae_features[layer_id] = features

    return {
        "input_ids": input_ids[0],
        "tokens": tokens,
        "logits": logits,
        "sae_features": sae_features,
        "mlp_activations": mlp_activations,
    }
