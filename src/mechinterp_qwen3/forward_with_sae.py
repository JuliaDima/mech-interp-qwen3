"""Forward pass with SAE feature extraction.

Runs model forward pass while capturing MLP activations and extracting
SAE features using transcoders.
"""

from __future__ import annotations

from typing import Any

import torch
from circuit_tracer.transcoder import SingleLayerTranscoder as Transcoder
from transformers import PreTrainedModel, PreTrainedTokenizer

from .hooks import LayerActs, LinearizedHookManager, MLPHookManager


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


def forward_linearized_with_sae_features(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    transcoders: dict[int, Transcoder],
    prompt: str,
    layers_to_analyze: list[int],
) -> dict[str, Any]:
    """Forward pass with linearized gradient flow and SAE feature extraction.

    Uses LinearizedHookManager for proper gradient flow matching the
    Attribution Graphs paper methodology:
    - Embedding gradients enabled
    - Attention outputs detached (gradients flow through residual skip only)
    - RMSNorm scale factors treated as constant in backward pass

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        transcoders: Dictionary mapping layer_id -> Transcoder
        prompt: Input prompt text
        layers_to_analyze: List of layer IDs to extract features from

    Returns:
        Dict with: tokens, logits, sae_features, mlp_activations, pre_logit_hidden
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode([tok_id]) for tok_id in input_ids[0]]

    # Install linearized gradient hooks (embed grad, attn detach, LN freeze)
    lin_hooks = LinearizedHookManager(model)
    lin_hooks.install()

    # Install MLP hooks (with gradients preserved)
    mlp_hooks = MLPHookManager(model, layer_ids=layers_to_analyze, detach=False)
    mlp_hooks.install()

    # Hook on final norm to capture pre-logit hidden state
    pre_logit_hidden = {}

    def final_norm_hook(module, input, output):
        pre_logit_hidden["value"] = output

    final_norm_handle = model.model.norm.register_forward_hook(final_norm_hook)

    # Forward pass with gradients
    with torch.set_grad_enabled(True):
        outputs = model(**inputs)
        logits = outputs.logits[0]  # [seq_len, vocab_size]

    # Extract MLP activations
    mlp_activations = {}
    for layer_id in layers_to_analyze:
        layer_acts: LayerActs = mlp_hooks.cache[layer_id]
        if layer_acts.mlp_out is None:
            raise RuntimeError(f"No MLP activations captured for layer {layer_id}")
        mlp_activations[layer_id] = layer_acts.mlp_out
        mlp_activations[layer_id].retain_grad()

    # Get pre-logit hidden state and retain grad
    pre_logit = pre_logit_hidden["value"]
    pre_logit.retain_grad()

    # Remove all hooks
    final_norm_handle.remove()
    mlp_hooks.remove()
    lin_hooks.remove()

    # Extract SAE features using transcoders
    sae_features = {}
    for layer_id in layers_to_analyze:
        transcoder = transcoders[layer_id]
        mlp_acts = mlp_activations[layer_id]
        features = transcoder.encode(mlp_acts)
        features = features.squeeze(0)
        features.retain_grad()
        sae_features[layer_id] = features

    return {
        "input_ids": input_ids[0],
        "tokens": tokens,
        "logits": logits,
        "sae_features": sae_features,
        "mlp_activations": mlp_activations,
        "pre_logit_hidden": pre_logit,
    }
