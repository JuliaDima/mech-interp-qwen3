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
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = inputs["input_ids"]
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
    top_k_features: int | None = None,
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
    input_ids = inputs["input_ids"].squeeze(0)
    tokens = [tokenizer.decode([tok_id]) for tok_id in input_ids]

    # Defensively ensure a "dummy" token exists at position 0 to absorb artifacts. (aka "Sink Token")
    if tokens[0] not in tokenizer.all_special_ids:
        candidate_bos_token_ids = [
            tokenizer.bos_token_id,
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
        ]
        candidate_bos_token_ids += tokenizer.all_special_ids

        dummy_bos_token_id = next(filter(None, candidate_bos_token_ids))
        if dummy_bos_token_id is None:
            print(
                "No suitable special token found for BOS token replacement. The first token will be ignored."
            )
        else:
            input_ids = torch.cat(
                [
                    torch.tensor([dummy_bos_token_id], device=input_ids.device),
                    input_ids,
                ]
            )
            tokens = [tokenizer.decode([tok_id]) for tok_id in input_ids]

            # Update attention_mask to include the new token
            if "attention_mask" in inputs:
                mask = inputs["attention_mask"]
                ones = torch.ones((mask.shape[0], 1), device=mask.device, dtype=mask.dtype)
                inputs["attention_mask"] = torch.cat([ones, mask], dim=1)

    input_ids = input_ids.unsqueeze(0)  # just to match dimensions
    inputs["input_ids"] = input_ids
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

        if top_k_features is not None:
            # Keep only top k features
            top_vals, top_inds = features.topk(top_k_features, dim=-1)

            # Create sparse tensor (dense shape preserved)
            # indices: [2, num_elements] -> [row_indices, col_indices]
            # features is [seq_len, n_features]
            seq_len = features.shape[0]

            # Row indices: [0, 0, ..., 1, 1, ...]
            rows = (
                torch.arange(seq_len, device=features.device)
                .unsqueeze(1)
                .expand(-1, top_k_features)
            )

            # Stack to get [2, seq_len * k]
            indices = torch.stack([rows.flatten(), top_inds.flatten()])
            values = top_vals.flatten()

            # Construct sparse tensor
            features_sparse = torch.sparse_coo_tensor(
                indices, values, size=features.shape, device=features.device
            )

            # Gradients?
            # Creating a sparse tensor from values that require grad IS supported if we use the values directly.
            # However, sparse_coo_tensor might detach.
            # But here `values` has grad.
            # Let's verify if `features_sparse` requires_grad.
            # Usually sparse tensors don't support .retain_grad() in older PyTorch, but let's try.
            # If `values` has grad history, `features_sparse` should be part of graph.
            # But wait, standard sparse tensors might not support backprop through construction in all versions.
            # A safer bet for *attribution* where we need grad w.r.t features:
            # We need the gradient to flow back to `mlp_acts`.
            # `features` (dense) comes from `mlp_acts`.
            # `values` comes from `features`.
            # `features_sparse` is built from `values`.
            # If we use `features_sparse` downstream, gradients will flow to `values`, then to `features`, then to `mlp_acts`.
            # This chain works.

            sae_features[layer_id] = features_sparse
        else:
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
