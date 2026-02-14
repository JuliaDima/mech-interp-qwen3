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
        if layer_acts.mlp_out is None or layer_acts.mlp_in is None:
            raise RuntimeError(f"No MLP activations captured for layer {layer_id}")
        mlp_activations[layer_id] = layer_acts.mlp_out

        # We need mlp_in for the transcoder!
        # layer_acts.mlp_in is [seq, d_model] (detached or not depends on manager)
        # But wait, MLPHookManager only saves mlp_in if we asked it to?
        # In hooks.py, pre_hook saves mlp_in.

        # Retain gradients on the original tensor used in computation
        mlp_activations[layer_id].retain_grad()

        # Also retain grad for mlp_in if we are going to use it?
        # mlp_in is an intermediate activation.
        # But we are in a torch.set_grad_enabled(True) block.
        # layer_acts.mlp_in should have grad_fn if it's from the graph.
        # We should check if we need to retain grad on it.
        # The transcoder path will be a branch off mlp_in.
        # If we don't retain grad, `mlp_in.grad` will be None, but backprop will still work through it to earlier layers?
        # Yes. We only need retain_grad if we want to INSPECT the gradient at mlp_in.
        # We don't need to inspect it here.
        # But we DO need to use it for encoding.

    hooker.remove()

    # Extract SAE features using transcoders
    sae_features = {}
    for layer_id in layers_to_analyze:
        transcoder = transcoders[layer_id]
        # Use MLP Input for encoding!
        mlp_in = hooker.cache[layer_id].mlp_in

        # Already batched [1, seq_len, d_model]?
        # MLPHookManager saves [seq, d_model] usually (x[0]).
        # Let's check hooks.py: `self.cache[lid].mlp_in = x[0]`.
        # So it is [seq, d_model].

        # Encode to get SAE features (with gradients)
        features = transcoder.encode(mlp_in.unsqueeze(0))  # Expects [batch, seq, d_model] usually?

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
    input_ids = inputs["input_ids"].squeeze(0)
    tokens = [tokenizer.decode([tok_id]) for tok_id in input_ids]  # [seq_len]

    # Defensively ensure a "dummy" token exists at position 0 to absorb artifacts. (aka "sink token")
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
    inputs["input_ids"] = input_ids  # [1, seq_len]
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

    # Hook on embeddings to capture them
    embedding_captured = {}

    def embed_hook(module, input, output):
        embedding_captured["value"] = output

    embed_module = model.get_input_embeddings()
    embed_handle = embed_module.register_forward_hook(embed_hook)

    # --- MONKEY PATCHING ATTENTION ---
    # We need to freeze attention patterns (QK) but allow gradient flow through values (OV).
    # The only way to do this strictly is to patch the attention forward pass.

    import contextlib

    from transformers.models.qwen2 import modeling_qwen2

    # 1. Save original eager attention function
    original_eager_forward = modeling_qwen2.eager_attention_forward

    # 2. Define patched function that detaches attn_weights
    def patched_eager_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        # Call the ORIGINAL logic up to weight computation, but we have to replicate it
        # because the original function does everything in one go.
        # We'll rely on the source code of eager_attention_forward from transformers.

        # Re-implementation of eager_attention_forward with DETACH
        # Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py

        key_states = modeling_qwen2.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_qwen2.repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )

        # --- CRITICAL CHANGE: DETACH ATTENTION WEIGHTS ---
        attn_weights = attn_weights.detach()
        # -------------------------------------------------

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    # Context manager to apply checks
    @contextlib.contextmanager
    def patch_attention():
        # Force eager implementation
        original_attn_implementation = model.config._attn_implementation
        model.config._attn_implementation = "eager"

        # Try to update all layer configs as well, just in case
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "config"):
                layer.self_attn.config._attn_implementation = "eager"

        # Apply patch
        modeling_qwen2.eager_attention_forward = patched_eager_attention_forward
        try:
            yield
        finally:
            # Restore
            modeling_qwen2.eager_attention_forward = original_eager_forward
            model.config._attn_implementation = original_attn_implementation
            for layer in model.model.layers:
                if hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = original_attn_implementation

    # Forward pass with gradients AND monkey-patched attention
    with torch.set_grad_enabled(True), patch_attention():
        # Ensure model uses the new config
        outputs = model(**inputs)
        logits = outputs.logits[0]  # [seq_len, vocab_size]

    # Extract MLP activations
    mlp_activations = {}
    for layer_id in layers_to_analyze:
        layer_acts: LayerActs = mlp_hooks.cache[layer_id]
        if layer_acts.mlp_out is None:
            raise RuntimeError(f"No MLP activations captured for layer {layer_id}")
        mlp_activations[layer_id] = layer_acts.mlp_out  # dict of [1, seq_len, d_model] per key
        mlp_activations[layer_id].retain_grad()

    # Get pre-logit hidden state and retain grad
    pre_logit = pre_logit_hidden["value"]  # [1, seq_len, d_model]
    pre_logit.retain_grad()

    if "value" not in embedding_captured:
        raise RuntimeError("No embedding activations captured.")
    embedding_captured["value"].retain_grad()

    # Remove all hooks
    final_norm_handle.remove()
    embed_handle.remove()
    mlp_hooks.remove()
    lin_hooks.remove()

    # Extract SAE features using transcoders
    sae_features = {}
    for layer_id in layers_to_analyze:
        transcoder = transcoders[layer_id]
        # Use MLP Input for encoding!
        mlp_in = mlp_hooks.cache[layer_id].mlp_in

        if mlp_in is None:
            raise RuntimeError(f"No MLP input captured for layer {layer_id}")

        features = transcoder.encode(mlp_in.unsqueeze(0))  # [1, seq_len, transcoder hidden size]
        features = features.squeeze(0)  # [seq_len, transcoder hidden size]

        # Convert to sparse if highly sparse (>80% zeros) for memory efficiency
        sparsity = 1.0 - (features.count_nonzero().item() / features.numel())
        if sparsity > 0.8:
            features = features.to_sparse()

        features.retain_grad()
        sae_features[layer_id] = features

    return {
        "input_ids": input_ids[0],
        "tokens": tokens,
        "logits": logits,
        "sae_features": sae_features,
        "mlp_activations": mlp_activations,
        "pre_logit_hidden": pre_logit,
        "embedding_activations": embedding_captured["value"],
    }
