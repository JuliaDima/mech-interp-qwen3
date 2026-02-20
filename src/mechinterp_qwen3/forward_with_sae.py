"""Forward pass with SAE feature extraction.

Runs model forward pass while capturing MLP activations and extracting
SAE features using transcoders.
"""

from __future__ import annotations

from typing import Any

import torch
from nnsight import LanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from .hooks import LinearizedHookManager
from .transcoder import SingleLayerTranscoder as Transcoder


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

    nn_model = LanguageModel(model, dispatch=True)

    with torch.set_grad_enabled(True), nn_model.trace(input_ids):
        # Extract MLP activations from hooks
        mlp_activations_proxy = {}
        for layer_id in layers_to_analyze:
            layer_module = nn_model.model.layers[layer_id]
            # Save mlp input proxy (we need this for the encoder)
            mlp_in_proxy = layer_module.mlp.input[0][0].save()
            # Save mlp output proxy (for manual dictionary)
            mlp_out_proxy = layer_module.mlp.output.save()

            mlp_activations_proxy[layer_id] = (mlp_in_proxy, mlp_out_proxy)

        # Output logits
        logits_proxy = nn_model.lm_head.output.save()

    logits = logits_proxy.value[0]  # [seq_len, vocab_size]

    # Extract MLP activations from hooks
    mlp_activations = {}
    mlp_in_saved = {}
    for layer_id in layers_to_analyze:
        mlp_in_proxy, mlp_out_proxy = mlp_activations_proxy[layer_id]

        # Unpack tuple if NNSight returned it
        val_out = mlp_out_proxy.value
        mlp_out = val_out[0] if isinstance(val_out, tuple) else val_out

        val_in = mlp_in_proxy.value
        mlp_in = val_in[0] if isinstance(val_in, tuple) else val_in

        mlp_activations[layer_id] = mlp_out
        mlp_in_saved[layer_id] = mlp_in

        mlp_activations[layer_id].retain_grad()

    # Extract SAE features using transcoders
    sae_features = {}
    for layer_id in layers_to_analyze:
        transcoder = transcoders[layer_id]
        # Use MLP Input for encoding!
        mlp_in = mlp_in_saved[layer_id]

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
    *,
    use_patching: bool = False,
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
        use_patching: If True, replaces MLP output with SAE reconstruction
                      to enable inter-layer feature connectivity.

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

    nn_model = LanguageModel(model, dispatch=True)

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

    # Declare before `with` to avoid Python's UnboundLocalError if the block raises.
    sae_features_proxy: dict = {}
    mlp_activations_proxy: dict = {}
    embed_proxy = None
    pre_logit_proxy = None
    logits_proxy = None

    # Forward pass with gradients AND monkey-patched attention
    with (
        torch.set_grad_enabled(True),
        patch_attention(),
        nn_model.trace(inputs["input_ids"]),
    ):
        # NNSight requires saves to be registered in execution order.
        # embed_tokens runs first, layers run next, then norm/lm_head.
        embed_proxy = nn_model.model.embed_tokens.output.save()

        for layer_id in layers_to_analyze:
            layer_module = nn_model.model.layers[layer_id]
            transcoder = transcoders[layer_id]

            # mlp.input is a tuple of (hidden_states, ...), we want the first element
            mlp_in = layer_module.mlp.input[0][0]

            # Encode features natively in the NNSight graph
            features = transcoder.encode(mlp_in.unsqueeze(0)).squeeze(0)

            if use_patching:
                reconstruction = transcoder.decode(
                    features.unsqueeze(0), mlp_in.unsqueeze(0)
                ).squeeze(0)
                # Patch the MLP output directly!
                layer_module.mlp.output = reconstruction

            # Save the node proxies for later extraction
            sae_features_proxy[layer_id] = features.save()
            mlp_activations_proxy[layer_id] = (mlp_in.save(), layer_module.mlp.output.save())

        pre_logit_proxy = nn_model.model.norm.output.save()
        logits_proxy = nn_model.lm_head.output.save()

    def _unpack_proxy(proxy):
        """Unpack an NNSight SaveProxy or a plain tensor."""
        val = proxy.value if hasattr(proxy, "value") else proxy
        return val[0] if isinstance(val, tuple) else val

    logits = _unpack_proxy(logits_proxy)

    # Extract real tensors from NNSight proxies
    mlp_activations = {}
    sae_features = {}
    mlp_in_saved = {}

    for layer_id in layers_to_analyze:
        mlp_in_proxy, mlp_out_proxy = mlp_activations_proxy[layer_id]

        mlp_out = _unpack_proxy(mlp_out_proxy)
        mlp_in = _unpack_proxy(mlp_in_proxy)

        # NNSight retains grad natively into these tensors
        mlp_activations[layer_id] = mlp_out.unsqueeze(
            0
        )  # Make it [1, seq_len, d_model] for compute_attribution
        mlp_activations[layer_id].retain_grad()
        mlp_in_saved[layer_id] = mlp_in

        # Process SAE Features
        features = _unpack_proxy(sae_features_proxy[layer_id])

        # Convert to sparse if highly sparse (>80% zeros) for memory efficiency
        sparsity = 1.0 - (features.count_nonzero().item() / features.numel())
        if sparsity > 0.8:
            features = features.to_sparse()

        features.retain_grad()
        sae_features[layer_id] = features

    # Get pre-logit hidden state and retain grad
    val_pre_logit = _unpack_proxy(pre_logit_proxy)
    pre_logit = val_pre_logit.unsqueeze(0)
    pre_logit.retain_grad()

    embed_act = _unpack_proxy(embed_proxy)
    embed_act.retain_grad()

    lin_hooks.remove()

    return {
        "input_ids": input_ids[0],
        "tokens": tokens,
        "logits": logits,
        "sae_features": sae_features,
        "mlp_activations": mlp_activations,
        "pre_logit_hidden": pre_logit,
        "embedding_activations": embed_act,
    }
