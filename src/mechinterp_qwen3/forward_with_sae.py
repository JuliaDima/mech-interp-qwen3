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


def forward_linearized_with_sae_features(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    transcoders: dict[int, Transcoder],
    prompt: str,
    layers_to_analyze: list[int],
    *,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Forward pass with linearized gradient flow and SAE feature extraction.

    Implements the local replacement model for the Attribution Graphs paper:
    - Embedding gradients enabled (gradients can flow back to token positions)
    - Attention weights detached inside the NNSight trace (gradients flow
      through the residual stream and the OV circuit, but NOT through QK)
    - RMSNorm scale factors treated as constant in backward pass

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        transcoders: Dictionary mapping layer_id -> Transcoder
        prompt: Input prompt text
        layers_to_analyze: List of layer IDs to extract features from
        feature_to_feature_edges: If True, replaces MLP output with SAE reconstruction
                      to enable inter-layer feature connectivity.

    Returns:
        Dict with: tokens, logits, sae_features, mlp_activations, pre_logit_hidden,
                   embedding_activations
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"].squeeze(0)
    tokens = [tokenizer.decode([tok_id]) for tok_id in input_ids]  # [seq_len]

    # Defensively ensure a "dummy" token exists at position 0 to absorb artifacts. (aka "sink token")
    if input_ids[0].item() not in tokenizer.all_special_ids:
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

    input_ids = input_ids.unsqueeze(0)  # [1, seq_len]
    inputs["input_ids"] = input_ids

    # Install RMSNorm linearization hooks (treat scale denominator as constant).
    # These use PyTorch forward hooks to recompute norm output with a frozen scale,
    # which is mathematically more precise than detaching the full norm output.
    lin_hooks = LinearizedHookManager(model)
    lin_hooks.install()

    # Force eager attention so NNSight can trace internal sub-nodes
    # (attention_interface_0.source.nn_functional_dropout_0).  Flash/SDPA
    # kernels don't expose those nodes. We restore the original impl after.
    original_attn_impl = model.config._attn_implementation
    model.config._attn_implementation = "eager"

    nn_model = LanguageModel(model, dispatch=True)

    # Declare before `with` to avoid Python's UnboundLocalError if the block raises.
    sae_features_proxy: dict = {}
    mlp_activations_proxy: dict = {}
    embed_proxy = None
    pre_logit_proxy = None
    logits_proxy = None
    analyze_set = set(layers_to_analyze)

    with torch.no_grad(), nn_model.trace(inputs["input_ids"]):
        for layer_id in sorted(analyze_set):
            layer_module = nn_model.model.layers[layer_id]
            mlp_in = layer_module.mlp.input[0]
            features = transcoders[layer_id].encode(mlp_in)
            sae_features_proxy[layer_id] = features.save()

        logits_proxy = nn_model.lm_head.output.save()

    def _unpack_proxy(proxy):
        """Unpack an NNSight SaveProxy or a plain tensor."""
        val = proxy.value if hasattr(proxy, "value") else proxy
        return val[0] if isinstance(val, tuple) else val

    # Extract and sparsify features
    sae_features = {}
    sae_features_dense = {}
    for layer_id in layers_to_analyze:
        features = _unpack_proxy(sae_features_proxy[layer_id])
        if features.dim() == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        elif features.dim() == 1:
            raise RuntimeError(f"SAE features for layer {layer_id} are unexpectedly 1D")

        sae_features_dense[layer_id] = features.clone()

        sparsity = 1.0 - (features.count_nonzero().item() / features.numel())
        if sparsity > 0.8:
            features = features.to_sparse()
        sae_features[layer_id] = features

    # ── Trace 2: Gradient Flow (batch_size = N) ──
    batched_input_ids = inputs["input_ids"].expand(batch_size, -1)

    mlp_inputs_proxy: dict = {}

    with torch.set_grad_enabled(True), nn_model.trace(batched_input_ids):
        # Linearization 1: Enable gradients on the embedding output
        embed_out = nn_model.model.embed_tokens.output
        embed_out.requires_grad_(True)
        embed_proxy = embed_out.save()

        # Linearization 2: Detach QK and MLPs, retain grad on mlp_in and mlp_out
        for layer_id in range(len(nn_model.model.layers)):
            layer_module = nn_model.model.layers[layer_id]

            # Detach post-softmax attention weights
            attn_weights_node = (
                layer_module.self_attn.source.attention_interface_0.source.nn_functional_dropout_0
            )
            attn_weights_node.output = attn_weights_node.output.detach()

            if layer_id in analyze_set:
                mlp_in_node = layer_module.mlp.input[0]
                mlp_in_node.retain_grad()
                mlp_inputs_proxy[layer_id] = mlp_in_node.save()

            # Detach MLP output so gradients don't flow through MLP weights
            mlp_out_node = layer_module.mlp.output
            detached_out = mlp_out_node.detach().requires_grad_(True)
            layer_module.mlp.output = detached_out

            if layer_id in analyze_set:
                detached_out.retain_grad()
                mlp_activations_proxy[layer_id] = detached_out.save()

        pre_logit_out = nn_model.model.norm.output
        pre_logit_out.retain_grad()
        pre_logit_proxy = pre_logit_out.save()

    def _unpack_proxy(proxy):
        """Unpack an NNSight SaveProxy or a plain tensor."""
        val = proxy.value if hasattr(proxy, "value") else proxy
        res = val[0] if isinstance(val, tuple) else val
        return res

    logits = _unpack_proxy(logits_proxy)
    if logits.dim() == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)

    # Extract real tensors from NNSight proxies
    mlp_activations = {}
    mlp_inputs = {}

    for layer_id in layers_to_analyze:
        m_in = _unpack_proxy(mlp_inputs_proxy[layer_id])
        m_out = _unpack_proxy(mlp_activations_proxy[layer_id])

        # Ensure batch dim
        if m_in.dim() == 2:
            m_in = m_in.unsqueeze(0)
        if m_out.dim() == 2:
            m_out = m_out.unsqueeze(0)

        m_in.retain_grad()
        m_out.retain_grad()
        mlp_inputs[layer_id] = m_in
        mlp_activations[layer_id] = m_out

    # Get pre-logit hidden state and retain grad
    val_pre_logit = _unpack_proxy(pre_logit_proxy)
    pre_logit = val_pre_logit.unsqueeze(0) if val_pre_logit.dim() == 2 else val_pre_logit
    pre_logit.retain_grad()

    embed_act = _unpack_proxy(embed_proxy)
    if embed_act.dim() == 2:
        embed_act = embed_act.unsqueeze(0)
    embed_act.retain_grad()

    lin_hooks.remove()
    model.config._attn_implementation = original_attn_impl

    return {
        "input_ids": input_ids[0],
        "tokens": tokens,
        "logits": logits,
        "sae_features": sae_features,
        "mlp_inputs": mlp_inputs,
        "mlp_activations": mlp_activations,
        "pre_logit_hidden": pre_logit,
        "embedding_activations": embed_act,
    }
