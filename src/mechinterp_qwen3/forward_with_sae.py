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
    use_patching: bool = False,
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
        use_patching: If True, replaces MLP output with SAE reconstruction
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

    with torch.set_grad_enabled(True), nn_model.trace(inputs["input_ids"]):
        # ── Linearization 1: Enable gradients on the embedding output ──────────
        embed_out = nn_model.model.embed_tokens.output
        embed_out.requires_grad_(True)
        embed_proxy = embed_out.save()

        # ── Linearization 2 & Feature extraction (single pass per layer) ──────
        # NNSight requires interventions in model-execution order. Merging the
        # attention detach and feature extraction into a single sorted loop prevents
        # OutOfOrderError on mlp.input.
        #
        # Detaching the post-softmax attention here (not self_attn.output) preserves OV-circuit gradients
        # while severing QK, making the model linear in the residual stream.
        analyze_set = set(layers_to_analyze)
        for layer_id in range(len(nn_model.model.layers)):
            layer_module = nn_model.model.layers[layer_id]

            # Detach post-softmax attention weights
            attn_weights_node = (
                layer_module.self_attn.source.attention_interface_0.source.nn_functional_dropout_0
            )
            attn_weights_node.output = attn_weights_node.output.detach()

            if layer_id in analyze_set:
                transcoder = transcoders[layer_id]

                # mlp.input[0] is hidden_states: [1, seq_len, d_model]
                mlp_in = layer_module.mlp.input[0]

                # Encode features natively in the NNSight graph.
                features = transcoder.encode(mlp_in)

                # Retain grads BEFORE .save() — after the trace, .value returns
                # the same tensor object so autograd can still flow back through it.
                mlp_out = layer_module.mlp.output
                mlp_out.retain_grad()
                features.retain_grad()

                if use_patching:
                    reconstruction = transcoder.decode(features, mlp_in)
                    layer_module.mlp.output = reconstruction
                    mlp_out = reconstruction
                    mlp_out.retain_grad()

                sae_features_proxy[layer_id] = features.save()
                mlp_activations_proxy[layer_id] = (mlp_in.save(), mlp_out.save())

        pre_logit_out = nn_model.model.norm.output
        pre_logit_out.retain_grad()
        pre_logit_proxy = pre_logit_out.save()
        logits_proxy = nn_model.lm_head.output.save()

    def _unpack_proxy(proxy):
        """Unpack an NNSight SaveProxy or a plain tensor."""
        val = proxy.value if hasattr(proxy, "value") else proxy
        return val[0] if isinstance(val, tuple) else val

    logits = _unpack_proxy(logits_proxy)
    if logits.dim() == 3:  # [batch, seq_len, vocab_size] -> [seq_len, vocab_size]
        logits = logits.squeeze(0)

    # Extract real tensors from NNSight proxies
    mlp_activations = {}
    sae_features = {}

    for layer_id in layers_to_analyze:
        mlp_in_proxy, mlp_out_proxy = mlp_activations_proxy[layer_id]

        mlp_out = _unpack_proxy(mlp_out_proxy)
        mlp_in = _unpack_proxy(mlp_in_proxy)

        # NNSight proxies already include the batch dim -> only unsqueeze if 2D
        if mlp_out.dim() == 2:  # [seq_len, d_model] -> [1, seq_len, d_model]
            mlp_out = mlp_out.unsqueeze(0)
        mlp_activations[layer_id] = mlp_out  # [1, seq_len, d_model]
        mlp_activations[layer_id].retain_grad()

        # Process SAE Features
        features = _unpack_proxy(sae_features_proxy[layer_id])

        # NNSight proxy ops (e.g. squeeze) may not fire identically in the
        # trace graph, leaving an extra batch dim.  Normalise to [seq_len, n_features].
        if features.dim() == 3 and features.shape[0] == 1:
            features = features.squeeze(0)  # [1, seq_len, n_features] -> [seq_len, n_features]
        elif features.dim() == 1:
            raise RuntimeError(
                f"SAE features for layer {layer_id} are unexpectedly 1D "
                f"(shape={tuple(features.shape)}). This is a proxy extraction bug."
            )

        # Convert to sparse if highly sparse (>80% zeros) for memory efficiency
        sparsity = 1.0 - (features.count_nonzero().item() / features.numel())
        if sparsity > 0.8:
            features = features.to_sparse()

        features.retain_grad()
        sae_features[layer_id] = features

    # Get pre-logit hidden state and retain grad
    val_pre_logit = _unpack_proxy(pre_logit_proxy)
    # NNSight proxy already has batch dim; only unsqueeze if 2D
    pre_logit = val_pre_logit.unsqueeze(0) if val_pre_logit.dim() == 2 else val_pre_logit
    pre_logit.retain_grad()

    embed_act = _unpack_proxy(embed_proxy)
    embed_act.retain_grad()

    lin_hooks.remove()
    model.config._attn_implementation = original_attn_impl

    return {
        "input_ids": input_ids[0],
        "tokens": tokens,
        "logits": logits,
        "sae_features": sae_features,
        "mlp_activations": mlp_activations,
        "pre_logit_hidden": pre_logit,
        "embedding_activations": embed_act,
    }
