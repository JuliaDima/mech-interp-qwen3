"""Gradient-based attribution from SAE features to output logits.

Computes how much each SAE feature contributes to the output logits using gradients.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .attribution_graph import AttributionGraph, Edge, Node
from .forward_with_sae import forward_with_sae_features_grad
from .load_transcoder import load_transcoders_for_layers


def compute_attribution_graph(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    layers_to_analyze: list[int],
    transcoder_repo: str = "mwhanna/qwen3-4b-transcoders",
    max_n_logits: int = 10,
    feature_threshold: float = 0.01,
) -> AttributionGraph:
    """
    Compute attribution graph from input tokens through SAE features to output logits.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        prompt: Input prompt text
        layers_to_analyze: List of layer IDs to extract features from
        transcoder_repo: HuggingFace repo containing transcoders
        max_n_logits: Maximum number of top logits to attribute from
        feature_threshold: Minimum feature activation to include

    Returns:
        Attribution graph with nodes and edges
    """
    print(f"Loading transcoders for layers: {layers_to_analyze}")
    transcoders = load_transcoders_for_layers(
        layer_ids=layers_to_analyze,
        transcoder_repo=transcoder_repo,
        device=str(model.device),
    )

    print("Running forward pass with SAE features...")
    forward_result = forward_with_sae_features_grad(
        model=model,
        tokenizer=tokenizer,
        transcoders=transcoders,
        prompt=prompt,
        layers_to_analyze=layers_to_analyze,
    )

    # input_ids = forward_result["input_ids"]
    tokens = forward_result["tokens"]
    logits = forward_result["logits"]  # [seq_len, vocab_size]
    sae_features = forward_result["sae_features"]  # {layer_id: [seq_len, n_features]}

    # sanity check
    for l_id in layers_to_analyze:
        x = sae_features[l_id]  # [seq, n_feat]
        x = x.float()
        print(l_id, x.shape, x.dtype)
        print("abs mean", x.abs().mean().item())
        print("abs p50", x.abs().median().item())
        print("abs p90", x.abs().quantile(0.9).item())
        print("abs p99", x.abs().quantile(0.99).item())
        print("frac > 0.01", (x.abs() > 0.01).float().mean().item())
    exit()
    # Focus on the last token position (where the answer is generated)
    last_pos = len(tokens) - 1

    # Get top-k logits at the last position
    last_logits = logits[last_pos]  # [vocab_size]
    top_logit_values, top_logit_indices = torch.topk(last_logits, k=max_n_logits)

    print(f"\nTop {max_n_logits} predicted tokens:")
    for i, (logit_idx, logit_val) in enumerate(
        zip(top_logit_indices, top_logit_values, strict=False)
    ):
        token_str = tokenizer.decode([logit_idx.item()])
        print(f"  {i + 1}. '{token_str}' (logit={logit_val.item():.2f})")

    # Initialize attribution graph
    graph = AttributionGraph()

    # Add input token nodes
    print(f"\nAdding {len(tokens)} input token nodes...")
    for pos, token_str in enumerate(tokens):
        node = Node(
            node_id=f"token_{pos}",
            node_type="token",
            token_pos=pos,
            token_str=token_str,
        )
        graph.add_node(node)

    # Add SAE feature nodes (only for features with non-zero activation)
    print("Adding SAE feature nodes...")
    feature_count = 0
    for layer_id in layers_to_analyze:
        features = sae_features[layer_id]  # [seq_len, n_features]

        for pos in range(features.shape[0]):
            for feat_id in range(features.shape[1]):
                activation = features[pos, feat_id].item()

                # Only include features above threshold
                if abs(activation) > feature_threshold:
                    node = Node(
                        node_id=f"feature_L{layer_id}_P{pos}_F{feat_id}",
                        node_type="feature",
                        layer=layer_id,
                        feature_id=feat_id,
                        token_pos=pos,
                        activation=activation,
                    )
                    graph.add_node(node)
                    feature_count += 1

    print(f"Added {feature_count} SAE feature nodes (threshold={feature_threshold})")

    # Add logit nodes
    print(f"Adding {max_n_logits} logit nodes...")
    edge_count = 0
    for logit_idx in top_logit_indices:
        logit_value = last_logits[logit_idx]
        token_str = tokenizer.decode([logit_idx.item()])
        logit_node = Node(
            node_id=f"logit_{logit_idx.item()}",
            node_type="logit",
            logit_token_id=logit_idx.item(),
            logit_token_str=token_str,
            activation=logit_value.item(),
        )
        graph.add_node(logit_node)
        mlp_activations = forward_result["mlp_activations"]
        mlp_tensors = [mlp_activations[layer_id] for layer_id in layers_to_analyze]

        model.zero_grad(set_to_none=True)
        mlp_grads = torch.autograd.grad(
            logit_value,
            mlp_tensors,
            retain_graph=True,
            allow_unused=True,
        )

        for layer_id, mlp_acts, dlogit_dmlp in zip(
            layers_to_analyze, mlp_tensors, mlp_grads, strict=False
        ):
            features = sae_features[layer_id]  # [seq_len, n_features]
            transcoder = transcoders[layer_id]

            if mlp_acts.dim() == 3:
                mlp_acts = mlp_acts[0]
            if dlogit_dmlp.dim() == 3:
                dlogit_dmlp = dlogit_dmlp[0]

            W_dec = transcoder.W_dec  # [d_transcoder, d_model]
            n_features = features.shape[1]

            active = (features.abs() > feature_threshold).nonzero(as_tuple=False)
            for pos, feat_id in active.tolist():
                a = features[pos, feat_id].item()
                d_vec = W_dec[feat_id] if W_dec.shape[0] == n_features else W_dec[:, feat_id]
                attribution = (dlogit_dmlp[pos] * (a * d_vec)).sum().item()

                if abs(attribution) > 1e-6:
                    feature_node = graph.get_node(f"feature_L{layer_id}_P{pos}_F{feat_id}")
                    if feature_node and logit_node:
                        graph.add_edge(
                            Edge(
                                source=feature_node,
                                target=logit_node,
                                attribution_score=attribution,
                            )
                        )
                        edge_count += 1

    print(f"Added {edge_count} attribution edges")
    print(f"\nGraph construction complete: {graph}")

    return graph
