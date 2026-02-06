"""Gradient-based attribution from SAE features to output logits.

Implements the Attribution Graphs paper methodology:
- Linearized gradient flow (detached attention, frozen RMSNorm scale)
- Cumulative-probability logit selection with demeaned unembedding vectors
- Vectorized attribution computation
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .attribution_graph import AttributionGraph, Edge, Node
from .forward_with_sae import forward_linearized_with_sae_features
from .load_transcoder import load_transcoders_for_layers
from .salient_logits import compute_salient_logits


def compute_attribution_graph(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    layers_to_analyze: list[int],
    transcoder_repo: str = "mwhanna/qwen3-4b-transcoders",
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    feature_threshold: float = 0.01,
    min_attribution: float = 1e-3,
) -> AttributionGraph:
    """Compute attribution graph from input tokens through SAE features to output logits.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        prompt: Input prompt text
        layers_to_analyze: List of layer IDs to extract features from
        transcoder_repo: HuggingFace repo containing transcoders
        max_n_logits: Maximum number of top logits to consider
        desired_logit_prob: Cumulative probability threshold for logit selection
        feature_threshold: Minimum feature activation to include
        min_attribution: Minimum |attribution score| to include an edge

    Returns:
        Attribution graph with nodes and edges
    """
    # Phase 1: Setup
    print(f"Loading transcoders for layers: {layers_to_analyze}")
    transcoders = load_transcoders_for_layers(
        layer_ids=layers_to_analyze,
        transcoder_repo=transcoder_repo,
        device=str(model.device),
    )

    print("Running linearized forward pass with SAE features...")
    forward_result = forward_linearized_with_sae_features(
        model=model,
        tokenizer=tokenizer,
        transcoders=transcoders,
        prompt=prompt,
        layers_to_analyze=layers_to_analyze,
    )

    tokens = forward_result["tokens"]
    logits = forward_result["logits"]  # [seq_len, vocab_size]
    sae_features = forward_result["sae_features"]  # {layer_id: [seq_len, n_features]}
    mlp_activations = forward_result["mlp_activations"]
    pre_logit_hidden = forward_result["pre_logit_hidden"]  # [batch, seq_len, d_model]

    last_pos = len(tokens) - 1

    # Phase 2: Salient logit selection
    unembed_weight = model.lm_head.weight.detach()
    logit_indices, logit_probs, demeaned_vecs = compute_salient_logits(
        logits[last_pos].detach(),
        unembed_weight,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )
    n_logits = len(logit_indices)

    print(f"\nSalient logits ({n_logits} tokens, cumprob={logit_probs.sum():.3f}):")
    for i in range(n_logits):
        tok_str = tokenizer.decode([logit_indices[i].item()])
        print(f"  {i + 1}. '{tok_str}' (prob={logit_probs[i].item():.4f})")

    # Phase 3: Demeaned gradient computation
    mlp_tensors = [mlp_activations[layer_id] for layer_id in layers_to_analyze]

    # h is the pre-logit hidden state at the last position (with live grad)
    h = pre_logit_hidden[0, last_pos]  # [d_model]

    # Cast demeaned_vecs to same dtype as h for the element-wise multiply
    demeaned_vecs = demeaned_vecs.to(dtype=h.dtype, device=h.device)

    # Compute gradients for each salient logit
    grads_per_logit = []
    for j in range(n_logits):
        # Element-wise multiply + sum avoids matmul backward codepath
        # (matmul backward can call .H on 1-D tensors in some PyTorch versions)
        target_j = (h * demeaned_vecs[j]).sum()

        model.zero_grad(set_to_none=True)
        grads_j = torch.autograd.grad(
            target_j,
            mlp_tensors,
            retain_graph=(j < n_logits - 1),
            allow_unused=True,
        )
        # Detach grads immediately to free graph memory
        grads_per_logit.append(tuple(g.detach() if g is not None else None for g in grads_j))

    # Free the computation graph — no longer needed
    del pre_logit_hidden, h, mlp_activations, mlp_tensors, logits
    del forward_result

    # Phase 4: Build graph
    graph = AttributionGraph()

    # Add input token nodes
    for pos, token_str in enumerate(tokens):
        graph.add_node(
            Node(
                node_id=f"token_{pos}",
                node_type="token",
                token_pos=pos,
                token_str=token_str,
            )
        )

    # Add logit nodes
    logit_nodes = []
    for j in range(n_logits):
        tok_id = logit_indices[j].item()
        tok_str = tokenizer.decode([tok_id])
        logit_node = Node(
            node_id=f"logit_{tok_id}",
            node_type="logit",
            logit_token_id=tok_id,
            logit_token_str=tok_str,
            activation=logit_probs[j].item(),
        )
        graph.add_node(logit_node)
        logit_nodes.append(logit_node)

    # Phase 5: Vectorized attribution via matmul (no large gather)
    # Key identity: attribution[pos,feat] = features[pos,feat] * (W_dec[feat] @ grad[pos])
    #             = features * (grad @ W_dec.T)     — [seq, n_feat], zeros stay zero
    print("Computing attributions...")
    edge_count = 0
    feature_count = 0

    for layer_idx, layer_id in enumerate(layers_to_analyze):
        features = sae_features[layer_id].detach()  # [seq_len, n_features]
        transcoder = transcoders[layer_id]
        W_dec = transcoder.W_dec.detach().float()  # [n_features, d_model]
        features_f = features.float()

        # Track which (pos, feat_id) have a node already
        created_nodes: dict[tuple[int, int], str] = {}  # (pos, feat_id) -> node_id

        with torch.no_grad():
            # Diagnostics: compute attribution distribution for first logit
            diag_grad = grads_per_logit[0][layer_idx]
            if diag_grad is not None:
                if diag_grad.dim() == 3:
                    diag_grad = diag_grad[0]
                diag_attr = features_f * (diag_grad.float() @ W_dec.t())
                active = diag_attr[features.abs() > feature_threshold].abs()
                if active.numel() > 0:
                    pcts = torch.quantile(
                        active.float(),
                        torch.tensor([0.5, 0.9, 0.95, 0.99, 1.0], device=active.device),
                    )
                    print(
                        f"  Layer {layer_id} attribution |a| distribution "
                        f"(active features, logit 0):"
                    )
                    print(
                        f"    p50={pcts[0]:.4f}  p90={pcts[1]:.4f}  "
                        f"p95={pcts[2]:.4f}  p99={pcts[3]:.4f}  max={pcts[4]:.4f}"
                    )
                    for thresh in [0.001, 0.01, 0.1, 0.5]:
                        n = (active > thresh).sum().item()
                        print(f"    |a|>{thresh}: {n} edges")
                del diag_grad, diag_attr, active

            for j in range(n_logits):
                dlogit_dmlp = grads_per_logit[j][layer_idx]
                if dlogit_dmlp is None:
                    continue
                if dlogit_dmlp.dim() == 3:
                    dlogit_dmlp = dlogit_dmlp[0]  # [seq_len, d_model]

                # Single matmul: [seq, d_model] @ [d_model, n_feat] = [seq, n_feat]
                dec_dot_grad = dlogit_dmlp.float() @ W_dec.t()

                # Element-wise: zero features produce zero attribution automatically
                attributions = features_f * dec_dot_grad  # [seq, n_feat]

                # Joint filter: active feature AND significant attribution
                mask = (features.abs() > feature_threshold) & (attributions.abs() > min_attribution)
                edge_pos, edge_feat = mask.nonzero(as_tuple=True)

                if len(edge_pos) == 0:
                    continue

                # Batch-extract values to Python (one GPU→CPU transfer)
                attr_vals = attributions[edge_pos, edge_feat].tolist()
                pos_list = edge_pos.tolist()
                feat_list = edge_feat.tolist()

                for idx in range(len(pos_list)):
                    pos = pos_list[idx]
                    feat_id = feat_list[idx]
                    key = (pos, feat_id)

                    if key not in created_nodes:
                        node_id = f"feature_L{layer_id}_P{pos}_F{feat_id}"
                        graph.add_node(
                            Node(
                                node_id=node_id,
                                node_type="feature",
                                layer=layer_id,
                                feature_id=feat_id,
                                token_pos=pos,
                                activation=features[pos, feat_id].item(),
                            )
                        )
                        created_nodes[key] = node_id
                        feature_count += 1

                    graph.add_edge(
                        Edge(
                            source=graph.get_node(created_nodes[key]),
                            target=logit_nodes[j],
                            attribution_score=attr_vals[idx],
                        )
                    )
                    edge_count += 1

        print(f"  Layer {layer_id}: {len(created_nodes)} feature nodes, {edge_count} edges so far")

    print(f"Added {feature_count} SAE feature nodes (threshold={feature_threshold})")
    print(f"Added {edge_count} attribution edges (threshold={min_attribution})")
    print(f"\nGraph construction complete: {graph}")

    return graph
