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
    top_k_features: int | None = None,
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
        top_k_features: If set, keep only top K features per token (sparse storage)

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
        top_k_features=top_k_features,
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

    # Save detached MLP activations for error computation
    # (We need the values, but not the graph, to compute residuals)
    mlp_acts_detached = {lid: act.detach() for lid, act in mlp_activations.items()}

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

    # Start attribution from position 1 to ignore BOS/system tokens which often cause artifacts
    start_pos = 1

    for layer_idx, layer_id in enumerate(layers_to_analyze):
        features = sae_features[layer_id]  # [seq_len, n_features] (maybe sparse)
        transcoder = transcoders[layer_id]
        W_dec = transcoder.W_dec.detach().float()  # [n_features, d_model]

        # Get MLP activations for error computation
        mlp_act = mlp_acts_detached[layer_id].float()
        if mlp_act.dim() == 3:
            mlp_act = mlp_act[0]

        # Compute reconstruction and error
        is_sparse = features.is_sparse
        with torch.no_grad():
            if is_sparse:
                # decode_sparse returns (reconstruction, scaled_decoders)
                reconstruction, _ = transcoder.decode_sparse(features, mlp_act)
            else:
                reconstruction = transcoder.decode(features, mlp_act)

            # Compute reconstruction error (MLP_out - Reconstruction)
            # This represents what the SAE failed to explain
            error = mlp_act - reconstruction.detach()

        features_f = features.float() if not is_sparse else features

        # Track which (pos, feat_id) have a node already
        created_nodes: dict[tuple[int, int], str] = {}  # (pos, feat_id) -> node_id

        # ---------------------------------------------------------------------
        # Part A: Diagnostics (Logit 0 only, optional)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            if not is_sparse:
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

        # ---------------------------------------------------------------------
        # Part B: Attribution Computation (Features & Error)
        # ---------------------------------------------------------------------
        for j in range(n_logits):
            dlogit_dmlp = grads_per_logit[j][layer_idx]
            if dlogit_dmlp is None:
                continue
            if dlogit_dmlp.dim() == 3:
                dlogit_dmlp = dlogit_dmlp[0]  # [seq_len, d_model]

            # --- 1. Feature Attribution ---
            dec_dot_grad = dlogit_dmlp.float() @ W_dec.t()
            attributions = features_f * dec_dot_grad

            if is_sparse:
                attributions = attributions.coalesce()
                attr_vals = attributions.values()
                attr_indices = attributions.indices()
                feat_vals = features_f.coalesce().values()

                mask = (feat_vals.abs() > feature_threshold) & (attr_vals.abs() > min_attribution)
                valid_indices = torch.nonzero(mask).squeeze(1)

                if valid_indices.numel() > 0:
                    final_attr_vals = attr_vals[valid_indices].tolist()
                    final_pos_list = attr_indices[0, valid_indices].tolist()
                    final_feat_list = attr_indices[1, valid_indices].tolist()
                    final_feat_vals = feat_vals[valid_indices].tolist()

                    loop_zip = zip(
                        final_pos_list,
                        final_feat_list,
                        final_attr_vals,
                        final_feat_vals,
                        strict=True,
                    )
                else:
                    loop_zip = []
            else:
                mask = (features.abs() > feature_threshold) & (attributions.abs() > min_attribution)
                edge_pos, edge_feat = mask.nonzero(as_tuple=True)
                if len(edge_pos) > 0:
                    attr_vals_list = attributions[edge_pos, edge_feat].tolist()
                    pos_list = edge_pos.tolist()
                    feat_list = edge_feat.tolist()
                    feat_vals_list = features[edge_pos, edge_feat].tolist()
                    loop_zip = zip(pos_list, feat_list, attr_vals_list, feat_vals_list, strict=True)
                else:
                    loop_zip = []

            for pos, feat_id, attr_val, feat_val in loop_zip:
                # Filter out positions before start_pos
                if pos < start_pos:
                    continue

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
                            activation=feat_val,
                        )
                    )
                    created_nodes[key] = node_id
                    feature_count += 1

                graph.add_edge(
                    Edge(
                        source=graph.get_node(created_nodes[key]),
                        target=logit_nodes[j],
                        attribution_score=attr_val,
                    )
                )
                edge_count += 1

            # --- 2b. Error Term Attribution ---
            # Attribution = Error · Grad
            # Error is [seq, d_model], Grad is [seq, d_model]
            error_attr = (error * dlogit_dmlp.float()).sum(dim=-1)  # [seq]

            for pos, attr_val in enumerate(error_attr.tolist()):
                if pos < start_pos:
                    continue
                if abs(attr_val) > min_attribution:
                    # Check/Create Error Node
                    # One error node per (layer, position)
                    error_node_id = f"error_L{layer_id}_P{pos}"
                    if graph.get_node(error_node_id) is None:
                        graph.add_node(
                            Node(
                                node_id=error_node_id,
                                node_type="error",
                                layer=layer_id,
                                token_pos=pos,
                                activation=error[pos].norm().item(),
                                feature_id=-1,  # Marker for error
                            )
                        )

                    # Add Edge
                    graph.add_edge(
                        Edge(
                            source=graph.get_node(error_node_id),
                            target=logit_nodes[j],
                            attribution_score=attr_val,
                        )
                    )
                    edge_count += 1

            # --- 2c. Bias Term Attribution ---
            # The decoder bias b_dec is a constant vector added to the reconstruction
            # Attribution = b_dec · Grad
            if hasattr(transcoder, "b_dec"):
                b_dec = transcoder.b_dec.detach().float()  # [d_model]
                bias_attr = (b_dec * dlogit_dmlp.float()).sum(
                    dim=-1
                )  # scalar (since batch=1 usually)

                # If dlogit_dmlp has sequence dim, bias_attr might be [seq]
                # But b_dec is global (not seq-dependent), so this is technically
                # adding the *same* bias vector at every position.
                # The gradient dlogit_dmlp is [seq, d_model].
                # So bias_attr is [seq].

                for pos, attr_val in enumerate(bias_attr.view(-1).tolist()):
                    if pos < start_pos:
                        continue
                    if abs(attr_val) > min_attribution:
                        # Create Bias Node (one per layer, shared across positions?
                        # Actually, bias contribution is position-dependent because gradient is pos-dependent)
                        # Let's create one bias node per layer/position to be consistent with error nodes

                        bias_node_id = f"bias_L{layer_id}_P{pos}"
                        if graph.get_node(bias_node_id) is None:
                            graph.add_node(
                                Node(
                                    node_id=bias_node_id,
                                    node_type="bias",
                                    layer=layer_id,
                                    token_pos=pos,
                                    activation=1.0,  # Bias is constant, effectively activity 1.0 * b_dec
                                    feature_id=-2,  # Marker for bias
                                )
                            )

                        graph.add_edge(
                            Edge(
                                source=graph.get_node(bias_node_id),
                                target=logit_nodes[j],
                                attribution_score=attr_val,
                            )
                        )
                        edge_count += 1
            else:
                bias_attr = torch.zeros_like(error_attr)

            # --- 3. Verification: Check Sum ---
            # Direct Attribution: mlp_act * grad
            # Component Attribution: Sum(Features * grad @ W_dec.T) + (Error * grad) + (Bias * grad)
            # which equals (Rec + Error) * grad = mlp_act * grad

            with torch.no_grad():
                # Direct attribution of MLP output to logit (exclude start_pos)
                if mlp_act.shape[0] > start_pos and dlogit_dmlp.shape[0] > start_pos:
                    direct_attr = (mlp_act[start_pos:] * dlogit_dmlp[start_pos:].float()).sum()
                else:
                    direct_attr = torch.tensor(0.0, device=mlp_act.device)

                # Sum of all feature attributions (sparse or dense)
                # We need to filter feature attributions for pos >= start_pos
                if is_sparse:
                    # Filter values where stored index (pos) >= start_pos
                    # attributions is sparse [seq, n_feat]
                    # indices[0] is pos
                    mask_pos = attributions.indices()[0] >= start_pos
                    feat_attr_sum = attributions.values()[mask_pos].sum()
                else:
                    feat_attr_sum = attributions[start_pos:].sum()

                # Sum of error attribution
                err_attr_sum = error_attr[start_pos:].sum()

                # Sum of bias attribution
                bias_attr_sum = bias_attr[start_pos:].sum()

                total_component_attr = feat_attr_sum + err_attr_sum + bias_attr_sum
                diff = abs(direct_attr - total_component_attr).item()
                rel_diff = diff / (abs(direct_attr).item() + 1e-8)

                print(f"  Layer {layer_id} Verification (Logit {j}):")
                print(f"    Direct MLP Attr:   {direct_attr.item():.4f}")
                print(
                    f"    Component Sum:     {total_component_attr.item():.4f} "
                    f"(Feat: {feat_attr_sum.item():.4f} + Err: {err_attr_sum.item():.4f} + Bias: {bias_attr_sum.item():.4f})"
                )
                print(f"    Difference:        {diff:.6f} (Rel: {rel_diff:.2%})")

                if rel_diff > 0.01:  # >1% error warning
                    print(f"    [WARNING] Attribution mismatch in Layer {layer_id}!")

        print(f"  Layer {layer_id}: {len(created_nodes)} feature nodes, {edge_count} edges so far")

    print(f"Added {feature_count} SAE feature nodes (threshold={feature_threshold})")
    print(f"Added {edge_count} attribution edges (threshold={min_attribution})")
    print(f"\nGraph construction complete: {graph}")

    return graph
