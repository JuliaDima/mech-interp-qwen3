import time
from typing import Literal

import torch
from tqdm import tqdm

from .attribution_model import AttributionModel
from .graph import Graph, compute_partial_influences
from .utils.model_utils import compute_salient_logits, offload_modules
from .utils.token_utils import tokenize_qwen_input


def attribute(
    prompt: str | torch.Tensor | list[int],
    model: AttributionModel,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
) -> Graph:
    """Compute an attribution graph for *prompt*.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``AttributionModel``
        max_n_logits: Max number of logit nodes.
        desired_logit_prob: Keep logits until cumulative prob >= this value.
        batch_size: How many source nodes to process per backward pass.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """
    offload_handles = []
    start_time = time.time()

    try:
        # --- step 1: tokenise and precompute transcoder activations ---
        if verbose:
            print("Step 1/5: tokenising and precomputing transcoder activations")
        phase_start = time.time()

        input_ids = tokenize_qwen_input(prompt, model.tokenizer, model.cfg.device)
        ctx = model.setup_attribution(input_ids)
        activation_matrix = ctx.activation_matrix

        if verbose:
            print(
                f"  done in {time.time() - phase_start:.2f}s  ({activation_matrix._nnz()} active features)"
            )

        if offload:
            offload_handles += offload_modules(model.transcoders, offload)

        # --- step 2: batched forward pass to populate residual cache ---
        if verbose:
            print("Step 2/5: running batched forward pass")
        phase_start = time.time()

        with ctx.install_hooks(model):
            residual = model.forward(
                input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers
            )
            ctx._resid_activations[-1] = model.ln_final(residual)

        if verbose:
            print(f"  done in {time.time() - phase_start:.2f}s")

        if offload:
            offload_handles += offload_modules([block.mlp for block in model.blocks], offload)

        # --- step 3: select salient logits and allocate the edge matrix ---
        if verbose:
            print("Step 3/5: selecting salient logits and allocating edge matrix")
        phase_start = time.time()

        feat_layers, feat_pos, _ = activation_matrix.indices()
        n_layers, n_pos, _ = activation_matrix.shape
        total_active_feats = activation_matrix._nnz()

        logit_idx, logit_p, logit_vecs = compute_salient_logits(
            ctx.logits[0, -1],
            model.unembed.W_U,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
        )
        n_logits = len(logit_idx)

        if verbose:
            print(f"  {n_logits} logits selected (cumulative p = {logit_p.sum().item():.4f})")

        if offload:
            offload_handles += offload_modules([model.unembed, model.embed], offload)

        # logit_offset: first column index belonging to logit nodes
        logit_offset = total_active_feats + (n_layers + 1) * n_pos
        total_nodes = logit_offset + n_logits
        max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)

        if verbose:
            print(f"  using {max_feature_nodes} of {total_active_feats} feature nodes")

        n_rows = max_feature_nodes + n_logits
        edge_matrix = torch.zeros(n_rows, total_nodes)
        row_to_node_index = torch.zeros(n_rows, dtype=torch.int32)

        if verbose:
            print(f"  done in {time.time() - phase_start:.2f}s")

        # --- step 4: attribute logit nodes ---
        if verbose:
            print("Step 4/5: attributing logit nodes")
        phase_start = time.time()

        for i in range(0, n_logits, batch_size):
            batch = logit_vecs[i : i + batch_size]
            b = batch.shape[0]
            rows = ctx.compute_batch(
                layers=torch.full((b,), n_layers),
                positions=torch.full((b,), n_pos - 1),
                inject_values=batch,
            )
            edge_matrix[i : i + b, :logit_offset] = rows.cpu()
            row_to_node_index[i : i + b] = torch.arange(i, i + b) + logit_offset

        if verbose:
            print(f"  done in {time.time() - phase_start:.2f}s")

        # --- step 5: attribute feature nodes ---
        if verbose:
            print("Step 5/5: attributing feature nodes")
        phase_start = time.time()

        write_head = n_logits
        visited = torch.zeros(total_active_feats, dtype=torch.bool)
        n_visited = 0

        pbar = tqdm(total=max_feature_nodes, desc="feature attribution", disable=not verbose)

        while n_visited < max_feature_nodes:
            if max_feature_nodes == total_active_feats:
                pending = torch.arange(total_active_feats)
            else:
                influences = compute_partial_influences(
                    edge_matrix[:write_head], logit_p, row_to_node_index[:write_head]
                )
                feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
                queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
                pending = feature_rank[~visited[feature_rank]][:queue_size]

            for batch_start in range(0, len(pending), batch_size):
                idx_batch = pending[batch_start : batch_start + batch_size]
                n_visited += len(idx_batch)

                rows = ctx.compute_batch(
                    layers=feat_layers[idx_batch],
                    positions=feat_pos[idx_batch],
                    inject_values=ctx.encoder_vecs[idx_batch],
                    retain_graph=n_visited < max_feature_nodes,
                )

                end = write_head + rows.shape[0]
                edge_matrix[write_head:end, :logit_offset] = rows.cpu()
                row_to_node_index[write_head:end] = idx_batch
                visited[idx_batch] = True
                write_head = end
                pbar.update(len(idx_batch))

        pbar.close()
        if verbose:
            print(f"  done in {time.time() - phase_start:.2f}s")

        # --- package the graph ---
        selected_features = torch.where(visited)[0]
        if max_feature_nodes < total_active_feats:
            col_read = torch.cat([selected_features, torch.arange(total_active_feats, total_nodes)])
            edge_matrix = edge_matrix[:, col_read]

        edge_matrix = edge_matrix[row_to_node_index.argsort()]
        full_edge_matrix = torch.zeros(edge_matrix.shape[1], edge_matrix.shape[1])
        full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
        full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]

        graph = Graph(
            input_string=model.tokenizer.decode(input_ids),
            input_tokens=input_ids,
            logit_tokens=logit_idx,
            logit_probabilities=logit_p,
            active_features=activation_matrix.indices().T,
            activation_values=activation_matrix.values(),
            selected_features=selected_features,
            adjacency_matrix=full_edge_matrix,
            cfg=model.cfg,
            scan=model.scan,
        )

        if verbose:
            print(f"Attribution finished in {time.time() - start_time:.2f}s total")

        return graph

    finally:
        for reload_handle in offload_handles:
            reload_handle()
