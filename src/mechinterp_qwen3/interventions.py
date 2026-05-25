"""Feature inhibition and constrained patching for circuit validation.

Two intervention strategies for testing causal hypotheses in transformer circuits:

1. **Direct inhibition** — Scale target features to zero (or by alpha) in a
   standard forward pass. Fast but potentially confounded since upstream activations
   may already differ between clean and perturbed inputs.

2. **Constrained patching** — Cache MLP inputs from a perturbed forward pass,
   then re-run the clean prompt with:
   - Layers < intervention_layer: MLP inputs replaced by the perturbed cache
   - intervention_layer: feature scaling applied
   - Layers above: run normally from the modified residual stream

   Clamping earlier layers holds upstream state fixed, isolating the effect
   of the intervention at the chosen layer.
"""

from __future__ import annotations

import json as _json
import logging as _logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

_log = _logging.getLogger(__name__)

if TYPE_CHECKING:
    from mechinterp_qwen3.attribution_model import AttributionModel
    from mechinterp_qwen3.graph import Graph


# ---------------------------------------------------------------------------
# activation caching utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_mlp_inputs(
    model: AttributionModel,
    tokens: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Run a forward pass and return the pre-transcoder hidden state for each layer.

    Args:
        model: AttributionModel instance
        tokens: Token IDs, shape (1, n_pos) or (n_pos,)

    Returns:
        Mapping from layer index to MLP input tensor (1, n_pos, d_model), stored on CPU
    """
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)

    cache: dict[int, torch.Tensor] = {}
    hooks: list[tuple[str, Callable]] = []

    for layer in range(model.cfg.n_layers):

        def _hook(acts: torch.Tensor, hook, *, _layer: int = layer) -> torch.Tensor:
            cache[_layer] = acts.detach().cpu()
            return acts

        hook_name = f"blocks.{layer}.{model.feature_input_hook}"
        hooks.append((hook_name, _hook))

    model.run_with_hooks(tokens, fwd_hooks=hooks)
    return cache


# ---------------------------------------------------------------------------
# inhibition hook construction
# ---------------------------------------------------------------------------


def make_inhibit_hook(
    model: AttributionModel,
    layer: int,
    feature_ids: list[int],
    alpha: float = 0.0,
) -> tuple[str, Callable]:
    """Build a hook that scales selected feature activations by alpha at the given layer.

    Re-encodes the MLP input, scales the targeted features, then decodes back to residual space.

    Args:
        model: AttributionModel instance
        layer: Layer at which to apply the scaling
        feature_ids: Feature indices to scale
        alpha: Multiplicative scale (0.0 = full suppression, 1.0 = no change)

    Returns:
        (hook_name, hook_fn) ready to pass to run_with_hooks
    """
    transcoder = model.transcoders[layer]  # type: ignore[index]

    def _hook(mlp_out: torch.Tensor, hook) -> torch.Tensor:
        h_in = _hook._last_mlp_in  # type: ignore[attr-defined]
        if h_in is None:
            return mlp_out

        with torch.no_grad():
            W_enc = transcoder.W_enc  # (d_tc, d_model)
            b_enc = transcoder.b_enc
            W_dec = transcoder.W_dec  # (d_tc, d_model)
            b_dec = transcoder.b_dec

            # Encode
            pre = F.linear(h_in.to(W_enc.dtype), W_enc, b_enc)
            acts = transcoder.activation_function(pre)  # (1, n_pos, d_tc)

            # Scale inhibited features
            for fid in feature_ids:
                acts[..., fid] = acts[..., fid] * alpha

            # Decode
            reconstructed = acts @ W_dec + b_dec

            if transcoder.W_skip is not None:
                reconstructed = reconstructed + h_in.to(reconstructed.dtype) @ transcoder.W_skip.T

        return reconstructed.to(mlp_out.dtype)

    _hook._last_mlp_in = None  # type: ignore[attr-defined]
    hook_name = f"blocks.{layer}.{model.original_feature_output_hook}"
    return hook_name, _hook


def make_capture_hook(
    model: AttributionModel,
    layer: int,
    inhibit_hook_fn: Callable,
) -> tuple[str, Callable]:
    """Build a hook that saves the MLP input so the inhibit hook can use it.

    Must be registered before the corresponding inhibit hook.

    Args:
        model: AttributionModel instance
        layer: Layer index
        inhibit_hook_fn: The inhibit hook whose ``_last_mlp_in`` attribute will be populated

    Returns:
        (hook_name, hook_fn) ready to pass to run_with_hooks
    """

    def _capture(acts: torch.Tensor, hook) -> torch.Tensor:
        inhibit_hook_fn._last_mlp_in = acts.detach()  # type: ignore[attr-defined]
        return acts

    return (f"blocks.{layer}.{model.feature_input_hook}", _capture)


# ---------------------------------------------------------------------------
# intervention implementations
# ---------------------------------------------------------------------------


@torch.no_grad()
def inhibit_features(
    model: AttributionModel,
    tokens: torch.Tensor,
    feature_ids_by_layer: dict[int, list[int]],
    *,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Direct (unconstrained) feature inhibition: scale features and run a normal forward pass.

    Args:
        model: AttributionModel instance
        tokens: Token IDs, shape (1, n_pos) or (n_pos,)
        feature_ids_by_layer: Mapping {layer_idx: [feature_indices]} to scale
        alpha: Scale factor applied to each listed feature (0.0 = full suppression)

    Returns:
        Output logits, shape (1, n_pos, d_vocab)
    """
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)

    hooks: list[tuple[str, Callable]] = []
    for layer, feat_ids in feature_ids_by_layer.items():
        inhibit_name, inhibit_fn = make_inhibit_hook(model, layer, feat_ids, alpha)
        capture_name, capture_fn = make_capture_hook(model, layer, inhibit_fn)
        hooks.append((capture_name, capture_fn))
        hooks.append((inhibit_name, inhibit_fn))

    return model.run_with_hooks(tokens, fwd_hooks=hooks)


@torch.no_grad()
def constrained_patch(
    model: AttributionModel,
    tokens_clean: torch.Tensor,
    tokens_perturbed: torch.Tensor,
    intervention_layer: int,
    feature_ids_by_layer: dict[int, list[int]],
    *,
    alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run constrained patching with upstream activation clamping.

    Steps:
      1. Forward pass on the perturbed prompt to cache MLP inputs per layer
      2. Re-run the clean prompt with:
         - Layers < intervention_layer: MLP inputs replaced by the perturbed cache
         - intervention_layer: feature scaling applied
         - Layers above: run normally from the modified residual
      3. Also run unconstrained inhibition (no clamping) for comparison

    Args:
        model: AttributionModel instance
        tokens_clean: Clean prompt token ids, shape (1, n_pos)
        tokens_perturbed: Perturbed prompt token ids, shape (1, n_pos)
        intervention_layer: Layer at which feature scaling is applied
        feature_ids_by_layer: Mapping {layer_idx: [feature_indices]}
        alpha: Feature scale factor (0.0 = full suppression)

    Returns:
        (constrained_logits, unconstrained_logits), each shape (1, n_pos, d_vocab)
    """
    if tokens_clean.ndim == 1:
        tokens_clean = tokens_clean.unsqueeze(0)
    if tokens_perturbed.ndim == 1:
        tokens_perturbed = tokens_perturbed.unsqueeze(0)

    # Cache MLP inputs from the perturbed run
    perturbed_acts = collect_mlp_inputs(model, tokens_perturbed)

    # Build feature-inhibition hooks
    inhibit_hooks: list[tuple[str, Callable]] = []
    for layer, feat_ids in feature_ids_by_layer.items():
        inhibit_name, inhibit_fn = make_inhibit_hook(model, layer, feat_ids, alpha)
        capture_name, capture_fn = make_capture_hook(model, layer, inhibit_fn)
        inhibit_hooks.append((capture_name, capture_fn))
        inhibit_hooks.append((inhibit_name, inhibit_fn))

    # Build clamping hooks for all layers before the intervention point
    clamp_hooks: list[tuple[str, Callable]] = []
    for layer in range(intervention_layer):
        if layer not in perturbed_acts:
            continue
        clamp_val = perturbed_acts[layer].to(model.cfg.device)

        def _clamp(acts: torch.Tensor, hook, *, _v: torch.Tensor = clamp_val) -> torch.Tensor:
            return _v.to(acts.dtype)

        clamp_hooks.append((f"blocks.{layer}.{model.feature_input_hook}", _clamp))

    # Run the constrained forward pass with both sets of hooks active
    constrained_logits = model.run_with_hooks(tokens_clean, fwd_hooks=clamp_hooks + inhibit_hooks)

    # Run unconstrained inhibition for comparison
    unconstrained_logits = inhibit_features(model, tokens_clean, feature_ids_by_layer, alpha=alpha)

    return constrained_logits, unconstrained_logits


# ---------------------------------------------------------------------------
# logit measurement
# ---------------------------------------------------------------------------


def compute_logit_diff(
    baseline_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    target_token_id: int,
    pos: int = -1,
) -> tuple[float, float]:
    """Measure the change in raw logit and softmax probability for a target token.

    Args:
        baseline_logits: Pre-intervention logits, shape (1, n_pos, d_vocab)
        intervention_logits: Post-intervention logits, same shape
        target_token_id: Vocabulary index of the token to track
        pos: Sequence position to evaluate (default: last position)

    Returns:
        (delta_logit, delta_prob) — signed differences (intervention minus baseline)
    """
    bl = baseline_logits[0, pos, target_token_id].item()
    il = intervention_logits[0, pos, target_token_id].item()
    delta_logit = il - bl

    bp = torch.softmax(baseline_logits[0, pos], dim=-1)[target_token_id].item()
    ip = torch.softmax(intervention_logits[0, pos], dim=-1)[target_token_id].item()
    delta_prob = ip - bp

    return delta_logit, delta_prob


# ---------------------------------------------------------------------------
# node serialization
# ---------------------------------------------------------------------------


def _node_list(
    graph: Graph,
    node_mask: torch.Tensor,
) -> list[dict]:
    """Flatten kept nodes from a pruned graph into a list of attribute dicts."""
    from mechinterp_qwen3.graph import Graph  # noqa: F401 (type only)

    nodes: list[dict] = []
    n_layers = graph.cfg.n_layers  # type: ignore[attr-defined]
    n_pos = graph.n_pos
    n_features = len(graph.selected_features)

    active_feats = graph.active_features  # (n_active, 3)  [layer, pos, feat_idx]
    act_values = graph.activation_values  # (n_active,)

    # Feature nodes
    for sel_i in range(n_features):
        if not bool(node_mask[sel_i]):
            continue
        global_idx = int(graph.selected_features[sel_i])
        layer = int(active_feats[global_idx, 0])
        pos = int(active_feats[global_idx, 1])
        feat_idx = int(active_feats[global_idx, 2])
        activation = float(act_values[global_idx])
        nodes.append(
            {
                "node_id": f"{layer}_{feat_idx}_{pos}",
                "feature": feat_idx,
                "layer": layer,
                "ctx_idx": pos,
                "feature_type": "CLT",
                "activation": activation,
            }
        )

    # Error nodes
    error_start = n_features
    for layer in range(n_layers):
        for pos in range(n_pos):
            node_idx = error_start + layer * n_pos + pos
            if not bool(node_mask[node_idx]):
                continue
            nodes.append(
                {
                    "node_id": f"E_{layer}_{pos}",
                    "feature": -1,
                    "layer": layer,
                    "ctx_idx": pos,
                    "feature_type": "mlp_error",
                    "activation": None,
                }
            )

    # Token embedding nodes
    tok_start = error_start + n_layers * n_pos
    tok_ids = graph.input_tokens.tolist()
    for pos in range(n_pos):
        node_idx = tok_start + pos
        if not bool(node_mask[node_idx]):
            continue
        nodes.append(
            {
                "node_id": f"embed_{pos}",
                "feature": tok_ids[pos] if pos < len(tok_ids) else -1,
                "layer": "E",
                "ctx_idx": pos,
                "feature_type": "embedding",
                "activation": None,
            }
        )

    # Logit nodes
    logit_start = tok_start + n_pos
    logit_ids = graph.logit_tokens.tolist()
    logit_probs = graph.logit_probabilities.tolist()
    for i, (tok_id, prob) in enumerate(zip(logit_ids, logit_probs, strict=False)):
        node_idx = logit_start + i
        if not bool(node_mask[node_idx]):
            continue
        nodes.append(
            {
                "node_id": f"logit_{tok_id}_{n_pos - 1}",
                "feature": tok_id,
                "layer": n_layers + 1,
                "ctx_idx": n_pos - 1,
                "feature_type": "logit",
                "token_prob": prob,
                "activation": prob,
            }
        )

    return nodes


# ---------------------------------------------------------------------------
# feature grouping (supernode assignment)
# ---------------------------------------------------------------------------


def propose_supernodes(
    nodes: list[dict],
    n_layers: int,
) -> dict[str, list[str]]:
    """Assign nodes to named functional groups based on type and layer depth.

    Groups:
    - ``embedding_inputs``:    token embedding nodes
    - ``error_nodes``:         transcoder error nodes
    - ``logit_nodes``:         output logit nodes
    - ``low_precision_sum``:   CLT features in the first third of layers
    - ``ones_digit_lookup``:   CLT features in the middle third of layers
    - ``sum_near_X``:          CLT features in the last third of layers
    - ``say_number_ending_Y``: CLT features in the final two layers
    """
    third = max(1, n_layers // 3)
    last_two = max(1, n_layers - 2)

    supernodes: dict[str, list[str]] = {
        "embedding_inputs": [],
        "error_nodes": [],
        "logit_nodes": [],
        "low_precision_sum": [],
        "ones_digit_lookup": [],
        "sum_near_X": [],
        "say_number_ending_Y": [],
    }

    for node in nodes:
        nid = node["node_id"]
        ft = node.get("feature_type", "")
        layer = node.get("layer")

        if ft == "logit":
            supernodes["logit_nodes"].append(nid)
        elif ft == "embedding":
            supernodes["embedding_inputs"].append(nid)
        elif ft == "mlp_error":
            supernodes["error_nodes"].append(nid)
        elif ft == "CLT":
            try:
                layer_val = int(layer)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                supernodes["sum_near_X"].append(nid)
                continue

            if layer_val >= last_two:
                supernodes["say_number_ending_Y"].append(nid)
            elif layer_val >= 2 * third:
                supernodes["sum_near_X"].append(nid)
            elif layer_val >= third:
                supernodes["ones_digit_lookup"].append(nid)
            else:
                supernodes["low_precision_sum"].append(nid)

    return supernodes


# ---------------------------------------------------------------------------
# main orchestrator
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_interventions(
    model: AttributionModel,
    graph: Graph,
    out_dir,
    *,
    prompt: str,
    perturbed_prompt: str,
    target_token_id: int | None = None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    alpha: float = 0.0,
    top_n_groups: int = 4,
) -> list[dict]:
    """Test causal importance of supernode groups via direct inhibition and constrained patching.

    For each of the top-N groups (ranked by feature count):
      - Baseline: plain forward pass on the clean prompt
      - Direct inhibition: scale group features by alpha
      - Constrained patch: clamp upstream layers to the perturbed run, then inhibit

    Writes results to:
      - ``out_dir/intervention_results.json``
      - ``out_dir/intervention_table.md``

    Args:
        model: AttributionModel instance
        graph: Attribution graph produced by ``run_attribution.attribute``
        out_dir: Output directory (str or Path)
        prompt: Clean prompt to analyse
        perturbed_prompt: A variant with one input operand changed
        target_token_id: Vocabulary index to track. Defaults to argmax at the last position.
        node_threshold: Fraction of total influence to retain when pruning nodes
        edge_threshold: Fraction of total influence to retain when pruning edges
        alpha: Feature scale factor for inhibition (0.0 = full suppression)
        top_n_groups: How many supernode groups to evaluate

    Returns:
        List of per-group result dicts
    """
    from mechinterp_qwen3.graph import prune_graph

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize prompts
    tokens_clean = tokenize_qwen_input(prompt, model.tokenizer, model.cfg.device).unsqueeze(0)
    tokens_perturbed = tokenize_qwen_input(
        perturbed_prompt, model.tokenizer, model.cfg.device
    ).unsqueeze(0)

    # Baseline
    baseline_logits = model(tokens_clean)

    # Resolve target token
    if target_token_id is None:
        target_token_id = int(baseline_logits[0, -1].argmax().item())
    _log.info(
        "Target token: %d (%r)",
        target_token_id,
        model.tokenizer.decode([target_token_id]),
    )

    # Prune the graph and extract node/supernode assignments
    prune_result = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    node_mask = prune_result.node_mask

    nodes = _node_list(graph, node_mask)
    supernodes = propose_supernodes(nodes, n_layers=graph.cfg.n_layers)  # type: ignore[attr-defined]

    # Index CLT nodes by node_id for fast lookup
    nid_to_lf: dict[str, tuple[int, int]] = {}
    for node in nodes:
        if node["feature_type"] == "CLT":
            nid_to_lf[node["node_id"]] = (int(node["layer"]), int(node["feature"]))

    # Focus on CLT groups only (exclude embedding, error, and logit groups)
    clt_groups = {
        k: v
        for k, v in supernodes.items()
        if k not in ("embedding_inputs", "error_nodes", "logit_nodes")
    }
    sorted_groups = sorted(clt_groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    groups_to_test = sorted_groups[:top_n_groups]

    results: list[dict] = []

    for group_name, node_ids in groups_to_test:
        lf_pairs = [nid_to_lf[nid] for nid in node_ids if nid in nid_to_lf]
        if not lf_pairs:
            _log.info("Group %r has no CLT features — skipping", group_name)
            continue

        min_layer = min(lf[0] for lf in lf_pairs)
        feat_by_layer: dict[int, list[int]] = {}
        for layer_idx, feat_idx in lf_pairs:
            feat_by_layer.setdefault(layer_idx, []).append(feat_idx)

        _log.info(
            "Testing %r: %d features across layers %s (intervene at L%d)",
            group_name,
            len(lf_pairs),
            sorted(feat_by_layer.keys()),
            min_layer,
        )

        # Unconstrained inhibition
        unconstrained_logits = inhibit_features(model, tokens_clean, feat_by_layer, alpha=alpha)
        d_logit_unc, d_prob_unc = compute_logit_diff(
            baseline_logits, unconstrained_logits, target_token_id
        )

        # Constrained patching
        constrained_logits, _ = constrained_patch(
            model,
            tokens_clean,
            tokens_perturbed,
            intervention_layer=min_layer,
            feature_ids_by_layer=feat_by_layer,
            alpha=alpha,
        )
        d_logit_con, d_prob_con = compute_logit_diff(
            baseline_logits, constrained_logits, target_token_id
        )

        row: dict = {
            "group": group_name,
            "n_features": len(lf_pairs),
            "intervention_layer": min_layer,
            "layers": sorted(feat_by_layer.keys()),
            "alpha": alpha,
            "delta_logit_unconstrained": round(d_logit_unc, 4),
            "delta_prob_unconstrained": round(d_prob_unc, 4),
            "delta_logit_constrained": round(d_logit_con, 4),
            "delta_prob_constrained": round(d_prob_con, 4),
            "constrained_differs_from_unconstrained": abs(d_logit_unc - d_logit_con) > 1e-4,
        }
        results.append(row)
        _log.info(
            "  Δlogit: unc=%.3f con=%.3f | Δp: unc=%.4f con=%.4f",
            d_logit_unc,
            d_logit_con,
            d_prob_unc,
            d_prob_con,
        )

    # Write outputs
    with open(out_dir / "intervention_results.json", "w") as f:
        _json.dump({"target_token_id": target_token_id, "results": results}, f, indent=2)
    _write_markdown_table(results, out_dir / "intervention_table.md", target_token_id)
    _log.info("Results written to %s", out_dir)
    return results


# ---------------------------------------------------------------------------
# output formatting
# ---------------------------------------------------------------------------


def _write_markdown_table(results: list[dict], path, target_token_id: int) -> None:
    """Render intervention results as a markdown table and write to disk."""
    lines = [
        f"# Intervention Results (target token id: {target_token_id})\n",
        "",
        "| Group | N features | Int. layer | Δlogit (unc.) | Δp (unc.) "
        "| Δlogit (con.) | Δp (con.) | Constrained differs? |",
        "|-------|-----------|------------|--------------|-----------|"
        "--------------|-----------|----------------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['group']} | {r['n_features']} | {r['intervention_layer']} "
            f"| {r['delta_logit_unconstrained']:+.3f} | {r['delta_prob_unconstrained']:+.4f} "
            f"| {r['delta_logit_constrained']:+.3f} | {r['delta_prob_constrained']:+.4f} "
            f"| {'✓' if r['constrained_differs_from_unconstrained'] else '✗'} |"
        )
    lines.append("")
    lines.append(
        "> **Constrained patching** replaces MLP inputs at layers before intervention_layer "
        "with cached perturbed activations, holding upstream state fixed. "
        "A '✓' indicates that constrained and unconstrained outcomes diverge."
    )
    Path(path).write_text("\n".join(lines))
