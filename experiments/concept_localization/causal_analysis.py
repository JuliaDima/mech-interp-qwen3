"""Causal analysis for concept localization.

Two complementary methods:

1. Activation patching (causal tracing)
   For each layer L, patch the positive prompt's residual stream at the anchor
   position into a neg-prompt forward pass.  Score = Δlogit(label_pos first
   token at last position).  A large positive score means that layer's
   representation causally drives the correct answer.

2. Gradient-dot-delta
   Run the positive prompt with gradient tracking.  For each layer L compute
   g_L = ∂logit(label_pos) / ∂h[L, anchor, :] then score = g_L · δ_L.
   This is the first-order prediction of how much moving the residual stream
   in the concept-delta direction would shift the output logit (one
   forward + one backward per pair) and interpretable as a linear causal
   weight.

Both methods are anchored at the same position used during delta extraction
(last token position where pos and neg prompts differ).

Usage:
python -m experiments.concept_localization.run_concept --concept <name> --causal --causal_pairs <n>

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from experiments.concept_localization.extract_deltas import _find_anchor

log = logging.getLogger(__name__)


@dataclass
class CausalScores:
    layers: list[int]
    patching_mean: dict[int, float] = field(default_factory=dict)
    patching_std: dict[int, float] = field(default_factory=dict)
    grad_dot_delta_mean: dict[int, float] = field(default_factory=dict)
    grad_dot_delta_std: dict[int, float] = field(default_factory=dict)
    n_pairs: int = 0


def _target_token_id(tokenizer, label: str) -> int | None:
    ids = tokenizer(label, add_special_tokens=False).input_ids
    return ids[0] if ids else None


def run_activation_patching(
    model,
    pairs,
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, list[float]]:
    """Activation patching across layers.

    For each pair and each layer L:
      - cache pos residual at anchor across all layers in one forward pass
      - run neg-prompt once for baseline logit
      - for each L: re-run neg with h[L, anchor] replaced by pos value

    Returns dict[layer -> list of per-pair Δlogit].
    """
    model.eval()
    scores: dict[int, list[float]] = {l: [] for l in layers}
    skipped = 0

    with torch.no_grad():
        for pair in tqdm(pairs, desc="Activation patching"):
            ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
            ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids

            anchor = _find_anchor(ids_pos, ids_neg)
            if anchor is None:
                skipped += 1
                continue

            target_id = _target_token_id(model.tokenizer, pair.label_pos)
            if target_id is None:
                skipped += 1
                continue

            toks_pos = torch.tensor([ids_pos], dtype=torch.long, device=device)
            toks_neg = torch.tensor([ids_neg], dtype=torch.long, device=device)

            # One pass: cache pos residuals at anchor for all layers
            pos_cache: dict[int, torch.Tensor] = {}
            cache_hooks = [
                (
                    f"blocks.{l}.hook_resid_post",
                    lambda act, hook, _l=l, _a=anchor: (
                        pos_cache.update({_l: act[0, _a, :].clone()}) or act
                    ),
                )
                for l in layers
            ]
            model.run_with_hooks(toks_pos, fwd_hooks=cache_hooks)

            # Baseline: neg without patching
            base_logit = model(toks_neg)[0, -1, target_id].item()

            # One patched pass per layer
            for l in layers:
                if l not in pos_cache:
                    continue
                vec = pos_cache[l]  # (d_model,)

                def make_hook(v, pos):
                    def hook_fn(act, hook):
                        act = act.clone()
                        act[:, pos, :] = v
                        return act

                    return hook_fn

                logits_patch = model.run_with_hooks(
                    toks_neg,
                    fwd_hooks=[(f"blocks.{l}.hook_resid_post", make_hook(vec, anchor))],
                )
                scores[l].append(logits_patch[0, -1, target_id].item() - base_logit)

    if skipped:
        log.warning("Activation patching: skipped %d pairs", skipped)
    return scores


def run_gradient_dot_delta(
    model,
    pairs,
    layer_deltas: dict[int, torch.Tensor],
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, list[float]]:
    """Gradient-dot-delta across layers.

    For each pair: one forward + one backward pass on the positive prompt.
    At each layer L, extracts grad of the target logit w.r.t. h[L, anchor, :]
    then computes dot product with the precomputed concept delta δ_L.

    Returns dict[layer -> list of per-pair g·δ scores].
    """
    model.eval()
    scores: dict[int, list[float]] = {l: [] for l in layers}
    skipped = 0

    for pair in tqdm(pairs, desc="Gradient dot delta"):
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids

        anchor = _find_anchor(ids_pos, ids_neg)
        if anchor is None:
            skipped += 1
            continue

        target_id = _target_token_id(model.tokenizer, pair.label_pos)
        if target_id is None:
            skipped += 1
            continue

        toks_pos = torch.tensor([ids_pos], dtype=torch.long, device=device)

        # Forward pass — hooks retain intermediate tensors for backward
        resid_refs: dict[int, torch.Tensor] = {}

        def make_retain_hook(layer_idx):
            def hook_fn(act, hook):
                act.retain_grad()
                resid_refs[layer_idx] = act
                return act

            return hook_fn

        hooks = [(f"blocks.{l}.hook_resid_post", make_retain_hook(l)) for l in layers]

        logits = model.run_with_hooks(toks_pos, fwd_hooks=hooks)
        logits[0, -1, target_id].backward()

        for l in layers:
            if l not in resid_refs:
                continue
            grad = resid_refs[l].grad
            if grad is None:
                log.debug("No grad at layer %d — possibly detached in model", l)
                continue
            if l not in layer_deltas:
                continue
            g = grad[0, anchor, :].float()
            delta = layer_deltas[l].to(device=g.device, dtype=torch.float32)
            scores[l].append(torch.dot(g.detach(), delta).item())

        model.zero_grad()

    if skipped:
        log.warning("Gradient-dot-delta: skipped %d pairs", skipped)
    return scores


def aggregate(
    raw: dict[int, list[float]],
    layers: list[int],
) -> tuple[dict[int, float], dict[int, float]]:
    """Return (mean, std) dicts over pairs for each layer."""
    means, stds = {}, {}
    for l in layers:
        vals = raw.get(l, [])
        if vals:
            t = torch.tensor(vals, dtype=torch.float32)
            means[l] = t.mean().item()
            stds[l] = t.std().item() if len(vals) > 1 else 0.0
        else:
            means[l] = 0.0
            stds[l] = 0.0
    return means, stds


def run_positional_attribution(
    model,
    pairs,
    layer: int,
    device: torch.device,
    dtype: torch.dtype,
    n_tail: int = 6,
) -> tuple[dict[int, list[float]], list[str]]:
    """Attribution score at each of the last n_tail token positions at a fixed layer.

    score[rel_pos] = grad[layer, pos] · (h_pos[layer, pos] - h_neg[layer, pos])

    rel_pos 0 = last token, 1 = second-to-last, ...

    Cost per pair: 1 neg forward (no grad) + 1 pos forward + 1 backward.

    Returns (scores, token_labels) where token_labels are decoded from the first
    valid pair's positive prompt (representative of the template structure).
    """
    model.eval()
    scores: dict[int, list[float]] = {i: [] for i in range(n_tail)}
    token_labels: list[str] = []
    skipped = 0

    for pair in tqdm(pairs, desc=f"Positional attribution L{layer}"):
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids

        if len(ids_pos) != len(ids_neg):
            skipped += 1
            continue
        seq_len = len(ids_pos)
        if seq_len < n_tail:
            skipped += 1
            continue

        target_id = _target_token_id(model.tokenizer, pair.label_pos)
        if target_id is None:
            skipped += 1
            continue

        toks_pos = torch.tensor([ids_pos], dtype=torch.long, device=device)
        toks_neg = torch.tensor([ids_neg], dtype=torch.long, device=device)

        # Decode token labels from first valid pair
        if not token_labels:
            tokens = model.tokenizer.convert_ids_to_tokens(ids_pos)
            token_labels = [tokens[seq_len - 1 - i] for i in range(n_tail)]

        # Neg forward — cache hidden states at this layer (no grad needed)
        h_neg_store: list[torch.Tensor] = []

        def _cache_neg(act, hook):
            h_neg_store.append(act[0].detach().clone())
            return act

        with torch.no_grad():
            model.run_with_hooks(
                toks_neg, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", _cache_neg)]
            )

        if not h_neg_store:
            skipped += 1
            continue
        h_neg = h_neg_store[0]  # (seq_len, d_model)

        # Pos forward — retain activations for backward
        h_pos_store: list[torch.Tensor] = []

        def _retain_pos(act, hook):
            act.retain_grad()
            h_pos_store.append(act)
            return act

        logits = model.run_with_hooks(
            toks_pos, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", _retain_pos)]
        )
        logits[0, -1, target_id].backward()

        if not h_pos_store or h_pos_store[0].grad is None:
            model.zero_grad()
            skipped += 1
            continue

        h_pos = h_pos_store[0][0]  # (seq_len, d_model)
        grad = h_pos_store[0].grad[0]  # (seq_len, d_model)

        for rel_pos in range(n_tail):
            abs_pos = seq_len - 1 - rel_pos
            g = grad[abs_pos].float().detach()
            diff = (h_pos[abs_pos] - h_neg[abs_pos].to(device=g.device)).float().detach()
            g_norm = g.norm()
            diff_norm = diff.norm()
            if g_norm > 1e-8 and diff_norm > 1e-8:
                score = torch.dot(g / g_norm, diff / diff_norm).item()
            else:
                score = 0.0
            scores[rel_pos].append(score)

        model.zero_grad()

    if skipped:
        log.warning("Positional attribution: skipped %d pairs", skipped)
    return scores, token_labels


def run_positional_attribution_sweep(
    model,
    pairs,
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
    n_tail: int = 6,
) -> tuple[dict[int, dict[int, list[float]]], list[str]]:
    """Attribution scores at each of the last n_tail positions across all layers in one pass.

    Runs a single neg forward (no grad) and a single pos forward+backward per pair,
    registering hooks at every layer simultaneously.  Cost is O(1) passes regardless
    of how many layers are requested.

    score[layer][rel_pos] = cosine(grad[layer, pos], Δh[layer, pos])

    Returns (scores, token_labels) where scores[layer][rel_pos] is a list of
    per-pair cosine alignment values.
    """
    model.eval()
    scores: dict[int, dict[int, list[float]]] = {l: {i: [] for i in range(n_tail)} for l in layers}
    token_labels: list[str] = []
    skipped = 0

    layer_set = set(layers)

    for pair in tqdm(pairs, desc="Positional attribution sweep"):
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids

        if len(ids_pos) != len(ids_neg):
            skipped += 1
            continue
        seq_len = len(ids_pos)
        if seq_len < n_tail:
            skipped += 1
            continue

        target_id = _target_token_id(model.tokenizer, pair.label_pos)
        if target_id is None:
            skipped += 1
            continue

        toks_pos = torch.tensor([ids_pos], dtype=torch.long, device=device)
        toks_neg = torch.tensor([ids_neg], dtype=torch.long, device=device)

        if not token_labels:
            tokens = model.tokenizer.convert_ids_to_tokens(ids_pos)
            token_labels = [tokens[seq_len - 1 - i] for i in range(n_tail)]

        # Neg forward: cache residuals at all layers in one pass
        h_neg_all: dict[int, torch.Tensor] = {}
        hooks_neg = []
        for l in layers:

            def _cache_neg(act, hook, _l=l):
                h_neg_all[_l] = act[0].detach().clone()
                return act

            hooks_neg.append((f"blocks.{l}.hook_resid_post", _cache_neg))

        with torch.no_grad():
            model.run_with_hooks(toks_neg, fwd_hooks=hooks_neg)

        # Pos forward: retain grads at all layers, then one backward
        h_pos_all: dict[int, torch.Tensor] = {}
        hooks_pos = []
        for l in layers:

            def _retain_pos(act, hook, _l=l):
                act.retain_grad()
                h_pos_all[_l] = act
                return act

            hooks_pos.append((f"blocks.{l}.hook_resid_post", _retain_pos))

        logits = model.run_with_hooks(toks_pos, fwd_hooks=hooks_pos)
        logits[0, -1, target_id].backward()

        valid = all(
            l in h_pos_all and h_pos_all[l].grad is not None and l in h_neg_all for l in layers
        )
        if not valid:
            model.zero_grad()
            skipped += 1
            continue

        for l in layers:
            h_pos = h_pos_all[l][0]  # (seq_len, d_model)
            grad_l = h_pos_all[l].grad[0]  # (seq_len, d_model)
            h_neg = h_neg_all[l]  # (seq_len, d_model)

            for rel_pos in range(n_tail):
                abs_pos = seq_len - 1 - rel_pos
                g = grad_l[abs_pos].float().detach()
                diff = (h_pos[abs_pos] - h_neg[abs_pos].to(device=g.device)).float().detach()
                g_norm = g.norm()
                diff_norm = diff.norm()
                if g_norm > 1e-8 and diff_norm > 1e-8:
                    score = torch.dot(g / g_norm, diff / diff_norm).item()
                else:
                    score = 0.0
                scores[l][rel_pos].append(score)

        model.zero_grad()

    if skipped:
        log.warning("Positional attribution sweep: skipped %d pairs", skipped)
    return scores, token_labels


def run_causal_analysis(
    model,
    pairs,
    layer_deltas: dict[int, torch.Tensor],
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
    max_pairs: int | None = None,
) -> dict[str, CausalScores]:
    """Run both causal analyses per template and return dict[key -> CausalScores].

    Keys: "all" (aggregate) and one entry per template name found in pairs.
    Runs each template group once — no redundant forward passes.  The "all"
    aggregate is formed by merging the per-template raw score lists.
    """
    subset = pairs[:max_pairs] if max_pairs is not None else pairs

    template_keys = list(dict.fromkeys(p.template for p in subset))
    groups = {t: [p for p in subset if p.template == t] for t in template_keys}

    tmpl_patching: dict[str, dict[int, list[float]]] = {}
    tmpl_grad: dict[str, dict[int, list[float]]] = {}
    for t, grp in groups.items():
        log.info("Causal analysis — template %s (%d pairs)", t, len(grp))
        tmpl_patching[t] = run_activation_patching(model, grp, layers, device, dtype)
        tmpl_grad[t] = run_gradient_dot_delta(model, grp, layer_deltas, layers, device, dtype)

    def _make_scores(patch_raw, grad_raw) -> CausalScores:
        p_mean, p_std = aggregate(patch_raw, layers)
        g_mean, g_std = aggregate(grad_raw, layers)
        n = max((len(v) for v in patch_raw.values()), default=0)
        return CausalScores(
            layers=layers,
            patching_mean=p_mean,
            patching_std=p_std,
            grad_dot_delta_mean=g_mean,
            grad_dot_delta_std=g_std,
            n_pairs=n,
        )

    results: dict[str, CausalScores] = {}

    # "all": merge raw score lists across templates
    all_patch: dict[int, list[float]] = {l: [] for l in layers}
    all_grad: dict[int, list[float]] = {l: [] for l in layers}
    for t in template_keys:
        for l in layers:
            all_patch[l].extend(tmpl_patching[t].get(l, []))
            all_grad[l].extend(tmpl_grad[t].get(l, []))
    results["all"] = _make_scores(all_patch, all_grad)

    for t in template_keys:
        results[t] = _make_scores(tmpl_patching[t], tmpl_grad[t])

    return results
