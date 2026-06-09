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

Both methods are anchored from the same anchor_mode used during delta extraction.

Usage:
python -m experiments.concept_localization.run_concept --concept <name> --causal --causal_pairs <n>

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from experiments.concept_localization.extract_deltas_generic import AnchorFactory, _resolve_anchor
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

log = logging.getLogger(__name__)


def _resolve_pair_anchor(
    pair,
    ids_pos: list[int],
    tokenizer,
    anchor_mode: str,
    anchor_factory: AnchorFactory | None,
) -> int:
    """Resolve the raw-token anchor position used by delta extraction."""
    return _resolve_anchor(ids_pos, tokenizer, anchor_mode, anchor_factory, pair)


# Surface forms (after stripping the Ġ / ▁ BPE space prefix) that mark the
# structural delimiter closing the concept expression: punctuation tokens and
# 'is' immediately before the trailing space.  Scanning backwards finds the
# true closing delimiter first, skipping any identical characters embedded
# earlier in the prompt (e.g. the ':' in 'calc: {a}+{b}=' or '=' in 'v1={v1}').
_DELIM_SURFACE = frozenset({":", "=", "?"})
_DELIM_COMPOUND = frozenset({")", ")=", ")?", ").", "),"})


def _find_expression_end(tokens: list[str]) -> int | None:
    """Return the index of the last structural delimiter by backward scan.

    The sweep is restricted to this delimiter and the trailing space
    (rel_pos 1 and 0 from the end).  Returns None when no hard punctuation
    is found, in which case callers fall back to the last token.
    """
    for i in range(len(tokens) - 1, -1, -1):
        surface = tokens[i].lstrip("Ġ▁ ")
        if surface in _DELIM_SURFACE or tokens[i] in _DELIM_COMPOUND:
            return i
    return None


@dataclass
class CausalScores:
    layers: list[int]
    patching_mean: dict[int, float] = field(default_factory=dict)
    patching_std: dict[int, float] = field(default_factory=dict)
    grad_dot_delta_mean: dict[int, float] = field(default_factory=dict)
    grad_dot_delta_std: dict[int, float] = field(default_factory=dict)
    n_pairs: int = 0


def _resolve_target(tokenizer, pair) -> tuple[list[int], int | None, int | None]:
    """Resolve the first diverging predicted token for a pair.

    Uses pair.predict_pos / predict_neg when set, otherwise falls back to
    label_pos / label_neg.  Returns (shared_prefix_ids, pos_target_id,
    neg_target_id) so callers can do teacher forcing: append shared_prefix_ids
    to both prompt sequences and measure the logit margin at position -1.
    NOT to be used in ablation studies!
    """
    pred_pos = pair.predict_pos if pair.predict_pos else pair.label_pos
    pred_neg = pair.predict_neg if pair.predict_neg else pair.label_neg

    ids_pos = tokenizer(pred_pos, add_special_tokens=False).input_ids
    ids_neg = tokenizer(pred_neg, add_special_tokens=False).input_ids

    if not ids_pos:
        return [], None, None

    k = 0
    while k < len(ids_pos) and k < len(ids_neg) and ids_pos[k] == ids_neg[k]:
        k += 1

    prefix = ids_pos[:k]
    target_pos = ids_pos[k] if k < len(ids_pos) else None
    target_neg = ids_neg[k] if k < len(ids_neg) else None
    return prefix, target_pos, target_neg


def run_activation_patching(
    model,
    pairs,
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
    anchor_mode: str = "delimiter",
    anchor_factory: AnchorFactory | None = None,
) -> dict[int, list[float]]:
    """Activation patching across layers at the selected delta-extraction anchor.

    For each pair and each layer L:
      - resolve p from anchor_mode exactly as delta extraction does
      - cache pos residuals at p for all layers in one forward pass
      - run neg once for baseline margin logit^+ - logit^-
      - for each L: re-run neg with h[L, p] swapped to the pos value

    Returns dict[layer -> list of per-pair delta-margin scores].
    """
    model.eval()
    scores: dict[int, list[float]] = {l: [] for l in layers}
    skipped = 0

    with torch.no_grad():
        for pair in tqdm(pairs, desc="Activation patching"):
            ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
            ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids

            if len(ids_pos) != len(ids_neg):
                skipped += 1
                continue

            try:
                anchor = _resolve_pair_anchor(pair, ids_pos, model.tokenizer, anchor_mode, anchor_factory)
            except ValueError:
                skipped += 1
                continue

            patch_pos = anchor
            if patch_pos >= len(ids_pos):
                skipped += 1
                continue

            prefix_ids, target_id, neg_id = _resolve_target(model.tokenizer, pair)
            if target_id is None:
                skipped += 1
                continue

            # Teacher forcing: extend both prompts with the shared answer prefix
            # so that position -1 predicts the first diverging output token.
            ids_pos_ext = ids_pos + prefix_ids
            ids_neg_ext = ids_neg + prefix_ids

            toks_pos = tokenize_qwen_input(ids_pos_ext, model.tokenizer, device).unsqueeze(0)
            toks_neg = tokenize_qwen_input(ids_neg_ext, model.tokenizer, device).unsqueeze(0)
            _pp = patch_pos + 1  # +1 for the sink token at position 0

            # One pass: cache pos residuals at patch_pos for all layers
            pos_cache: dict[int, torch.Tensor] = {}
            cache_hooks = [
                (
                    f"blocks.{l}.hook_resid_post",
                    lambda act, hook=None, _l=l, _p=_pp: (
                        pos_cache.update({_l: act[0, _p, :].clone()}) or act
                    ),
                )
                for l in layers
            ]
            model.run_with_hooks(toks_pos, fwd_hooks=cache_hooks)

            # Baseline margin on unpatched negative prompt
            base_logits = model(toks_neg)[0, -1]
            base_margin = base_logits[target_id].item()
            if neg_id is not None:
                base_margin -= base_logits[neg_id].item()

            # One patched pass per layer
            for l in layers:
                if l not in pos_cache:
                    continue
                vec = pos_cache[l]  # (d_model,)

                def make_hook(v, pos):
                    def hook_fn(act, hook=None):
                        act = act.clone()
                        act[:, pos, :] = v
                        return act

                    return hook_fn

                logits_patch = model.run_with_hooks(
                    toks_neg,
                    fwd_hooks=[(f"blocks.{l}.hook_resid_post", make_hook(vec, _pp))],
                )
                patch_margin = logits_patch[0, -1, target_id].item()
                if neg_id is not None:
                    patch_margin -= logits_patch[0, -1, neg_id].item()
                scores[l].append(patch_margin - base_margin)

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
    anchor_mode: str = "delimiter",
    anchor_factory: AnchorFactory | None = None,
) -> dict[int, list[float]]:
    """Gradient-dot-delta across layers.

    For each pair: one forward + one backward pass on the negative prompt.
    At each layer L, extracts grad of the margin logit(label_pos) - logit(label_neg)
    w.r.t. h[L, anchor, :] evaluated at h_L^neg, then computes the dot product
    with the precomputed concept delta δ_L.

    This is the first-order linear approximation of activation patching: both
    methods operate at the negative prompt's operating point and measure the
    change in margin caused by moving layer L's representation in the direction
    of δ_L.

    Returns dict[layer -> list of per-pair g·δ scores].
    """
    model.eval()
    scores: dict[int, list[float]] = {l: [] for l in layers}
    skipped = 0

    for pair in tqdm(pairs, desc="Gradient dot delta"):
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids

        if len(ids_pos) != len(ids_neg):
            skipped += 1
            continue

        try:
            anchor = _resolve_pair_anchor(pair, ids_pos, model.tokenizer, anchor_mode, anchor_factory)
        except ValueError:
            skipped += 1
            continue

        prefix_ids, target_id, neg_id = _resolve_target(model.tokenizer, pair)
        if target_id is None:
            skipped += 1
            continue

        # Teacher forcing: extend neg prompt with shared answer prefix so
        # position -1 predicts the first diverging output token.
        ids_neg_ext = ids_neg + prefix_ids
        toks_neg = tokenize_qwen_input(ids_neg_ext, model.tokenizer, device).unsqueeze(0)

        # Forward pass on negative prompt — hooks retain intermediate tensors for backward
        resid_refs: dict[int, torch.Tensor] = {}

        def make_retain_hook(layer_idx):
            def hook_fn(act, hook):
                act.retain_grad()
                resid_refs[layer_idx] = act
                return act

            return hook_fn

        hooks = [(f"blocks.{l}.hook_resid_post", make_retain_hook(l)) for l in layers]

        logits = model.run_with_hooks(toks_neg, fwd_hooks=hooks)
        objective = logits[0, -1, target_id]
        if neg_id is not None:
            objective = objective - logits[0, -1, neg_id]
        objective.backward()

        for l in layers:
            if l not in resid_refs:
                continue
            grad = resid_refs[l].grad
            if grad is None:
                log.debug("No grad at layer %d — possibly detached in model", l)
                continue
            if l not in layer_deltas:
                continue
            g = grad[0, anchor + 1, :].float()  # +1 for sink token
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
    anchor: str = "delimiter",
) -> tuple[dict[int, list[float]], list[str]]:
    """Attribution score at the anchor token and trailing space at a fixed layer.

    score[rel_pos] = cosine(grad[layer, pos], Δh[layer, pos])

    rel_pos 0 = last token (trailing space), rel_pos 1 = structural delimiter.
    With anchor="last", only the final token is scored.

    Cost per pair: 1 neg forward (no grad) + 1 pos forward + 1 backward.

    Returns (scores, token_labels) where token_labels are decoded from the first
    valid pair's positive prompt.
    """
    model.eval()
    scores: dict[int, list[float]] = {i: [] for i in range(2)}
    token_labels: list[str] = []
    skipped = 0
    _logged_delimiter = False

    for pair in tqdm(pairs, desc=f"Positional attribution L{layer}"):
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids

        if len(ids_pos) != len(ids_neg):
            skipped += 1
            continue
        seq_len = len(ids_pos)

        tokens = model.tokenizer.convert_ids_to_tokens(ids_pos)

        if anchor == "last":
            expr_end = seq_len - 1
        else:
            expr_end = _find_expression_end(tokens)
            if expr_end is None:
                expr_end = seq_len - 1  # fallback: trailing space

        if any(ids_pos[i] != ids_neg[i] for i in range(expr_end, seq_len)):
            log.debug("Tokens after expression end differ between pos/neg — skipping pair")
            skipped += 1
            continue

        n_valid = min(2, seq_len - expr_end)
        if n_valid == 0:
            skipped += 1
            continue

        if not _logged_delimiter:
            log.info(
                "Expression end: delimiter=%r at abs_pos=%d  |  valid tokens: %s  |  prompt: %r",
                tokens[expr_end],
                expr_end,
                tokens[expr_end:],
                pair.prompt_pos,
            )
            _logged_delimiter = True

        prefix_ids, target_id, _ = _resolve_target(model.tokenizer, pair)
        if target_id is None:
            skipped += 1
            continue

        # Teacher forcing: extend prompts so position -1 predicts the first
        # diverging output token.  Causal attention means h at earlier positions
        # is unaffected by the extension.
        ids_pos_ext = ids_pos + prefix_ids
        ids_neg_ext = ids_neg + prefix_ids
        toks_pos = tokenize_qwen_input(ids_pos_ext, model.tokenizer, device).unsqueeze(0)
        toks_neg = tokenize_qwen_input(ids_neg_ext, model.tokenizer, device).unsqueeze(0)

        if not token_labels:
            token_labels = [tokens[seq_len - 1 - i] for i in range(n_valid)]

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
        h_neg = h_neg_store[0]  # (seq_len + prefix_len, d_model); index by abs_pos

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

        for rel_pos in range(n_valid):
            abs_pos = seq_len - rel_pos  # seq_len - 1 - rel_pos + 1 for sink offset
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
    anchor: str = "delimiter",
) -> tuple[dict[int, dict[int, list[float]]], list[str]]:
    """Attribution scores at the anchor position and trailing space across all layers in one pass.

    Runs a single neg forward (no grad) and a single pos forward+backward per pair,
    registering hooks at every layer simultaneously.  Cost is O(1) passes regardless
    of how many layers are requested.

    score[layer][rel_pos] = cosine(grad[layer, pos], Δh[layer, pos])

    The sweep covers rel_pos 0 (trailing space) and rel_pos 1 (structural delimiter).
    With anchor="last" both positions collapse to the final token.  Only positions
    where pos/neg tokens are identical are included, so Δh is free of raw-token
    embedding confounds.

    Returns (scores, token_labels) where scores[layer][rel_pos] is a list of
    per-pair cosine alignment values.
    """
    model.eval()
    scores: dict[int, dict[int, list[float]]] = {l: {i: [] for i in range(2)} for l in layers}
    token_labels: list[str] = []
    skipped = 0
    _logged_delimiter = False

    for pair in tqdm(pairs, desc="Positional attribution sweep"):
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids

        if len(ids_pos) != len(ids_neg):
            skipped += 1
            continue
        seq_len = len(ids_pos)

        tokens = model.tokenizer.convert_ids_to_tokens(ids_pos)

        if anchor == "last":
            expr_end = seq_len - 1
        else:
            expr_end = _find_expression_end(tokens)
            if expr_end is None:
                expr_end = seq_len - 1  # fallback: trailing space

        if any(ids_pos[i] != ids_neg[i] for i in range(expr_end, seq_len)):
            log.debug("Tokens after expression end differ between pos/neg — skipping pair")
            skipped += 1
            continue

        if not _logged_delimiter:
            log.info(
                "Expression end: delimiter=%r at abs_pos=%d  |  valid tokens: %s  |  prompt: %r",
                tokens[expr_end],
                expr_end,
                tokens[expr_end:],
                pair.prompt_pos,
            )
            _logged_delimiter = True

        n_valid = min(2, seq_len - expr_end)
        if n_valid == 0:
            skipped += 1
            continue

        prefix_ids, target_id, _ = _resolve_target(model.tokenizer, pair)
        if target_id is None:
            skipped += 1
            continue

        # Teacher forcing: extend prompts so position -1 predicts the first
        # diverging output token.  Causal attention means h at earlier positions
        # is unaffected by the extension.
        ids_pos_ext = ids_pos + prefix_ids
        ids_neg_ext = ids_neg + prefix_ids
        toks_pos = tokenize_qwen_input(ids_pos_ext, model.tokenizer, device).unsqueeze(0)
        toks_neg = tokenize_qwen_input(ids_neg_ext, model.tokenizer, device).unsqueeze(0)

        if not token_labels:
            token_labels = [tokens[seq_len - 1 - i] for i in range(n_valid)]

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

            for rel_pos in range(n_valid):
                abs_pos = seq_len - rel_pos  # seq_len - 1 - rel_pos + 1 for sink offset
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
    anchor_mode: str = "delimiter",
    anchor_factory: AnchorFactory | None = None,
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
        tmpl_patching[t] = run_activation_patching(
            model,
            grp,
            layers,
            device,
            dtype,
            anchor_mode=anchor_mode,
            anchor_factory=anchor_factory,
        )
        tmpl_grad[t] = run_gradient_dot_delta(
            model,
            grp,
            layer_deltas,
            layers,
            device,
            dtype,
            anchor_mode=anchor_mode,
            anchor_factory=anchor_factory,
        )

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
