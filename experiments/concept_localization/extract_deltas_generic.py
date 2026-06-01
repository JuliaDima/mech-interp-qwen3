"""Extract per-layer residual-stream deltas for generic ConceptPair datasets.

Mirrors extract_deltas.py but uses ConceptPair (prompt_pos / prompt_neg)
instead of CarryPair.  Template logic is dropped; all pairs are treated as
a single group.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import torch
from tqdm import tqdm

# Concept-specific anchor factory: (pair, tokenizer) → {mode_name: token_position}
AnchorFactory = Callable[[object, object], dict[str, int]]

from experiments.concept_localization.concept_pair import ConceptPair
from experiments.concept_localization.extract_deltas import LayerDeltas
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

log = logging.getLogger(__name__)

# Token strings that mark "end of expression / about to answer".
# The model compresses the computed result onto these tokens (per biology-of-LLMs paper).
_DELIMITER_STRINGS = (
    "=",
    "Ġ=",
    ":",
    "Ġ:",
    "?",
    "Ġ?",
    "\n",
    "Ċ",
)

_DELIMITER_CHARS = {"=", ":", "?", ")", "\n"}


def resolve_anchor_token(
    prompt: str,
    tokenizer,
    anchor_mode: str,
    anchor_factory: AnchorFactory | None = None,
    pair: object = None,
) -> tuple[int, str]:
    """Return (0-indexed position, decoded token string) for the resolved anchor."""
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    pos = _resolve_anchor(ids, tokenizer, anchor_mode, anchor_factory, pair)
    tok_str = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(ids[pos])])
    return pos, tok_str


def _resolve_anchor(
    ids: list[int],
    tokenizer,
    anchor_mode: str,
    anchor_factory: AnchorFactory | None,
    pair: object = None,
) -> int:
    """Resolve anchor_mode to a 0-indexed token position for one sequence.

    Raises ValueError for unrecognised anchor_mode rather than silently
    falling back to a delimiter scan.
    """
    if anchor_factory and pair is not None:
        positions = anchor_factory(pair, tokenizer)
        if anchor_mode in positions:
            return positions[anchor_mode]
        m = re.match(r"^(.*?)(\d+)$", anchor_mode)
        if m:
            prefix, n = m.group(1), int(m.group(2))
            for k in range(n - 1, 0, -1):
                candidate = f"{prefix}{k}"
                if candidate in positions:
                    log.warning("Anchor '%s' not in positions — using '%s' instead", anchor_mode, candidate)
                    return positions[candidate]
    if anchor_mode == "last":
        return len(ids) - 1
    if anchor_mode == "delimiter":
        return _find_delimiter_anchor(ids, tokenizer)
    try:
        return min(int(anchor_mode), len(ids) - 1)
    except ValueError:
        raise ValueError(
            f"Unknown anchor_mode {anchor_mode!r}. "
            f"Expected 'last', 'delimiter', an integer position string, "
            f"or a concept-specific mode defined in ANCHOR_MODES."
        ) from None


def _find_delimiter_anchor(ids: list[int], tokenizer) -> int:
    """Return the position of the last expression-end delimiter token.

    Matches any token whose decoded string ends with a structural delimiter
    character.  This handles merged tokens such as '():' or ')?' that would
    be missed by exact token-ID lookup.  Falls back to the final token.
    """
    tokens = tokenizer.convert_ids_to_tokens(ids)
    for i in range(len(tokens) - 1, -1, -1):
        tok = tokens[i].replace("Ġ", "").replace("Ċ", "\n")
        if tok and tok[-1] in _DELIMITER_CHARS:
            return i
    return len(ids) - 1  # fallback: last token


def extract_layer_deltas_generic(
    model,
    pairs: list[ConceptPair],
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
    per_template: bool = True,
    anchor_mode: str = "delimiter",
    delta_aggregation: str = "mean",
    anchor_factory: AnchorFactory | None = None,
) -> dict[str, LayerDeltas]:
    """Capture residual stream at the anchor token; return aggregated pos − neg delta.

    anchor_mode="delimiter" (default) — last structural delimiter token ('=', ':',
                              '?', ')').  Falls back to the last token when none found.
    anchor_mode="last"      — absolute last token of the sequence.
    anchor_mode="<int>"     — explicit 0-indexed token position.
    anchor_mode=<concept key> — resolved via anchor_factory(pair, tokenizer) if provided;
                              the factory returns a dict of named positions computed from
                              the template structure and actual operand values.

    delta_aggregation="mean" (default) — plain mean of per-pair deltas δ̄ = mean_i(δ_i).
    delta_aggregation="cosine_weighted" — weight each δ_i by its cosine alignment with
                              the initial mean direction, then renormalise.

    Pairs where tokenization lengths differ are always skipped.

    Returns a dict keyed by "all" (aggregate) and per template name if
    per_template=True and pairs have a non-empty template field.
    """

    template_keys = list(dict.fromkeys(p.template for p in pairs)) if per_template else []
    all_keys = ["all"] + template_keys

    buckets: dict[str, dict[int, dict[str, list[torch.Tensor]]]] = {
        key: {layer: {"pos": [], "neg": []} for layer in layers} for key in all_keys
    }

    model.eval()
    skipped = 0

    with torch.no_grad():
        for pair in tqdm(pairs, desc="Extracting deltas"):
            ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
            ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids

            if len(ids_pos) != len(ids_neg):
                skipped += 1
                continue

            # Anchor is computed on raw ids; +1 for the sink token prepended by tokenize_qwen_input
            anchor = (
                _resolve_anchor(ids_pos, model.tokenizer, anchor_mode, anchor_factory, pair) + 1
            )

            tmpl_key = pair.template

            for ids, bucket_name in [(ids_pos, "pos"), (ids_neg, "neg")]:
                input_ids = tokenize_qwen_input(ids, model.tokenizer, device).unsqueeze(0)
                cache: dict[int, torch.Tensor] = {}

                hooks = [
                    (
                        f"blocks.{layer}.hook_resid_post",
                        lambda act, hook, _l=layer, _pos=anchor: (
                            cache.update({_l: act[0, _pos, :].detach().clone()})
                            if _pos < act.shape[1]
                            else None
                        )
                        or act,
                    )
                    for layer in layers
                ]
                model.run_with_hooks(input_ids, fwd_hooks=hooks)

                for layer in layers:
                    if layer not in cache:
                        continue
                    vec = cache[layer].to(dtype=dtype, device="cpu")
                    buckets["all"][layer][bucket_name].append(vec)
                    if per_template and tmpl_key in buckets:
                        buckets[tmpl_key][layer][bucket_name].append(vec)

    if skipped:
        log.warning("Skipped %d pairs (tokenization length mismatch)", skipped)

    results: dict[str, LayerDeltas] = {}
    for key, layer_buckets in buckets.items():
        ld = LayerDeltas(skipped=skipped if key == "all" else 0)
        for layer in layers:
            pos_vecs = layer_buckets[layer]["pos"]
            neg_vecs = layer_buckets[layer]["neg"]
            if not pos_vecs or not neg_vecs:
                continue
            n = len(pos_vecs)
            assert n == len(neg_vecs), (
                f"Layer {layer}: {n} pos vectors but {len(neg_vecs)} neg vectors — "
                "pos/neg should always be appended together"
            )
            pair_deltas = torch.stack(pos_vecs) - torch.stack(neg_vecs)  # (n, d_model)
            mean_delta = pair_deltas.mean(0)  # (d_model,)
            mean_norm = mean_delta.norm()

            cos: torch.Tensor | None = None
            if mean_norm > 1e-8:
                cos = torch.nn.functional.cosine_similarity(
                    pair_deltas.float(),
                    mean_delta.float().unsqueeze(0).expand_as(pair_deltas.float()),
                    dim=-1,
                )  # (n,)
                ld.mean_pair_cos[layer] = cos.mean().item()

                if delta_aggregation == "cosine_weighted":
                    weights = cos.clamp(min=0)  # zero-out anti-aligned pairs
                    w_sum = weights.sum()
                    if w_sum > 1e-8:
                        mean_delta = (pair_deltas.float() * weights.unsqueeze(1)).sum(0) / w_sum

            ld.delta[layer] = mean_delta
            ld.n_pairs = max(ld.n_pairs, n)
            all_vecs = pos_vecs + neg_vecs
            ld.mean_act_norm[layer] = torch.stack(all_vecs).norm(dim=-1).mean().item()
        results[key] = ld
        log.info("key=%s  n_pairs=%d  layers_with_delta=%d", key, ld.n_pairs, len(ld.delta))

    return results
