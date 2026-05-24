"""Extract per-layer residual-stream deltas for generic ConceptPair datasets.

Mirrors extract_deltas.py but uses ConceptPair (prompt_pos / prompt_neg)
instead of CarryPair.  Template logic is dropped; all pairs are treated as
a single group.
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm

from experiments.concept_localization.concept_pair import ConceptPair
from experiments.concept_localization.extract_deltas import LayerDeltas

log = logging.getLogger(__name__)

# Token strings that mark "end of expression / about to answer".
# The model compresses the computed result onto these tokens (per biology-of-LLMs paper).
_DELIMITER_STRINGS = (
    "=", "Ġ=",
    ":", "Ġ:",
    "?", "Ġ?",
    ")", "Ġ)",
    ")=", "Ġ)=",
    ")?", "Ġ)?",
    ").", "Ġ).",
    "),", "Ġ),",
    "\n", "Ċ",
)


def resolve_anchor_token(prompt: str, tokenizer, anchor_mode: str) -> tuple[int, str]:
    """Return (0-indexed position, decoded token string) for the resolved anchor.

    Mirrors the per-pair anchor logic in extract_layer_deltas_generic so callers
    can inspect and save the anchor without re-running the full extraction.
    """
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    _fixed_pos: int | None = None
    if anchor_mode not in ("delimiter", "last"):
        try:
            _fixed_pos = int(anchor_mode)
        except ValueError:
            pass

    if _fixed_pos is not None:
        pos = min(_fixed_pos, len(ids) - 1)
    elif anchor_mode == "last":
        pos = len(ids) - 1
    else:
        pos = _find_delimiter_anchor(ids, tokenizer)

    tok_str = tokenizer.convert_tokens_to_string(
        [tokenizer.convert_ids_to_tokens(ids[pos])]
    )
    return pos, tok_str


def _find_delimiter_anchor(ids: list[int], tokenizer) -> int:
    """Return the position of the last expression-end delimiter token.

    Searches backwards for structural punctuation.  Falls back to the final
    token if none found.
    """
    delim_ids = {
        tokenizer.convert_tokens_to_ids(s)
        for s in _DELIMITER_STRINGS
        if tokenizer.convert_tokens_to_ids(s) != tokenizer.unk_token_id
    }
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] in delim_ids:
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
) -> dict[str, LayerDeltas]:
    """Capture residual stream at the anchor token; return mean pos − neg delta.

    anchor_mode="delimiter" (default) — last structural delimiter token ('=', ':',
                              '?', ')').  Per the biology-of-LLMs paper, models
                              store computed results on punctuation tokens that
                              close the expression.  Falls back to the last token
                              when no delimiter is found.
    anchor_mode="last"      — absolute last token of the sequence.
    anchor_mode="<int>"     — explicit 0-indexed token position (e.g. "5" for the
                              ones digit of the first operand in "calc: 36+59=").

    Pairs where tokenization lengths differ are always skipped.

    Returns a dict keyed by "all" (aggregate) and per template name if
    per_template=True and pairs have a non-empty template field.
    """
    _fixed_pos: int | None = None
    if anchor_mode not in ("delimiter", "last"):
        try:
            _fixed_pos = int(anchor_mode)
        except ValueError:
            raise ValueError(
                f"anchor_mode must be 'delimiter', 'last', or an integer position string, got {anchor_mode!r}"
            )

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

            if _fixed_pos is not None:
                anchor = _fixed_pos
            elif anchor_mode == "last":
                anchor = len(ids_pos) - 1
            else:
                anchor = _find_delimiter_anchor(ids_pos, model.tokenizer)

            tmpl_key = pair.template

            for ids, bucket_name in [(ids_pos, "pos"), (ids_neg, "neg")]:
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                cache: dict[int, torch.Tensor] = {}

                hooks = [
                    (
                        f"blocks.{layer}.hook_resid_post",
                        lambda act, hook, _l=layer, _pos=anchor: (
                            cache.update({_l: act[0, _pos, :].detach().clone()})
                            if _pos < act.shape[1] else None
                        ) or act,
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
            n = min(len(pos_vecs), len(neg_vecs))
            ld.delta[layer] = torch.stack(pos_vecs[:n]).mean(0) - torch.stack(neg_vecs[:n]).mean(0)
            ld.n_pairs = max(ld.n_pairs, n)
            all_vecs = pos_vecs + neg_vecs
            ld.mean_act_norm[layer] = torch.stack(all_vecs).norm(dim=-1).mean().item()
        results[key] = ld
        log.info("key=%s  n_pairs=%d  layers_with_delta=%d", key, ld.n_pairs, len(ld.delta))

    return results
