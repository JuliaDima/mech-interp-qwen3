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
from experiments.concept_localization.extract_deltas import LayerDeltas, _find_anchor

log = logging.getLogger(__name__)


def extract_layer_deltas_generic(
    model,
    pairs: list[ConceptPair],
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
    per_template: bool = True,
) -> dict[str, LayerDeltas]:
    """Capture residual stream at the anchor token; return mean pos − neg delta.

    Anchor = last token position where pos and neg tokenizations differ.
    Pairs where tokenization lengths differ are skipped.

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

            anchor = _find_anchor(ids_pos, ids_neg)
            if anchor is None:
                skipped += 1
                continue

            tmpl_key = pair.template

            for ids, bucket_name in [(ids_pos, "pos"), (ids_neg, "neg")]:
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                cache: dict[int, torch.Tensor] = {}

                hooks = [
                    (
                        f"blocks.{layer}.hook_resid_post",
                        lambda act, hook, _l=layer, _pos=anchor: (
                            cache.update({_l: act[0, _pos, :].detach().clone()}) or act
                        ),
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
        results[key] = ld
        log.info("key=%s  n_pairs=%d  layers_with_delta=%d", key, ld.n_pairs, len(ld.delta))

    return results
