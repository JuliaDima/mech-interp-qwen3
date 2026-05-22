"""Extract per-layer residual-stream deltas for carry vs no-carry prompts.

For every (carry, no-carry) pair the anchor token is the last position where
the two tokenizations differ — i.e. the units digit of B, which is the first
moment both operand digits are visible and carry is computable.

The mean delta across all pairs at each layer is the concept direction.
Per-template deltas are also kept so consistency across templates can be
verified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class LayerDeltas:
    delta: dict[int, torch.Tensor] = field(default_factory=dict)  # layer → (d_model,)
    mean_act_norm: dict[int, float] = field(
        default_factory=dict
    )  # layer → mean ‖h‖ across all pairs
    n_pairs: int = 0
    skipped: int = 0


def _find_anchor(ids_a: list[int], ids_b: list[int]) -> int | None:
    """Return the last position where the two token sequences differ."""
    if len(ids_a) != len(ids_b):
        return None
    diffs = [i for i, (x, y) in enumerate(zip(ids_a, ids_b, strict=False)) if x != y]
    return diffs[-1] if diffs else None


def extract_layer_deltas(
    model,
    pairs: list,
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
    per_template: bool = True,
) -> dict[str, LayerDeltas]:
    """Capture residual stream at the anchor token; return mean pos − neg delta.

    Keys in the returned dict:
      "all"        — aggregate over all pairs and templates
      "T0", "T1"…  — per-template (only if per_template=True)

    Pairs must have prompt_pos and prompt_neg attributes (ConceptPair).
    """
    template_keys = [str(p.template) for p in pairs]
    all_keys = ["all"] + (list(dict.fromkeys(template_keys)) if per_template else [])

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

            tmpl_key = str(pair.template)

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

    for key, ld in results.items():
        log.info(
            "Key=%s  n_pairs=%d  layers_with_delta=%d",
            key,
            ld.n_pairs,
            len(ld.delta),
        )
    return results
