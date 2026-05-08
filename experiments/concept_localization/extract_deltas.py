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

from experiments.concept_localization.dataset import CarryPair

log = logging.getLogger(__name__)


@dataclass
class LayerDeltas:
    delta: dict[int, torch.Tensor] = field(default_factory=dict)  # layer → (d_model,)
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
    pairs: list[CarryPair],
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
    per_template: bool = True,
) -> dict[str, LayerDeltas]:
    """Capture residual stream at the anchor token; return mean carry − no-carry.

    Keys in the returned dict:
      "all"        — aggregate over all pairs and templates
      "T0", "T1"…  — per-template (only if per_template=True)
    """
    template_keys = [str(p.template) for p in pairs]
    all_keys = ["all"] + (list(dict.fromkeys(template_keys)) if per_template else [])

    # buckets[key][layer]["carry"|"no_carry"] = list of (d_model,) tensors
    buckets: dict[str, dict[int, dict[str, list[torch.Tensor]]]] = {
        key: {layer: {"carry": [], "no_carry": []} for layer in layers} for key in all_keys
    }

    model.eval()
    skipped = 0

    with torch.no_grad():
        for pair in tqdm(pairs, desc="Extracting deltas"):
            ids_carry = model.tokenizer(pair.prompt_carry, add_special_tokens=False).input_ids
            ids_no_carry = model.tokenizer(pair.prompt_no_carry, add_special_tokens=False).input_ids

            anchor = _find_anchor(ids_carry, ids_no_carry)
            if anchor is None:
                skipped += 1
                continue

            tmpl_key = str(pair.template)

            for ids, bucket_name in [(ids_carry, "carry"), (ids_no_carry, "no_carry")]:
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
            carry_vecs = layer_buckets[layer]["carry"]
            no_carry_vecs = layer_buckets[layer]["no_carry"]
            if not carry_vecs or not no_carry_vecs:
                continue
            n = min(len(carry_vecs), len(no_carry_vecs))
            ld.delta[layer] = torch.stack(carry_vecs[:n]).mean(0) - torch.stack(
                no_carry_vecs[:n]
            ).mean(0)
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
