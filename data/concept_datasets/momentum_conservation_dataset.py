"""Momentum conservation concept dataset.

For unit masses (m=1), total momentum = v1 + v2.
An elastic collision exchanges velocities: v1_after = v2, v2_after = v1.

Pos: v2_after = v1 (momentum conserved) → "yes"
Neg: v2_after = v1 + delta for random delta ≠ 0 (momentum violated) → "no"

v1 and v2 are single-digit non-negative integers; v2_after varies between
pos and neg. Both are kept single-digit so tokenization lengths match.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES = {
    "T0": (
        "Yes or No: v1={v1}→{v1a}, v2={v2}→{v2a}. conserved: ",
        "Yes or No: v1={v1}→{v1a}, v2={v2}→{v2a}. conserved: ",
    ),
    "T1": (
        "Yes or No: object 1 goes {v1} then {v1a}, object 2 goes {v2} then {v2a}. momentum conserved: ",
        "Yes or No: object 1 goes {v1} then {v1a}, object 2 goes {v2} then {v2a}. momentum conserved: ",
    ),
    "T2": ("Yes or No: does {v1a}+{v2a} equal {v1}+{v2}? ", "Yes or No: does {v1a}+{v2a} equal {v1}+{v2}? "),
}


def generate_momentum_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Elastic exchange (conserved) vs perturbed outcome (violated) pairs."""
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int, int]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 300 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        n_digits = rng.choice([1, 2])
        lo = 10 ** (n_digits - 1) if n_digits > 1 else 1
        hi = 10**n_digits - 1

        v1 = rng.randint(lo + 1, hi)
        v2 = rng.randint(lo, v1 - 1)

        v1a = v2
        v2a_pos = v1

        delta = rng.choice([-2, -1, 1, 2])
        v2a_neg = v2a_pos + delta
        if not (lo <= v2a_neg <= hi):
            continue
        if len(str(v2a_pos)) != len(str(v2a_neg)):
            continue
        if v2a_neg == v2a_pos:
            continue

        key = (v1, v2, delta)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(v1=v1, v2=v2, v1a=v1a, v2a=v2a_pos),
                    prompt_neg=fmt_neg.format(v1=v1, v2=v2, v1a=v1a, v2a=v2a_neg),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos="Yes",
                    predict_neg="No",
                    template=t,
                    meta={
                        "v1": v1,
                        "v2": v2,
                        "v1a": v1a,
                        "v2a_pos": v2a_pos,
                        "v2a_neg": v2a_neg,
                        "delta": delta,
                    },
                )
            )
            counts[t] += 1

    return pairs
