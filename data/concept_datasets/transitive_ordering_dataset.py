"""Transitive ordering concept dataset (improved).

Improvements over original:
- Number of digits sampled uniformly from {1, 2, 3} so the concept is tested
  across different magnitude scales
- Larger combinatorial space reduces repetition
- Templates: compact math / natural language / question form
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES = {
    "T0": ("True or False: {a}>{b}>{c}: ", "True or False: {a}>{b}>{c}: "),
    "T1": (
        "True or False: {a} is greater than {b} which is greater than {c}: ",
        "True or False: {a} is greater than {b} which is greater than {c}: ",
    ),
    "T2": (
        "True or False: given {a}>{b} and {b}>{c}: ",
        "True or False: given {a}>{b} and {b}>{c}: ",
    ),
}

_SCALES = [
    (100, 999),  # 3-digit
]


def generate_ordering_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs: chain a>b>c holds (c_pos < b) vs chain breaks (c_neg > b).

    Only c varies; a and b are held fixed within each pair.
    n_digits is sampled uniformly from {1, 2, 3} per pair; c_pos and c_neg
    must have the same digit count for the anchor detection to succeed.
    """
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int, int, int]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 300 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        lo, hi = rng.choice(_SCALES)
        gap = max(1, (hi - lo) // 10)

        a = rng.randint(lo + 3 * gap, hi)
        b = rng.randint(lo + gap, a - gap)
        if b <= lo or b >= a:
            continue

        if b - 1 < lo or a - 1 < b + 1:
            continue
        c_pos = rng.randint(lo, b - 1)
        c_neg = rng.randint(b + 1, a - 1)
        if c_pos < lo or c_neg >= a:
            continue
        if len(str(c_pos)) != len(str(c_neg)):
            continue

        key = (a, b, c_pos, c_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(a=a, b=b, c=c_pos),
                    prompt_neg=fmt_neg.format(a=a, b=b, c=c_neg),
                    label_pos="True",
                    label_neg="False",
                    predict_pos="True",
                    predict_neg="False",
                    template=t,
                    meta={"a": a, "b": b, "c_pos": c_pos, "c_neg": c_neg, "n_digits": len(str(a))},
                )
            )
            counts[t] += 1

    return pairs
