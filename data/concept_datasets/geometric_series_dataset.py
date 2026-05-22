"""Geometric series convergence concept dataset.

Linear structure: converges iff |r| < 1. Single real threshold.

Pos: ratio r = p/q with p < q (|r| < 1) → converges → "yes"
Neg: ratio r = p/q with p > q (|r| > 1) → diverges → "no"

Ratios are expressed as "p/q" with single-digit p and q sharing the same
denominator within each pair, so token count is equal (anchor = numerator).
"""

from __future__ import annotations

import random
from math import gcd as _gcd

from experiments.concept_localization.concept_pair import ConceptPair

DENOMINATORS = [2, 3, 4, 5, 6, 7, 8]

TEMPLATES = {
    "T0": ("geometric series ratio {r} converges: ", "geometric series ratio {r} converges: "),
    "T1": (
        "the series with ratio {r} has a finite sum: ",
        "the series with ratio {r} has a finite sum: ",
    ),
    "T2": (
        "does the geometric series with ratio {r} converge? ",
        "does the geometric series with ratio {r} converge? ",
    ),
}


def generate_geometric_pairs(
    n_per_template: int = 80,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Convergent (p/q, p<q) vs divergent (p/q, p>q) ratio pairs.

    Denominator q is shared within each pair so token length matches.
    """
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int, int]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 200 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        q = rng.choice(DENOMINATORS)

        p_pos_choices = [p for p in range(1, q) if _gcd(p, q) == 1]
        p_neg_choices = [p for p in range(q + 1, q * 2) if _gcd(p, q) == 1 and p <= 9]
        if not p_pos_choices or not p_neg_choices:
            continue

        p_pos = rng.choice(p_pos_choices)
        p_neg = rng.choice(p_neg_choices)
        r_pos = f"{p_pos}/{q}"
        r_neg = f"{p_neg}/{q}"

        key = (p_pos, p_neg, q)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(r=r_pos),
                    prompt_neg=fmt_neg.format(r=r_neg),
                    label_pos="yes",
                    label_neg="no",
                    template=t,
                    meta={"p_pos": p_pos, "p_neg": p_neg, "q": q},
                )
            )
            counts[t] += 1

    return pairs
