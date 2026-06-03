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

# Large denominators so that both p_pos and p_neg are two-digit (10-99)
DENOMINATORS = [11, 13, 14, 17, 19, 21, 22, 23, 26, 29, 31, 34, 37, 38, 41, 43]

TEMPLATES = {
    "T0": ("Yes or No: geometric series ratio {r} converges: ", "Yes or No: geometric series ratio {r} converges: "),
    "T1": (
        "Yes or No: the series with ratio {r} has a finite sum: ",
        "Yes or No: the series with ratio {r} has a finite sum: ",
    ),
    "T2": (
        "Yes or No: does the geometric series with ratio {r} converge? ",
        "Yes or No: does the geometric series with ratio {r} converge? ",
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

        # restrict to two-digit numerators for both pos and neg
        p_pos_choices = [p for p in range(10, q) if _gcd(p, q) == 1]
        if not p_pos_choices:
            continue
        p_pos = rng.choice(p_pos_choices)

        p_neg_choices = [p for p in range(q + 1, q * 2)
                         if _gcd(p, q) == 1 and 10 <= p <= 99]
        if not p_neg_choices:
            continue
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
                    predict_pos="Yes",
                    predict_neg="No",
                    template=t,
                    meta={"p_pos": p_pos, "p_neg": p_neg, "q": q},
                )
            )
            counts[t] += 1

    return pairs
