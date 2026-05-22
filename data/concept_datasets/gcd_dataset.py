"""GCD concept dataset.

Improvements over original:
- Divisor g varies across {3, 5, 7, 11} instead of always 7
- Offset between pos/neg is random in [1, g-1] instead of always +1
- Number of digits varies uniformly across {2, 3, 4}
- Templates: compact math / natural language / question form
"""

from __future__ import annotations

import random
from math import gcd as _gcd

from experiments.concept_localization.concept_pair import ConceptPair

DIVISORS = [3, 5, 7, 11]

TEMPLATES = {
    "T0": ("gcd({a},{g})= ", "gcd({a},{g})= "),
    "T1": ("the gcd of {a} and {g} is ", "the gcd of {a} and {g} is "),
    "T2": ("what is gcd({a},{g})? ", "what is gcd({a},{g})? "),
}


def generate_gcd_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Generate pairs where a_pos is divisible by g (gcd=g) and a_neg is not (gcd=1).

    Divisor g varies across {3, 5, 7, 11}; offset between pos/neg varies in [1, g-1].
    """
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 200 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        g = rng.choice(DIVISORS)
        n_digits = rng.choice([1, 2, 3])
        lo, hi = max(1, 10 ** (n_digits - 1)), 10**n_digits - 1

        min_mult = (lo + g - 1) // g
        max_mult = hi // g
        if min_mult > max_mult:
            continue

        mult = rng.randint(min_mult, max_mult)
        a_pos = mult * g

        offset = rng.randint(1, g - 1)
        a_neg = a_pos + offset
        if not (lo <= a_pos <= hi and lo <= a_neg <= hi):
            continue
        if _gcd(a_neg, g) != 1:
            continue

        key = (a_pos, g)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(a=a_pos, g=g),
                    prompt_neg=fmt_neg.format(a=a_neg, g=g),
                    label_pos=str(g),
                    label_neg="1",
                    template=t,
                    meta={"a_pos": a_pos, "a_neg": a_neg, "g": g, "offset": offset},
                )
            )
            counts[t] += 1

    return pairs
