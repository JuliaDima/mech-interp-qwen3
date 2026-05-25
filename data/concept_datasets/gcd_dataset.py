"""GCD concept dataset.

Fixed divisor g=7 so every pair tests the same binary concept direction:
is a divisible by 7 (gcd=7) vs coprime to 7 (gcd=1)?

Fixing g ensures a single coherent delta direction and consistent target
tokens ("7" vs "1") across all pairs, which makes the causal analysis valid.
a varies freely over 2-4 digit numbers for statistical diversity.
"""

from __future__ import annotations

import random
from math import gcd as _gcd

from experiments.concept_localization.concept_pair import ConceptPair

G = 7

TEMPLATES = {
    "T0": ("calc: gcd({a},7)= ", "calc: gcd({a},7)= "),
    "T1": ("the gcd of {a} and 7 is: ", "the gcd of {a} and 7 is: "),
    "T2": ("what is gcd({a},7)? ", "what is gcd({a},7)? "),
}


def generate_gcd_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs where a_pos is divisible by 7 (gcd=7) vs a_neg coprime to 7 (gcd=1).

    a_neg = a_pos + offset, offset in [1, 6], ensuring same digit count and
    that a_neg is not divisible by 7.
    """
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[int] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 500 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        n_digits = rng.choice([2, 3, 4])
        lo, hi = 10 ** (n_digits - 1), 10**n_digits - 1

        min_mult = (lo + G - 1) // G
        max_mult = hi // G
        if min_mult > max_mult:
            continue

        mult = rng.randint(min_mult, max_mult)
        a_pos = mult * G

        offset = rng.randint(1, G - 1)
        a_neg = a_pos + offset
        if not (lo <= a_neg <= hi):
            continue
        if _gcd(a_neg, G) != 1:
            continue
        if a_pos in seen:
            continue
        seen.add(a_pos)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(a=a_pos),
                    prompt_neg=fmt_neg.format(a=a_neg),
                    label_pos=str(G),
                    label_neg="1",
                    predict_pos=str(G),
                    predict_neg="1",
                    template=t,
                    meta={"a_pos": a_pos, "a_neg": a_neg, "g": G, "offset": offset},
                )
            )
            counts[t] += 1

    return pairs
