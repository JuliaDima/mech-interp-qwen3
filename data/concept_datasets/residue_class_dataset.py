"""Residue class concept dataset.

Improvements over original:
- Modulus m varies across {5, 7, 11} instead of always 7
- Residue classes r_pos and r_neg are sampled randomly (not fixed 1 vs 6)
- Number of digits varies uniformly across {2, 3, 4}
- Gap between a_pos and a_neg varies (r_neg - r_pos, not fixed 5)
- Templates: compact math / natural language / question form
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

MODULI = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

TEMPLATES = {
    "T0": ("calc: {a}%{m}= ", "calc: {a}%{m}= "),
    "T1": ("the remainder of {a} divided by {m} is: ", "the remainder of {a} divided by {m} is: "),
    "T2": ("what is {a} mod {m}? ", "what is {a} mod {m}? "),
}


def generate_residue_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Generate pairs a_pos ≡ r_pos (mod m) vs a_neg ≡ r_neg (mod m).

    Modulus m, residue classes r_pos/r_neg, and number of digits all vary.
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
        m = rng.choice(MODULI)
        r_pos = rng.randint(0, m - 1)
        r_neg = rng.choice([r for r in range(m) if r != r_pos])

        n_digits = rng.choice([1, 2, 3])
        lo, hi = max(1, 10 ** (n_digits - 1)), 10**n_digits - 1

        min_base = max(0, (lo - r_pos + m - 1) // m)
        max_base = (hi - r_pos) // m
        if min_base > max_base:
            continue

        base = rng.randint(min_base, max_base)
        a_pos = base * m + r_pos
        a_neg = base * m + r_neg

        if not (lo <= a_pos <= hi and lo <= a_neg <= hi):
            continue
        if a_pos % m != r_pos or a_neg % m != r_neg:
            continue

        key = (base, m, r_pos)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(a=a_pos, m=m),
                    prompt_neg=fmt_neg.format(a=a_neg, m=m),
                    label_pos=str(r_pos),
                    label_neg=str(r_neg),
                    template=t,
                    meta={"a_pos": a_pos, "a_neg": a_neg, "m": m, "r_pos": r_pos, "r_neg": r_neg},
                )
            )
            counts[t] += 1

    return pairs
