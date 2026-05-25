"""Residue class concept dataset.

Fixed modulus m=7, fixed residue classes r_pos=1 vs r_neg=6 so every pair
tests the same binary concept direction: a ≡ 1 (mod 7) vs a ≡ 6 (mod 7).

Fixing m and (r_pos, r_neg) ensures a single coherent delta direction and
consistent target tokens ("1" vs "6") across all pairs, making the causal
analysis valid. a varies freely over 2-4 digit numbers for statistical diversity.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

M = 7
R_POS = 1
R_NEG = 6

TEMPLATES = {
    "T0": ("calc: {a}%7= ", "calc: {a}%7= "),
    "T1": ("the remainder of {a} divided by 7 is: ", "the remainder of {a} divided by 7 is: "),
    "T2": ("what is {a} mod 7? ", "what is {a} mod 7? "),
}


def generate_residue_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs a_pos ≡ 1 (mod 7) vs a_neg ≡ 6 (mod 7), same base k.

    a_pos = 7k + 1, a_neg = 7k + 6, so they differ by 5 and share the same
    digit count. k is sampled uniformly over 2-4 digit ranges.
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

        k_min = (lo - R_POS + M - 1) // M
        k_max = (hi - R_NEG) // M
        if k_min > k_max:
            continue

        k = rng.randint(k_min, k_max)
        a_pos = M * k + R_POS
        a_neg = M * k + R_NEG

        if not (lo <= a_pos <= hi and lo <= a_neg <= hi):
            continue
        if k in seen:
            continue
        seen.add(k)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(a=a_pos),
                    prompt_neg=fmt_neg.format(a=a_neg),
                    label_pos=str(R_POS),
                    label_neg=str(R_NEG),
                    predict_pos=str(R_POS),
                    predict_neg=str(R_NEG),
                    template=t,
                    meta={"a_pos": a_pos, "a_neg": a_neg, "m": M, "r_pos": R_POS, "r_neg": R_NEG},
                )
            )
            counts[t] += 1

    return pairs
