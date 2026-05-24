"""Decimal expansion termination concept dataset.

Modular structure: 1/n terminates iff n = 2^a * 5^b (only factors 2 and 5).

Pos: n_pos has only factors 2 and 5 → terminating decimal → "yes"
Neg: n_neg has another prime factor → repeating decimal → "no"

Pairs are matched by proximity in value to control for magnitude confound.
Pairs with different digit counts for n_pos vs n_neg are skipped.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

_POS = [2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 80, 100, 125, 160, 200, 250]
_NEG = [
    3,
    6,
    7,
    9,
    11,
    12,
    13,
    14,
    15,
    18,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    33,
    35,
    42,
    44,
    45,
    48,
    49,
    52,
    54,
    56,
    63,
    66,
    70,
    72,
    77,
    84,
    90,
    98,
    99,
    105,
    110,
    112,
    126,
    132,
    140,
    147,
    150,
    154,
    168,
    175,
    176,
    189,
    196,
    198,
    210,
    220,
    224,
    231,
    240,
    242,
    245,
]

TEMPLATES = {
    "T0": ("calc: 1/{n}= ", "calc: 1/{n}= "),
    "T1": ("1 divided by {n} has a finite decimal: ", "1 divided by {n} has a finite decimal: "),
    "T2": ("does 1/{n} have a terminating decimal? ", "does 1/{n} have a terminating decimal? "),
}


def generate_decimal_pairs(
    n_per_template: int = 80,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs n_pos (terminates) vs n_neg (repeats), matched by magnitude."""
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
        n_pos = rng.choice(_POS)
        close_negs = sorted(_NEG, key=lambda x: abs(x - n_pos))[:5]
        n_neg = rng.choice(close_negs)

        if len(str(n_pos)) != len(str(n_neg)):
            continue

        key = (n_pos, n_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(n=n_pos),
                    prompt_neg=fmt_neg.format(n=n_neg),
                    label_pos="yes",
                    label_neg="no",
                    template=t,
                    meta={"n_pos": n_pos, "n_neg": n_neg},
                )
            )
            counts[t] += 1

    return pairs
