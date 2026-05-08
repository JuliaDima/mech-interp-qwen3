"""GCD concept dataset.

Pairs: gcd(a_pos, 7) = 7  (7 | a_pos)
  vs   gcd(a_neg, 7) = 1  (7 ∤ a_neg, a_neg = a_pos + 1)

Only the last digit of a changes between pos and neg.
Three surface-form templates test that the delta direction is consistent.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES = {
    "T0": ("gcd({a},7)= ", "gcd({a},7)= "),
    "T1": ("gcd({a}, 7) = ", "gcd({a}, 7) = "),
    "T2": ("compute: gcd({a},7)= ", "compute: gcd({a},7)= "),
}


def generate_gcd_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[int] = set()

    attempts = 0
    while (
        any(sum(1 for p in pairs if p.template == t) < n_per_template for t in templates)
        and attempts < n_per_template * 200
    ):
        attempts += 1
        base = rng.randint(15, 142)  # 15*7=105 … 142*7=994
        a_pos = base * 7
        a_neg = a_pos + 1
        if not (100 <= a_pos <= 999 and a_neg <= 999):
            continue
        if a_neg % 7 == 0:
            continue
        if a_pos in seen:
            continue
        seen.add(a_pos)

        for t in templates:
            if sum(1 for p in pairs if p.template == t) >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(a=a_pos),
                    prompt_neg=fmt_neg.format(a=a_neg),
                    label_pos="7",
                    label_neg="1",
                    template=t,
                    meta={"a_pos": a_pos, "a_neg": a_neg},
                )
            )

    return pairs
