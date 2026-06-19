"""Triangle inequality concept dataset.

Linear structure: three sides a, b, c form a valid triangle iff
a+b>c, a+c>b, b+c>a. Single boundary region.

Pos: c_pos satisfies all inequalities given fixed a, b → "yes"
Neg: c_neg ≥ a+b (violates the main inequality) → "no"

a and b are 2-digit integers.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES: dict[str, tuple[str, str, str]] = {
    "T0": ("Triangle with sides {a},{b},{c} is valid? Answer yes or no: ", "yes", "no"),
    "T1": ("Can {a}, {b}, {c} be sides of a triangle? Answer yes or no: ", "yes", "no"),
    "T2": ("Does {a}+{b}>{c}? Answer yes or no: ", "yes", "no"),
}


def generate_triangle_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs with valid c_pos vs invalid c_neg, fixing a and b per pair."""
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
        a = rng.randint(15, 40)
        b = rng.randint(15, 40)

        min_c = abs(a - b) + 1
        max_c = a + b - 1
        if min_c > 99 or max_c < 10:
            continue
        c_pos = rng.randint(max(10, min_c), min(99, max_c))

        c_neg = rng.randint(a + b, a + b + 20)
        if c_neg > 99:
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
            fmt, predict_pos, predict_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt.format(a=a, b=b, c=c_pos),
                    prompt_neg=fmt.format(a=a, b=b, c=c_neg),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos=predict_pos,
                    predict_neg=predict_neg,
                    template=t,
                    meta={"a": a, "b": b, "c_pos": c_pos, "c_neg": c_neg},
                )
            )
            counts[t] += 1

    return pairs
