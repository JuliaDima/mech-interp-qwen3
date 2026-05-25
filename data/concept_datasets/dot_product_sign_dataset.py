"""Dot product sign concept dataset.

Linear structure: dot product a·b > 0 iff angle between vectors is acute.
Single threshold at 0.

Pos: a·b > 0 → acute angle → "yes"
Neg: a·b < 0 → obtuse angle → "no"

Vector a is fixed per pair; only b1 varies between pos and neg while b2
is held constant. Pairs where b1_pos and b1_neg have different signs
(which changes token count due to '-' prefix) are skipped.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES = {
    "T0": ("calc: ({a1},{a2})·({b1},{b2})= ", "calc: ({a1},{a2})·({b1},{b2})= "),
    "T1": (
        "Yes or No: vectors ({a1},{a2}) and ({b1},{b2}) form an acute angle: ",
        "Yes or No: vectors ({a1},{a2}) and ({b1},{b2}) form an acute angle: ",
    ),
    "T2": (
        "Yes or No: dot product of ({a1},{a2}) and ({b1},{b2}) is positive: ",
        "Yes or No: dot product of ({a1},{a2}) and ({b1},{b2}) is positive: ",
    ),
}


def generate_dot_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs with dot(a, b_pos) > 0 and dot(a, b_neg) < 0.

    a is fixed per pair; b1 varies; b2 is held constant.
    """
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 300 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        a1 = rng.randint(1, 3)
        a2 = rng.choice([-2, -1, 1, 2])
        b2 = rng.randint(-2, 2)

        b1_pos_choices = [b for b in range(1, 7) if a1 * b + a2 * b2 > 0]
        b1_neg_choices = [b for b in range(1, 7) if a1 * b + a2 * b2 < 0]
        if not b1_pos_choices or not b1_neg_choices:
            continue

        b1_pos = rng.choice(b1_pos_choices)
        b1_neg = rng.choice(b1_neg_choices)

        key = (a1, a2, b2, b1_pos, b1_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            dot_pos = a1 * b1_pos + a2 * b2
            dot_neg = a1 * b1_neg + a2 * b2
            if t == "T0":
                pred_pos = str(dot_pos)
                pred_neg = str(dot_neg)
            else:
                pred_pos = "Yes"
                pred_neg = "No"
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(a1=a1, a2=a2, b1=b1_pos, b2=b2),
                    prompt_neg=fmt_neg.format(a1=a1, a2=a2, b1=b1_neg, b2=b2),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos=pred_pos,
                    predict_neg=pred_neg,
                    template=t,
                    meta={"a1": a1, "a2": a2, "b2": b2, "b1_pos": b1_pos, "b1_neg": b1_neg},
                )
            )
            counts[t] += 1

    return pairs
