"""Negation scope concept dataset.

Pairs: '{m} is not greater than {n_pos}: '  (m ≤ n_pos → True)
  vs   '{m} is not greater than {n_neg}: '  (m > n_neg → False)

Only n changes (1 token, single digit); m is fixed.
By the anchor token the model has seen m and the full negated phrase.

Three templates rephrase 'not greater than' (≤) in different surface forms.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES = {
    "T0": ("{m} is not greater than {n}: ", "{m} is not greater than {n}: "),
    "T1": ("{m} does not exceed {n}: ", "{m} does not exceed {n}: "),
    "T2": ("{m} is at most {n}: ", "{m} is at most {n}: "),
}


def generate_negation_pairs(
    n_per_template: int = 50,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int, int]] = set()

    attempts = 0
    while (
        any(sum(1 for p in pairs if p.template == t) < n_per_template for t in templates)
        and attempts < n_per_template * 200
    ):
        attempts += 1
        m = rng.randint(2, 7)
        n_pos = rng.randint(m + 1, 9)
        n_neg = rng.randint(1, m - 1)
        if n_pos > 9 or n_neg < 1:
            continue
        key = (m, n_pos, n_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if sum(1 for p in pairs if p.template == t) >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(m=m, n=n_pos),
                    prompt_neg=fmt_neg.format(m=m, n=n_neg),
                    label_pos="True",
                    label_neg="False",
                    template=t,
                    meta={"m": m, "n_pos": n_pos, "n_neg": n_neg},
                )
            )

    return pairs
