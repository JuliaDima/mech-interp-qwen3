"""Negation scope concept dataset.

Pos: statement is true (m ≤ n, so "m is not greater than n" is correct).
Neg: statement is false (m > n, so "m is not greater than n" is incorrect).

Improved: values extend to two-digit numbers for more diversity.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES: dict[str, tuple[str, str, str]] = {
    "T0": ("{m} is not greater than {n}? Answer only with yes or no:", "yes", "no"),
    "T1": ("{m} does not exceed {n}? Answer only with yes or no: ", "yes", "no"),
    "T2": ("Is {m} at most {n}? Answer only with yes or no: ", "yes", "no"),
}


def generate_negation_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs: n_pos > m (statement true) vs n_neg < m (statement false).

    m is fixed; only n varies. Values are 3 digit integers.
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
        n_digits = 3
        lo = 10 ** (n_digits - 1) if n_digits > 1 else 1
        hi = 10**n_digits - 1

        m = rng.randint(lo + 1, hi - 1)
        n_pos = rng.randint(m + 1, hi)
        n_neg = rng.randint(lo, m - 1)
        if n_pos > hi or n_neg < lo:
            continue
        if len(str(n_pos)) != len(str(n_neg)):
            continue

        key = (m, n_pos, n_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt, predict_pos, predict_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt.format(m=m, n=n_pos),
                    prompt_neg=fmt.format(m=m, n=n_neg),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos=predict_pos,
                    predict_neg=predict_neg,
                    template=t,
                    meta={"m": m, "n_pos": n_pos, "n_neg": n_neg},
                )
            )
            counts[t] += 1

    return pairs
