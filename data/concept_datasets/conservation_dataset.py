"""Energy conservation concept dataset.

Pos: bounce height strictly less than drop height (energy conserved / plausible).
Neg: bounce height greater than drop height (violates energy conservation).

Improved: h ranges from 2-digit values to allow more diversity.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES: dict[str, tuple[str, str, str]] = {
    "T0": ("Ball drop={h}, bounce={b}. valid? Answer yes or no:", "yes", "no"),
    "T1": ("A ball dropped from {h}m rebounds to {b}m. Is this physical? Answer yes or no:", "yes", "no"),
    "T2": ("Does bouncing to {b}m after dropping from {h}m conserve energy? Answer yes or no:", "yes", "no"),
}




def generate_conservation_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs: b_pos < h (valid) vs b_neg > h (violates conservation).

    Only b varies between pos and neg; h is fixed within each pair.
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
        n_digits = 2
        lo = 10 ** (n_digits - 1) if n_digits > 1 else 1
        hi = 10**n_digits - 1

        h = rng.randint(lo + 2, hi)
        b_pos = rng.randint(lo, h - 1)
        b_neg = rng.randint(h + 1, min(hi * 2, h + hi))
        if len(str(b_pos)) != len(str(b_neg)):
            continue

        key = (h, b_pos, b_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt, predict_pos, predict_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt.format(h=h, b=b_pos),
                    prompt_neg=fmt.format(h=h, b=b_neg),
                    label_pos="Yes",
                    label_neg="No",
                    predict_pos=predict_pos,
                    predict_neg=predict_neg,
                    template=t,
                    meta={"h": h, "b_pos": b_pos, "b_neg": b_neg},
                )
            )
            counts[t] += 1

    return pairs
