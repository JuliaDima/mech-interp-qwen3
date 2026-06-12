"""Perfect square concept dataset.

Pos: n = k² for some integer k → "Yes"
Neg: n is not a perfect square → "No"

All templates use "Yes or No: " prefix so the model reliably outputs "Yes" or
"No" rather than a number or other continuation.

n_pos = k² for k in [10, 31], giving 3-digit numbers [100, 961];
n_neg = n_pos + offset for a random small offset in {-3,-2,-1,1,2,3}
that avoids other perfect squares. Offset is varied to avoid the fixed-gap confound.
"""

from __future__ import annotations

import math
import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES: dict[str, tuple[str, str, str]] = {
    "T0": ("Is sqrt({n}) an integer? Answer yes or no: ", "yes", "no"),
    "T1": ("sqrt({n}) is an integer? Answer yes or no: ", "yes", "no"),
    "T2": ("is {n} a perfect square? Answer yes or no: ", "yes", "no"),
}


def _is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def generate_perfect_square_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs n_pos=k² vs n_neg=k²+offset (not a perfect square)."""
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
        k = rng.randint(10, 31)
        n_pos = k * k

        offsets = [-3, -2, -1, 1, 2, 3]
        rng.shuffle(offsets)
        n_neg = None
        for offset in offsets:
            candidate = n_pos + offset
            if candidate > 0 and not _is_perfect_square(candidate):
                n_neg = candidate
                break
        if n_neg is None:
            continue

        if len(str(n_pos)) != len(str(n_neg)):
            continue

        key = (n_pos, n_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt, predict_pos, predict_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt.format(n=n_pos),
                    prompt_neg=fmt.format(n=n_neg),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos=predict_pos,
                    predict_neg=predict_neg,
                    template=t,
                    meta={"k": k, "n_pos": n_pos, "n_neg": n_neg},
                )
            )
            counts[t] += 1

    return pairs
