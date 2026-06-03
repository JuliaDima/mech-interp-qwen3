"""Transitive ordering concept dataset (improved).

Improvements over original:
- Number of digits: 3
  across different magnitude scales
- Larger combinatorial space reduces repetition
- Templates: compact math / natural language / question form
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES = {
    "T0": ("True or False: {a}>{b}>{c}: ", "True or False: {a}>{b}>{c}: "),
    "T1": (
        "True or False: {a} is greater than {b} which is greater than {c}: ",
        "True or False: {a} is greater than {b} which is greater than {c}: ",
    ),
    "T2": (
        "True or False: given {a}>{b} and {b}>{c}: ",
        "True or False: given {a}>{b} and {b}>{c}: ",
    ),
}

_SCALES = [
    (100, 999),  # 3-digit
]


def generate_ordering_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
    n_c_per_ab: int = 5,
) -> list[ConceptPair]:
    """Pairs: chain a>b>c holds (c_pos < b) vs chain breaks (c_neg > b).

    Only c varies; a and b are held fixed within each pair.
    c_pos and c_neg must have the same digit count for anchor detection.

    n_c_per_ab controls how many distinct (c_pos, c_neg) values are sampled
    per (a, b) context.  Values >1 allow the null permutation test to group
    by (a, b) via --context_keys a,b, holding the context fixed in the null
    exactly as it is held fixed in real pairs.
    """
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    lo, hi = _SCALES[0]
    gap = max(1, (hi - lo) // 10)

    # ── Step 1: sample unique (a, b) contexts ────────────────────────────────
    n_ab = max(1, (n_per_template + n_c_per_ab - 1) // n_c_per_ab)
    ab_list: list[tuple[int, int]] = []
    seen_ab: set[tuple[int, int]] = set()

    for _ in range(n_ab * 2000):
        if len(ab_list) >= n_ab:
            break
        a = rng.randint(lo + 3 * gap, hi)
        b = rng.randint(lo + gap, a - gap)
        if b <= lo or b >= a:
            continue
        if b - 1 < lo or a - 1 < b + 1:
            continue
        if (a, b) in seen_ab:
            continue
        seen_ab.add((a, b))
        ab_list.append((a, b))

    # ── Step 2: for each (a, b), sample n_c_per_ab valid (c_pos, c_neg) pairs ─
    pairs: list[ConceptPair] = []
    counts = {t: 0 for t in templates}

    for a, b in ab_list:
        seen_c: set[tuple[int, int]] = set()
        n_found = 0
        for _ in range(n_c_per_ab * 500):
            if n_found >= n_c_per_ab:
                break
            c_pos = rng.randint(lo, b - 1)
            c_neg = rng.randint(b + 1, a - 1)
            if c_pos < lo or c_neg >= a:
                continue
            if len(str(c_pos)) != len(str(c_neg)):
                continue
            if (c_pos, c_neg) in seen_c:
                continue
            seen_c.add((c_pos, c_neg))
            n_found += 1

            for t in templates:
                if counts[t] >= n_per_template:
                    continue
                fmt_pos, fmt_neg = TEMPLATES[t]
                pairs.append(
                    ConceptPair(
                        prompt_pos=fmt_pos.format(a=a, b=b, c=c_pos),
                        prompt_neg=fmt_neg.format(a=a, b=b, c=c_neg),
                        label_pos="True",
                        label_neg="False",
                        predict_pos="True",
                        predict_neg="False",
                        template=t,
                        meta={"a": a, "b": b, "c_pos": c_pos, "c_neg": c_neg, "n_digits": len(str(a))},
                    )
                )
                counts[t] += 1

    return pairs
