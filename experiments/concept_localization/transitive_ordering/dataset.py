"""Transitive ordering concept dataset.

Pairs: chain a>b>c holds (c < b)  vs  chain breaks (c > b).
Only c changes; a and b are fixed.

Three templates vary how the chain is expressed.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES = {
    "T0": ("{a}>{b}>{c}: ", "{a}>{b}>{c}: "),
    "T1": ("{a} > {b} > {c}: ", "{a} > {b} > {c}: "),
    "T2": ("{a}>{b} and {b}>{c}: ", "{a}>{b} and {b}>{c}: "),
}


def generate_ordering_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int, int, int]] = set()

    attempts = 0
    while (
        any(sum(1 for p in pairs if p.template == t) < n_per_template for t in templates)
        and attempts < n_per_template * 200
    ):
        attempts += 1
        a = rng.randint(3, 9)
        b = rng.randint(1, a - 2)
        if b < 2:
            continue
        c_pos = rng.randint(0, b - 1)
        c_neg = rng.randint(b + 1, a)
        if c_neg >= a:
            continue
        key = (a, b, c_pos, c_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if sum(1 for p in pairs if p.template == t) >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(a=a, b=b, c=c_pos),
                    prompt_neg=fmt_neg.format(a=a, b=b, c=c_neg),
                    label_pos="True",
                    label_neg="False",
                    template=t,
                    meta={"a": a, "b": b, "c_pos": c_pos, "c_neg": c_neg},
                )
            )

    return pairs
