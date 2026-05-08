"""Residue class concept dataset.

Pairs: a_pos ≡ 1 (mod 7)  vs  a_neg ≡ 6 (mod 7)
Both share the same prefix; only the units digit differs.

Three templates vary the surface form of the modular arithmetic expression.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

_TARGET_POS = 1
_TARGET_NEG = 6

TEMPLATES = {
    "T0": ("{a}%7= ", "{a}%7= "),
    "T1": ("{a} % 7 = ", "{a} % 7 = "),
    "T2": ("calc: {a}%7= ", "calc: {a}%7= "),
}


def generate_residue_pairs(
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
        base_mult = rng.randint(15, 141)
        base = base_mult * 7
        a_pos = base + _TARGET_POS
        a_neg = base + _TARGET_NEG
        if not (100 <= a_pos <= 999 and 100 <= a_neg <= 999):
            continue
        if a_pos % 7 != _TARGET_POS or a_neg % 7 != _TARGET_NEG:
            continue
        if base in seen:
            continue
        seen.add(base)

        for t in templates:
            if sum(1 for p in pairs if p.template == t) >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(a=a_pos),
                    prompt_neg=fmt_neg.format(a=a_neg),
                    label_pos=str(_TARGET_POS),
                    label_neg=str(_TARGET_NEG),
                    template=t,
                    meta={"a_pos": a_pos, "a_neg": a_neg},
                )
            )

    return pairs
