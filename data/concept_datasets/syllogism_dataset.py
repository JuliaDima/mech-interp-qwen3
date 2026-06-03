"""Syllogism validity concept dataset.

Pos: valid Barbara syllogism — all A are B; all B are C; so all A are C → "yes"
Neg: invalid undistributed middle — all A are B; all C are B; so all A are C → "no"

The only surface difference is the second premise: "all B are C" vs "all C are B".

Single uppercase letters are used as logical variables so that each variable
tokenises as exactly one token under Qwen's BPE, keeping anchor positions stable
across all pairs.  Letters carry no world-knowledge associations that could bias
the model's logical judgement.
"""

from __future__ import annotations

import itertools
import random

from experiments.concept_localization.concept_pair import ConceptPair

# 120 distinct ordered triples from {B,C,D,F,G,H,J,K,L,M,N,P}.
# Fixed shuffle seed so the list is stable across runs.
_POOL = list("BCDFGHJKLMNP")
_all_triples = list(itertools.permutations(_POOL, 3))
random.Random(0).shuffle(_all_triples)
_LETTER_TRIPLES: list[tuple[str, str, str]] = _all_triples[:120]

TEMPLATES = {
    "T0": (
        "Yes or No: {a} is in {b} and {b} is in {c}, therefore {a} is in {c}: ",
        "Yes or No: {a} is in {b} and {c} is in {b}, therefore {a} is in {c}: ",
    ),
    "T1": (
        "Yes or No: every {a} is a {b}; every {b} is a {c}; so every {a} is a {c}: ",
        "Yes or No: every {a} is a {b}; every {c} is a {b}; so every {a} is a {c}: ",
    ),
    "T2": (
        "Yes or No: all {a} are {b}, all {b} are {c}, so all {a} are {c}: ",
        "Yes or No: all {a} are {b}, all {c} are {b}, so all {a} are {c}: ",
    ),
}


def generate_syllogism_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Barbara (valid) vs undistributed middle (invalid) syllogism pairs."""
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    triples = list(_LETTER_TRIPLES)
    rng.shuffle(triples)

    pairs: list[ConceptPair] = []
    seen: set[tuple[str, str, str, str]] = set()
    counts = {t: 0 for t in templates}

    for a, b, c in triples * 2:
        for t in templates:
            if counts[t] >= n_per_template:
                continue
            key = (a, b, c, t)
            if key in seen:
                continue
            seen.add(key)
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(a=a, b=b, c=c),
                    prompt_neg=fmt_neg.format(a=a, b=b, c=c),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos="Yes",
                    predict_neg="No",
                    template=t,
                    meta={"a": a, "b": b, "c": c},
                )
            )
            counts[t] += 1

    return pairs
