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

TEMPLATES: dict[str, tuple[str, str, str]] = {
    "T0": ("All {a} are {b}, {mid}, so all {a} are {c}? Answer yes or no: ", "yes", "no"),
    "T1": ("Every {a} is a {b}; {mid}; so every {a} is a {c}? Answer yes or no: ", "yes", "no"),
    "T2": ("{a} is in {b} and {mid}, therefore {a} is in {c}? Answer yes or no: ", "yes", "no"),
}

# Valid (pos) and invalid (neg) middle premises per template
MID_POS: dict[str, str] = {
    "T0": "all {b} are {c}",
    "T1": "every {b} is a {c}",
    "T2": "{b} is in {c}",
}
MID_NEG: dict[str, str] = {
    "T0": "all {c} are {b}",
    "T1": "every {c} is a {b}",
    "T2": "{c} is in {b}",
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
            fmt, predict_pos, predict_neg = TEMPLATES[t]
            mid_pos = MID_POS[t].format(b=b, c=c)
            mid_neg = MID_NEG[t].format(b=b, c=c)
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt.format(a=a, b=b, c=c, mid=mid_pos),
                    prompt_neg=fmt.format(a=a, b=b, c=c, mid=mid_neg),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos=predict_pos,
                    predict_neg=predict_neg,
                    template=t,
                    meta={"a": a, "b": b, "c": c},
                )
            )
            counts[t] += 1

    return pairs
