"""Syllogism validity concept dataset (psycholinguistics dataset for testing abstract reasoning).

Pos: valid Barbara syllogism — all A are B; all B are C; so all A are C → "yes"
Neg: invalid undistributed middle — all A are B; all C are B; so all A are C → "no"

The only surface difference is the second premise: "all B are C" vs "all C are B".
Nonce words are used so the model cannot rely on world knowledge; it must
process the logical structure of the premises.

The dataset is contains made-up nonsense words ("dax", "wug", "fep", etc.) used as stand-ins
for the logical variables A, B, C in the syllogism templates. This is to prevent the model
from using world knowledge to judge validity (e.g., if one used real entities like "cats", "mammals", "animals",
the model might answer "yes" because it knows cats are animals, not necessarily because it processed the logical
structure of the premises).
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

_NONCE_TRIPLES: list[tuple[str, str, str]] = [
    ("dax", "wug", "fep"),
    ("blik", "zorn", "mip"),
    ("tov", "rath", "pim"),
    ("glurp", "snorf", "vem"),
    ("drig", "plon", "quet"),
    ("zing", "morb", "kel"),
    ("frob", "nack", "dug"),
    ("stip", "bant", "yux"),
    ("flep", "gors", "tunk"),
    ("criv", "dawt", "sulf"),
    ("morf", "jisp", "brek"),
    ("hund", "volp", "rast"),
    ("quiv", "stel", "borm"),
    ("drox", "fimp", "yeld"),
    ("clug", "snev", "worp"),
]

TEMPLATES = {
    "T0": ("Yes or No: {a}⊂{b} and {b}⊂{c}, therefore {a}⊂{c}: ", "Yes or No: {a}⊂{b} and {c}⊂{b}, therefore {a}⊂{c}: "),
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
    n_per_template: int = 45,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Barbara (valid) vs undistributed middle (invalid) syllogism pairs."""
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    triples = list(_NONCE_TRIPLES)
    rng.shuffle(triples)

    pairs: list[ConceptPair] = []
    seen: set[tuple[str, str, str, str]] = set()
    counts = {t: 0 for t in templates}

    for a, b, c in triples * 4:
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
