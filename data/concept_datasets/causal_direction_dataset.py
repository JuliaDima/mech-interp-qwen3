"""Causal direction concept dataset.

Pos: correct causal direction (A causes B).
Neg: reversed direction (B causes A) — physically implausible.

All entity pairs are single-token words to guarantee equal tokenization length.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

_CAUSAL_FACTS: list[tuple[str, str]] = [
    ("sun", "heat"), ("cold", "ice"), ("pressure", "pain"),
    ("impact", "damage"), ("smoke", "pollution"), ("hunger", "weakness"),
    ("heat", "evaporation"), ("drought", "famine"), ("noise", "stress"),
    ("fire", "ash"), ("salt", "rust"), ("acid", "rust"),
    ("force", "motion"), ("rain", "growth"), ("sun", "light"),
    ("cold", "death"), ("sun", "burn"), ("work", "stress"),
    ("rain", "rust"), ("heat", "rust"), ("cold", "snow"),
    ("fire", "heat"), ("wind", "erosion"), ("light", "warmth"),
    ("frost", "death"), ("flood", "damage"), ("spark", "fire"),
    ("debt", "stress"),
    ("fall", "injury"), ("shock", "pain"), ("age", "decay"),
    ("war", "death"), ("storm", "flood"), ("speed", "heat"),
    ("weight", "strain"), ("gravity", "fall"), ("cold", "frost"),
    ("heat", "fire"), ("wind", "waves"), ("rain", "mud"),
    ("light", "growth"), ("dark", "fear"), ("loud", "damage"),
    ("stress", "illness"), ("sleep", "rest"), ("food", "energy"),
    ("sun", "drought"), ("ice", "cold"), ("mud", "slip"),
    ("dust", "cough"), ("smoke", "cancer"), ("fat", "weight"),
    ("salt", "thirst"), ("sugar", "energy"), ("water", "life"),
    ("fear", "flight"), ("joy", "laughter"), ("pain", "crying"),
    ("cold", "flu"), ("heat", "sweat"), ("rain", "flood"),
    ("wind", "dust"), ("ice", "slip"), ("fire", "smoke"),
    ("shock", "fear"), ("loss", "grief"), ("gain", "joy"),
    ("effort", "result"), ("age", "wisdom"), ("youth", "energy"),
    ("sun", "warmth"), ("cold", "shivers"), ("rain", "mold"),
    ("heat", "drought"), ("storm", "damage"), ("flood", "loss"),
    ("war", "poverty"), ("peace", "growth"), ("trust", "bond"),
    ("doubt", "fear"), ("hope", "action"), ("love", "care"),
    ("hate", "conflict"), ("pride", "confidence"), ("shame", "silence"),
    ("light", "vision"), ("dark", "sleep"), ("sound", "hearing"),
    ("smell", "memory"), ("touch", "feeling"), ("taste", "pleasure"),
    ("fall", "bruise"), ("run", "sweat"), ("lift", "strain"),
    ("cut", "bleeding"), ("burn", "scar"), ("freeze", "numbness"),
    ("tension", "headache"), ("calm", "sleep"), ("anger", "aggression"),
    ("kindness", "trust"), ("cruelty", "fear"), ("neglect", "decline"),
]

TEMPLATES = {
    "T0": ("True or False: {A} causes {B}: ", "True or False: {B} causes {A}: "),
    "T1": ("True or False: {A} leads to {B}: ", "True or False: {B} leads to {A}: "),
    "T2": ("True or False: {A} produces {B}: ", "True or False: {B} produces {A}: "),
}




def generate_causal_pairs(
    n_per_template: int = 22,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    facts = list(_CAUSAL_FACTS)
    rng.shuffle(facts)

    pairs: list[ConceptPair] = []
    seen: set[tuple[str, str, str]] = set()
    counts = {t: 0 for t in templates}

    for cause, effect in facts * 4:
        for t in templates:
            if counts[t] >= n_per_template:
                continue
            key = (cause, effect, t)
            if key in seen:
                continue
            seen.add(key)
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(A=cause, B=effect),
                    prompt_neg=fmt_neg.format(A=cause, B=effect),
                    label_pos="True",
                    label_neg="False",
                    predict_pos="True",
                    predict_neg="False",
                    template=t,
                    meta={"cause": cause, "effect": effect},
                )
            )
            counts[t] += 1

    return pairs
