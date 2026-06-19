"""Causal direction concept dataset.

Pos: correct causal direction (A causes B).
Neg: reversed direction (B causes A) — physically implausible.

All entity pairs are single-token words to guarantee equal tokenization length.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

_CAUSAL_FACTS: list[tuple[str, str]] = [
    # Physical / environmental — reverse direction is clearly implausible
    ("sun", "heat"),            # heat doesn't cause the sun
    ("impact", "damage"),       # damage doesn't cause impacts
    ("heat", "evaporation"),    # evaporation cools, doesn't produce heat
    ("drought", "famine"),      # famine doesn't cause drought
    ("fire", "ash"),            # ash doesn't cause fire
    ("salt", "rust"),           # rust doesn't produce salt
    ("acid", "rust"),           # rust doesn't produce acid
    ("rain", "growth"),         # plant growth doesn't cause rain (on relevant scale)
    ("cold", "death"),          # death doesn't cause cold temperatures
    ("sun", "burn"),            # sunburn; burn doesn't produce the sun
    ("work", "stress"),         # stress doesn't drive people to work physically
    ("rain", "rust"),           # rust doesn't produce rain
    ("heat", "rust"),           # rust doesn't produce heat
    ("wind", "erosion"),        # erosion doesn't cause wind
    ("light", "warmth"),        # warmth doesn't produce light on its own
    ("frost", "death"),         # death doesn't cause frost
    ("flood", "damage"),        # damage doesn't cause floods
    ("storm", "flood"),         # flood doesn't cause storms
    ("gravity", "fall"),        # falling doesn't generate gravity
    ("wind", "waves"),          # waves don't cause wind
    ("rain", "mud"),            # mud doesn't cause rain
    ("dark", "fear"),           # fear doesn't create darkness
    ("sun", "drought"),         # drought doesn't produce the sun
    ("mud", "slip"),            # slipping doesn't create mud
    ("dust", "cough"),          # coughing doesn't produce dust
    ("smoke", "cancer"),        # cancer doesn't produce smoke
    ("salt", "thirst"),         # thirst doesn't produce salt
    ("sugar", "energy"),        # energy doesn't produce sugar
    ("heat", "sweat"),          # sweat doesn't generate heat
    ("rain", "flood"),          # flood doesn't cause rain
    ("wind", "dust"),           # dust doesn't cause wind
    ("ice", "slip"),            # slipping doesn't create ice
    ("fire", "smoke"),          # smoke doesn't cause fire
    ("storm", "damage"),        # damage doesn't cause storms
    ("flood", "loss"),          # loss doesn't cause floods
    ("lightning", "fire"),      # fire doesn't cause lightning
    ("poison", "death"),        # death doesn't produce poison
    ("heat", "burn"),           # burns don't generate heat
    ("cut", "scar"),            # scars don't cause cuts
    # Biological / physiological
    ("smoke", "pollution"),     # pollution doesn't produce smoke
    ("noise", "stress"),        # stress doesn't generate loud noise
    ("cold", "shivers"),        # shivering doesn't cause cold
    ("fall", "bruise"),         # bruises don't cause falls
    ("run", "sweat"),           # sweat doesn't cause running
    ("cut", "bleeding"),        # bleeding doesn't cause cuts
    ("burn", "scar"),           # scars don't cause burns
    ("debt", "stress"),         # stress doesn't produce debt
    ("weight", "strain"),       # strain doesn't create weight
    # Psychological / social
    ("fear", "flight"),         # flight (fleeing) doesn't cause fear
    ("loss", "grief"),          # grief doesn't cause loss
    ("effort", "result"),       # results don't generate effort
    ("rain", "mold"),           # mold doesn't cause rain
    ("light", "vision"),        # vision doesn't produce light
    ("sound", "hearing"),       # hearing doesn't produce sound
    ("smell", "memory"),        # memory doesn't produce smells
    ("touch", "feeling"),       # feelings don't cause touch
    ("tension", "headache"),    # headaches don't cause tension
    ("cruelty", "fear"),        # fear doesn't cause cruelty (weaker reverse)
    ("neglect", "decline"),     # decline doesn't cause neglect
    ("gain", "joy"),            # joy doesn't produce gain
    ("loss", "sadness"),        # sadness doesn't produce loss
]

TEMPLATES: dict[str, tuple[str, str, str]] = {
    "T0": ("Does {A} lead to {B}? Answer yes or no: ", "yes", "no"),
    "T1": ("Does {A} cause {B}? Answer yes or no: ", "yes", "no"),
    "T2": ("Does {A} produce {B}? Answer yes or no: ", "yes", "no"),
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
            fmt, predict_pos, predict_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt.format(A=cause, B=effect),
                    prompt_neg=fmt.format(A=effect, B=cause),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos=predict_pos,
                    predict_neg=predict_neg,
                    template=t,
                    meta={"cause": cause, "effect": effect},
                )
            )
            counts[t] += 1

    return pairs
