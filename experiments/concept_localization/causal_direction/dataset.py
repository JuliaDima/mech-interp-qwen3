"""Causal direction concept dataset.

Pairs: '{A} causes {B}: '  (correct direction)
  vs   '{B} causes {A}: '  (reversed — wrong)

Both A-token and B-token change (two positions differ); anchor is the last one.
Three templates vary the causal verb while keeping the same word pairs.
All pairs verified to give equal tokenization length when reversed in Qwen3.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

# (cause, effect) — reversed direction is clearly wrong.
# Verified: equal token count in all three template forms.
_CAUSAL_FACTS: list[tuple[str, str]] = [
    ("sun", "heat"),
    ("cold", "ice"),
    ("pressure", "pain"),
    ("impact", "damage"),
    ("smoke", "pollution"),
    ("hunger", "weakness"),
    ("heat", "evaporation"),
    ("drought", "famine"),
    ("noise", "stress"),
    ("fire", "ash"),
    ("salt", "rust"),
    ("acid", "rust"),
    ("force", "motion"),
    ("rain", "growth"),
    ("sun", "light"),
    ("cold", "death"),
    ("sun", "burn"),
    ("work", "stress"),
    ("rain", "rust"),
    ("heat", "rust"),
    ("cold", "snow"),
    ("fire", "heat"),
]

TEMPLATES = {
    "T0": ("{A} causes {B}: ", "{B} causes {A}: "),
    "T1": ("{A} leads to {B}: ", "{B} leads to {A}: "),
    "T2": ("{A} produces {B}: ", "{B} produces {A}: "),
}


def generate_causal_pairs(
    n_per_template: int = 22,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Generate pairs; capped at len(_CAUSAL_FACTS) unique pairs per template."""
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    facts = list(_CAUSAL_FACTS)
    rng.shuffle(facts)

    pairs: list[ConceptPair] = []
    seen: set[tuple[str, str, str]] = set()

    for cause, effect in facts * 3:  # allow cycling if n_per_template < len(facts)
        for t in templates:
            if sum(1 for p in pairs if p.template == t) >= n_per_template:
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
                    label_pos="Yes",
                    label_neg="No",
                    template=t,
                    meta={"cause": cause, "effect": effect},
                )
            )

        if all(sum(1 for p in pairs if p.template == t) >= n_per_template for t in templates):
            break

    return pairs
