"""Doppler shift direction concept dataset.

Linear structure: observed frequency increases iff source approaches,
decreases iff source recedes. Single threshold at zero relative velocity.

Pos: source approaching → observed pitch higher than emitted → "yes"
Neg: source receding → observed pitch higher than emitted → "no"

All template pairs use motion words verified to tokenize to the same length
under the Qwen3 tokenizer, so anchor detection succeeds for every pair.
The anchor position is chosen empirically by the positional attribution sweep
rather than hard-coded here.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

_SOURCES = [
    "train",
    "bus",
    "ship",
    "car",
    "truck",
    "jet",
    "drone",
    "horn",
    "siren",
    "locomotive",
    "helicopter",
    "motorcycle",
    "speedboat",
    "ambulance",
    "aircraft",
    "speaker",
    "whistle",
    "bell",
]

TEMPLATES = {
    "T0": (
        "{src} nears, observed frequency exceeds emitted: ",
        "{src} departs, observed frequency exceeds emitted: ",
    ),
    "T1": (
        "{src} is incoming, measured pitch above emitted pitch: ",
        "{src} is outgoing, measured pitch above emitted pitch: ",
    ),
    "T2": (
        "{src} inbound, pitch measured higher than emitted: ",
        "{src} outbound, pitch measured higher than emitted: ",
    ),
}


def generate_doppler_pairs(
    n_per_template: int = 60,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Approaching (pitch rises) vs receding (pitch falls) source pairs.

    All four templates use motion word pairs that tokenize to the same length
    under the Qwen3 tokenizer, so no pairs are skipped by anchor detection.
    """
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[str, str]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 200 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        src = rng.choice(_SOURCES)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            key = (src, t)
            if key in seen:
                continue
            seen.add(key)
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(src=src),
                    prompt_neg=fmt_neg.format(src=src),
                    label_pos="yes",
                    label_neg="no",
                    template=t,
                    meta={"src": src},
                )
            )
            counts[t] += 1

    return pairs
