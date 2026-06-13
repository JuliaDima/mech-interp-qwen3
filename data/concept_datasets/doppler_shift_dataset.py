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
    "train", "bus", "ship", "car", "truck", "jet", "drone", "horn",
    "siren", "locomotive", "helicopter", "motorcycle", "speedboat",
    "ambulance", "aircraft", "speaker", "whistle", "bell",
    "rocket", "alarm", "engine", "propeller", "foghorn", "cannon",
    "thunder", "chainsaw", "jackhammer", "lawnmower", "blender",
    "typewriter", "piano", "trumpet", "tuba", "saxophone", "flute",
    "cymbal", "drum", "tractor", "scooter", "skateboard", "bicycle",
    "submarine", "torpedo", "warplane", "bomber", "glider", "blimp",
    "hovercraft", "snowmobile", "ATV", "tank", "catapult", "launcher",
    "firework", "comet", "meteor", "shuttle", "satellite", "probe",
    "ferry", "tugboat", "canoe", "kayak", "yacht", "tanker",
    "bulldozer", "excavator", "crane", "forklift", "roller", "grader",
    "sander", "router", "drill", "lathe", "compressor", "generator",
    "turbine", "reactor", "motor", "piston", "fan", "pump",
    "whistle", "foghorn", "buzzer", "klaxon", "gong", "chime",
    "megaphone", "radio", "sonar", "radar", "beacon", "siren",
    "zeppelin", "helo", "gunship", "interceptor", "racer", "dragster",
    "streetcar", "monorail", "tram", "trolley", "metro", "subway",
]

TEMPLATES: dict[str, tuple[str, str, str]] = {
    "T0": ("Given {src} {direction}, observed frequency exceeds emitted? Answer yes or no: ", "yes", "no"),
    "T1": ("Given {src} {direction}, measured pitch above emitted pitch? Answer yes or no: ", "yes", "no"),
    "T2": ("Given {src} {direction}, pitch measured higher than emitted? Answer yes or no: ", "yes", "no"),
    }

# Approaching vs receding direction words per template
DIRECTION_PAIRS: dict[str, tuple[str, str]] = {
    "T0": ("oncoming", "receding"),
    "T1": ("incoming", "outgoing"),
    "T2": ("nearing", "leaving"),
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
            fmt, predict_pos, predict_neg = TEMPLATES[t]
            dir_pos, dir_neg = DIRECTION_PAIRS[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt.format(src=src, direction=dir_pos),
                    prompt_neg=fmt.format(src=src, direction=dir_neg),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos=predict_pos,
                    predict_neg=predict_neg,
                    template=t,
                    meta={"src": src},
                )
            )
            counts[t] += 1

    return pairs
