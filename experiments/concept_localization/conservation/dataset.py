"""Energy conservation concept dataset.

Pairs: bounce height b_pos < drop height h (physically OK)
  vs   bounce height b_neg > h (violates conservation).
Only b changes; h is fixed.

Three templates vary how the scenario is described.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES = {
    "T0": ("drop={h}, bounce={b}. OK? ", "drop={h}, bounce={b}. OK? "),
    "T1": ("height={h}, rebound={b}. Valid? ", "height={h}, rebound={b}. Valid? "),
    "T2": ("fall={h}, rise={b}. Physical? ", "fall={h}, rise={b}. Physical? "),
}


def generate_conservation_pairs(
    n_per_template: int = 50,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int, int]] = set()

    attempts = 0
    while (
        any(sum(1 for p in pairs if p.template == t) < n_per_template for t in templates)
        and attempts < n_per_template * 200
    ):
        attempts += 1
        h = rng.randint(3, 8)
        b_pos = rng.randint(1, h - 1)
        b_neg = rng.randint(h + 1, 9)
        key = (h, b_pos, b_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if sum(1 for p in pairs if p.template == t) >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(h=h, b=b_pos),
                    prompt_neg=fmt_neg.format(h=h, b=b_neg),
                    label_pos="Yes",
                    label_neg="No",
                    template=t,
                    meta={"h": h, "b_pos": b_pos, "b_neg": b_neg},
                )
            )

    return pairs
