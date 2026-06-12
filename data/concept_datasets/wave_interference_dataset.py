"""Wave interference concept dataset.

Modular structure: constructive iff path difference d = k*λ for integer k.

Pos: d is an integer multiple of λ → constructive interference → "yes"
Neg: d is a half-integer multiple of λ → destructive interference → "no"

λ varies across {2, 4, 6}; k varies per pair; d values are 1-2 digit integers.
Pairs with different digit counts for d_pos vs d_neg are skipped by the
anchor detection in extract_deltas_generic (different tokenization length).
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

WAVELENGTHS = [10, 12, 14, 15, 16, 18, 20, 21, 24, 25]

TEMPLATES: dict[str, tuple[str, str, str]] = {
    "T0": ("Waves lambda={lam} path={d} interfere constructively? Answer yes or no:", "yes", "no"),
    "T1": ("Is path difference {d} a multiple of wavelength {lam}? Answer yes or no:", "yes", "no"),
    "T2": ("Wavelength {lam} and path diff {d} is constructive? Answer yes or no:", "yes", "no"),
}


def generate_wave_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Constructive (d=k*lam) vs destructive (d=(k+0.5)*lam) interference pairs."""
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int, int]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 200 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        lam = rng.choice(WAVELENGTHS)
        k_pos = rng.randint(1, 9)
        d_pos = k_pos * lam

        k_neg = rng.randint(0, 8)
        d_neg = k_neg * lam + lam // 2

        if d_neg <= 0 or d_pos == d_neg:
            continue
        # both must be exactly 2 digits so tokenisation is identical
        if len(str(d_pos)) != 2 or len(str(d_neg)) != 2:
            continue

        key = (lam, d_pos, d_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt, predict_pos, predict_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt.format(lam=lam, d=d_pos),
                    prompt_neg=fmt.format(lam=lam, d=d_neg),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos=predict_pos,
                    predict_neg=predict_neg,
                    template=t,
                    meta={"lam": lam, "d_pos": d_pos, "d_neg": d_neg, "k_pos": k_pos},
                )
            )
            counts[t] += 1

    return pairs
