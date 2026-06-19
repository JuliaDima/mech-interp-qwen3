"""Residue class concept dataset.

Fixed modulus m=7, fixed positive residue r_pos=1.  The negative residue
r_neg is sampled uniformly from {2, 3, 4, 5, 6} per pair so that the numeric
offset (a_neg - a_pos = r_neg - 1 ∈ {1,...,5}) varies across pairs.  This
decorrelates residue-class membership from a fixed arithmetic magnitude
difference, making the mean delta a genuine "residue class 1 mod 7" direction
rather than a proxy for a constant numeric offset.

a varies freely over 2-4 digit numbers for statistical diversity.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

M = 7
R_POS = 1
_R_NEG_CHOICES = [2, 3, 4, 5, 6]  # sampled per pair to vary the offset

TEMPLATES: dict[str, tuple[str, str, str | None]] = {
    "T0": ("calc: {a}%7= ", str(R_POS), None),       # predict_pos fixed; neg varies per pair
    "T1": ("What is the remainder of {a} divided by 7? Answer: ", str(R_POS), None),
    "T2": ("Is {a} mod 7 equal to 1? Answer yes or no: ", "yes", "no"),  # pos: r_pos=1→yes, neg: r_neg≠1→no

}


def generate_residue_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs a_pos ≡ 1 (mod 7) vs a_neg ≡ r_neg (mod 7), r_neg ∈ {2,...,6}.

    a_pos = 7k + 1.  r_neg is drawn uniformly from {2,3,4,5,6} per pair so
    the offset a_neg - a_pos = r_neg - 1 ∈ {1,...,5} varies, decorrelating
    the magnitude difference from the residue-class signal.
    """
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 500 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        n_digits = 3  # fixed digit length to control for magnitude cues
        lo, hi = 10 ** (n_digits - 1), 10**n_digits - 1

        r_neg = rng.choice(_R_NEG_CHOICES)

        k_min = (lo - R_POS + M - 1) // M
        k_max = (hi - r_neg) // M
        if k_min > k_max:
            continue

        # Sample k_pos and k_neg independently so that a_pos and a_neg differ
        # by O(M * |k_pos - k_neg|) rather than only r_neg - R_POS ∈ {1,...,5}.
        # This makes the real pair delta comparable in scale to the within-class
        # null delta, which also varies a by O(M * k) across examples.
        k_pos = rng.randint(k_min, k_max)
        k_neg = rng.randint(k_min, k_max)
        while k_neg == k_pos:
            k_neg = rng.randint(k_min, k_max)
        a_pos = M * k_pos + R_POS
        a_neg = M * k_neg + r_neg

        if not (lo <= a_pos <= hi and lo <= a_neg <= hi):
            continue
        if (k_pos, r_neg) in seen:
            continue
        seen.add((k_pos, r_neg))

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt, predict_pos_tmpl, predict_neg_tmpl = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt.format(a=a_pos),
                    prompt_neg=fmt.format(a=a_neg),
                    label_pos=str(R_POS),
                    label_neg=str(r_neg),
                    predict_pos=predict_pos_tmpl,
                    predict_neg=predict_neg_tmpl if predict_neg_tmpl is not None else str(r_neg),
                    template=t,
                    meta={"a_pos": a_pos, "a_neg": a_neg, "m": M, "r_pos": R_POS, "r_neg": r_neg,
                          "offset": r_neg - R_POS},
                )
            )
            counts[t] += 1

    return pairs
