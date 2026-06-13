"""Decimal expansion termination concept dataset.

Modular structure: 1/n terminates iff n = 2^a * 5^b (only factors 2 and 5).

Pos: n_pos has only factors 2 and 5 → terminating decimal → "yes"
Neg: n_neg has another prime factor → repeating decimal → "no"

All denominators are 3-digit (100–999).  _POS is the complete set of 14
terminating denominators.  _NEG is sampled to cover each small prime factor
class (3, 7, 11, 13, 17, 19, …) so the dataset tests the full mathematical
rule, not just smooth vs large-prime denominators.

See also: decimal_termination_large_prime_dataset.py, which isolates the
smooth vs large-prime contrast specifically.
"""

from __future__ import annotations

import random
from collections import defaultdict

from sympy import factorint

from experiments.concept_localization.concept_pair import ConceptPair


def _terminates(n: int) -> bool:
    return all(p in (2, 5) for p in factorint(n))


def _smallest_non25(n: int) -> int | None:
    return min((p for p in factorint(n) if p not in (2, 5)), default=None)


def _build_neg(seed: int = 42) -> list[int]:
    """Sample NEG with ~12 values per small-prime factor class.

    by_factor maps each prime p → list of 3-digit n values whose smallest
    non-{2,5} factor is p.  p is a grouping label;
    all collected n values are in range(100, 1000).
    """
    rng = random.Random(seed)
    # Keys are prime factors; values are 3-digit n that belong to that class.
    by_factor: dict[int, list[int]] = defaultdict(list)
    for n in range(100, 1000):
        if not _terminates(n):
            p = _smallest_non25(n)  # e.g. 3 for n=300, 7 for n=700
            if p is not None:
                by_factor[p].append(n)  # n is 3-digit; p is the bucket key
    result: set[int] = set()
    # Sample ~12 three-digit n values per common small-prime class.
    for p in [3, 7, 11, 13, 17, 19]:
        result.update(rng.sample(by_factor[p], min(12, len(by_factor[p]))))
    # Add a few more from the next 9 prime-factor classes for coverage.
    for p in sorted(by_factor)[6:15]:
        result.update(rng.sample(by_factor[p], min(5, len(by_factor[p]))))
    return sorted(result)


_POS: list[int] = [n for n in range(100, 1000) if _terminates(n)]
_NEG: list[int] = _build_neg()

TEMPLATES: dict[str, tuple[str, str, str]] = {
    "T0": ("Does 1/{n} have a terminating decimal? Answer yes or no: ", "yes", "no"),
    "T1": ("1 divided by {n} has a finite decimal? Answer yes or no: ", "yes", "no"),
    "T2": ("Does 1/{n} terminate? Answer yes or no: ", "yes", "no"),
}


def make_anchor_positions(template_str: str, n: int, tokenizer) -> dict[str, int]:
    """Compute token positions for each digit of n.

    Keys are digit_1 (ones), digit_2 (tens), digit_3 (hundreds), numbered from
    the right.  Only keys up to len(str(n)) are present, so digit_2 is absent
    for single-digit n.
    """
    n_str = str(n)
    pre_n = template_str[: template_str.index("{n}")]
    n_digits = len(n_str)
    positions: dict[str, int] = {}
    for i in range(n_digits):
        prefix = pre_n + n_str[: i + 1]
        pos = len(tokenizer(prefix, add_special_tokens=False).input_ids) - 1
        digit_from_right = n_digits - i  # n_digits → ... → 2 → 1
        positions[f"digit_{digit_from_right}"] = pos
    return positions



def generate_decimal_pairs(
    n_per_template: int = 80,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs n_pos (terminates) vs n_neg (repeats), matched by magnitude."""
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 200 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        n_pos = rng.choice(_POS)
        close_negs = sorted(_NEG, key=lambda x: abs(x - n_pos))[:8]
        n_neg = rng.choice(close_negs)

        if len(str(n_pos)) != len(str(n_neg)):
            continue

        key = (n_pos, n_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt, predict_pos, predict_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt.format(n=n_pos),
                    prompt_neg=fmt.format(n=n_neg),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos=predict_pos,
                    predict_neg=predict_neg,
                    template=t,
                    meta={"n_pos": n_pos, "n_neg": n_neg},
                )
            )
            counts[t] += 1

    return pairs
