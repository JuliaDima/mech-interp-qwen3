"""Decimal termination dataset restricted to smooth vs large-prime denominators.

A variant of decimal_termination_dataset where:
  Pos: same smooth denominators (1/n terminates — n = 2^a * 5^b)
  Neg: denominators whose every prime factor is > 20 (primes, or products of
       large primes).  No multiples of 3, 7, 11, 13 appear in the neg set.

This isolates a specific regime: can the model distinguish smooth numbers from
numbers that have no small non-2/5 prime factors?  The concept is still
"1/n terminates" but probed only where the non-terminating examples involve
large, opaque prime structure rather than easily recognisable small factors.

Compare results against decimal_termination_dataset to see whether the model
uses different mechanisms for large-prime vs small-prime denominators.
"""

from __future__ import annotations

import random

from sympy import factorint

from experiments.concept_localization.concept_pair import ConceptPair


def _terminates(n: int) -> bool:
    return all(p in (2, 5) for p in factorint(n))


_POS: list[int] = [n for n in range(100, 1000) if _terminates(n)]
_NEG: list[int] = [
    n for n in range(100, 1000) if not _terminates(n) and all(p > 20 for p in factorint(n))
]

TEMPLATES = {
    "T0": ("Yes or No: 1/{n} terminates: ", "Yes or No: 1/{n} terminates: "),
    "T1": (
        "Yes or No: 1 divided by {n} has a finite decimal: ",
        "Yes or No: 1 divided by {n} has a finite decimal: ",
    ),
    "T2": (
        "Yes or No: does 1/{n} have a terminating decimal? ",
        "Yes or No: does 1/{n} have a terminating decimal? ",
    ),
}


def make_anchor_positions(template_str: str, n: int, tokenizer) -> dict[str, int]:
    """Compute token positions for each digit of n (digit_1 = ones, digit_2 = tens, …)."""
    n_str = str(n)
    pre_n = template_str[: template_str.index("{n}")]
    positions: dict[str, int] = {}
    for i in range(len(n_str)):
        prefix = pre_n + n_str[: i + 1]
        pos = len(tokenizer(prefix, add_special_tokens=False).input_ids) - 1
        positions[f"digit_{len(n_str) - i}"] = pos
    return positions


def _anchor_factory(pair, tokenizer) -> dict[str, int]:
    tmpl_str = TEMPLATES[pair.template][0]
    return make_anchor_positions(tmpl_str, pair.meta["n_pos"], tokenizer)


ANCHOR_FACTORY = _anchor_factory
ANCHOR_MODES = ("digit_1", "digit_2", "digit_3")


def generate_large_prime_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs smooth n_pos vs large-prime n_neg, matched by magnitude."""
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
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(n=n_pos),
                    prompt_neg=fmt_neg.format(n=n_neg),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos="Yes",
                    predict_neg="No",
                    template=t,
                    meta={"n_pos": n_pos, "n_neg": n_neg},
                )
            )
            counts[t] += 1

    return pairs
