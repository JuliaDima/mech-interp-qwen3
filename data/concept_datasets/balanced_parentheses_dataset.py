"""Balanced parentheses concept dataset.

Modular structure: depth counter increments on '(' and decrements on ')'.
Valid iff depth never goes negative and returns to 0.

Pos: balanced sequence of length 4, 6, or 8.
Neg: same sequence with one ')' replaced by '(' — same length, guaranteed unbalanced.

Because only one character changes and '(' and ')' are single tokens,
pos and neg tokenize to the same length; anchor = the substituted position.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES = {
    "T0": ("Yes or No: balanced: {seq}: ", "Yes or No: balanced: {seq}: "),
    "T1": (
        "Yes or No: the sequence {seq} has matched brackets: ",
        "Yes or No: the sequence {seq} has matched brackets: ",
    ),
    "T2": ("Yes or No: is {seq} a valid bracket string? ", "Yes or No: is {seq} a valid bracket string? "),
}


def _generate_balanced(n: int) -> list[str]:
    """All balanced parentheses strings of length 2*n."""
    results: list[str] = []

    def _gen(s: str, opens: int, closes: int) -> None:
        if len(s) == 2 * n:
            results.append(s)
            return
        if opens < n:
            _gen(s + "(", opens + 1, closes)
        if closes < opens:
            _gen(s + ")", opens, closes + 1)

    _gen("", 0, 0)
    return results


_BALANCED: list[str] = _generate_balanced(2) + _generate_balanced(3) + _generate_balanced(4)


def _make_unbalanced(s: str, rng: random.Random) -> str | None:
    """Replace a random ')' with '(' — same length, guaranteed unbalanced."""
    close_positions = [i for i, c in enumerate(s) if c == ")"]
    if not close_positions:
        return None
    pos = rng.choice(close_positions)
    return s[:pos] + "(" + s[pos + 1 :]


def generate_parentheses_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[str, int]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 500 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        seq_pos = rng.choice(_BALANCED)
        seq_neg = _make_unbalanced(seq_pos, rng)
        if seq_neg is None or seq_neg == seq_pos:
            continue

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            key = (seq_pos, hash(t))
            if key in seen:
                continue
            seen.add(key)
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(seq=seq_pos),
                    prompt_neg=fmt_neg.format(seq=seq_neg),
                    label_pos="yes",
                    label_neg="no",
                    predict_pos="Yes",
                    predict_neg="No",
                    template=t,
                    meta={"seq_pos": seq_pos, "seq_neg": seq_neg},
                )
            )
            counts[t] += 1

    return pairs
