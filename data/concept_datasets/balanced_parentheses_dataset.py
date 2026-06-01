"""Balanced parentheses concept dataset.

Parentheses are space-separated to ensure each '(' and ')' tokenizes individually.
Without spaces, consecutive parentheses compress into multi-character tokens.

Modular structure: depth counter increments on '(' and decrements on ')'.
Valid iff depth never goes negative and returns to 0.

Pos: balanced sequence with spaces, length 12 parens (6 pairs) → 23 token chars.
Neg: same sequence with one ')' in the last 3 paren positions replaced by '(' —
same length, guaranteed unbalanced, divergence always in the final paren window.
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
    "T2": (
        "Yes or No: is {seq} a valid bracket string? ",
        "Yes or No: is {seq} a valid bracket string? ",
    ),
}


def _generate_balanced(n: int) -> list[str]:
    """All balanced parentheses strings of length 2*n, space-separated."""
    results: list[str] = []

    def _gen(s: str, opens: int, closes: int) -> None:
        if len(s) == 2 * n:
            spaced = " ".join(s)
            results.append(spaced)
            return
        if opens < n:
            _gen(s + "(", opens + 1, closes)
        if closes < opens:
            _gen(s + ")", opens, closes + 1)

    _gen("", 0, 0)
    return results

# C(6) = 132 balanced sequences of 6 pairs (12 parens)
_BALANCED: list[str] = _generate_balanced(6)


def _make_unbalanced(s: str, rng: random.Random) -> str | None:
    """Replace a ')' in the last 3 paren positions with '(' to make s unbalanced.

    The flip is constrained to the final paren window so that pos and neg prompts
    diverge only in a known, anchorable region of the sequence.
    """
    paren_str_positions = [i for i, c in enumerate(s) if c in "()"]
    last_3 = paren_str_positions[-3:]
    close_positions = [i for i in last_3 if s[i] == ")"]
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
