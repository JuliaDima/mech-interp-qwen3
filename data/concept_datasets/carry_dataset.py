"""Carry concept dataset.

Each pair isolates a single carry at the units column (column 0).  Within a
pair, exactly one digit of one operand differs between pos (carry) and neg
(no carry); which operand varies is randomised.  Fixing the carry to the units
position ensures the contrastive signal is always at the same token position
relative to the expression, giving a clean concept direction.

Isolation conditions:
  - column 0:   a_0 + b_0 >= 10  (carry out of units) [pos]
                a_0 + b_0 < 10   [neg]
  - column 1:   a_1 + b_1 <= 8   (absorbs carry-in, no propagation)
  - columns j > 1:  a_j + b_j < 10  (no carry elsewhere)

Number of digits is sampled uniformly from {3, 4, 5} per pair.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES = {
    "T0": ("calc: {a}+{b}= ", "calc: {a}+{b}= "),
    "T1": ("{a} plus {b} is: ", "{a} plus {b} is: "),
    "T2": ("what is {a}+{b}? ", "what is {a}+{b}? "),
}


def make_anchor_positions(template_str: str, a: int, b: int, tokenizer) -> dict[str, int]:
    """Compute token positions by substituting a and b into the template.

    Tokenises the prefix ending at the last character of each number, giving
    the token that covers the ones digit — works for any template format.
    """
    a_str, b_str = str(a), str(b)
    pre_a = template_str[: template_str.index("{a}")]
    pre_b = template_str[: template_str.index("{b}")].replace("{a}", a_str)
    ones_a = len(tokenizer(pre_a + a_str, add_special_tokens=False).input_ids) - 1
    ones_b = len(tokenizer(pre_b + b_str, add_special_tokens=False).input_ids) - 1
    return {"ones_a": ones_a, "ones_b": ones_b, "separator": ones_a + 1}


def _carry_anchor_factory(pair, tokenizer) -> dict[str, int]:
    tmpl_str = TEMPLATES[pair.template][0]
    return make_anchor_positions(tmpl_str, pair.meta["a_pos"], pair.meta["b_pos"], tokenizer)


ANCHOR_FACTORY = _carry_anchor_factory
ANCHOR_MODES = ("ones_b", "ones_a", "separator")


def _digits(n: int, n_digits: int) -> list[int]:
    """Return digits of n least-significant-first."""
    return [(n // 10**i) % 10 for i in range(n_digits)]


def _from_digits(digs: list[int]) -> int:
    return sum(d * 10**i for i, d in enumerate(digs))


def generate_carry_pairs(
    n_per_template: int = 200,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Generate contrastive carry pairs with carry always at the units column.

    Each pair differs in exactly one digit of one operand.  Half the pairs vary
    a's units digit; the other half vary b's.  Fixing carry_col=0 ensures the
    contrastive signal is always anchored at the same token position, giving a
    clean delta direction unconfounded by carry-column variation.
    """
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int, int, int]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 300 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1

        n_digits = 3
        lo = 10 ** (n_digits - 1)
        hi = 10**n_digits - 1

        carry_col = 0  # always units digit
        vary_a = rng.random() < 0.5  # which operand's units digit varies

        # Draw both operands digit-by-digit (index 0 = least significant)
        a_digs = [rng.randint(0, 9) for _ in range(n_digits)]
        b_digs = [rng.randint(0, 9) for _ in range(n_digits)]
        a_digs[n_digits - 1] = rng.randint(1, 9)  # no leading zero
        b_digs[n_digits - 1] = rng.randint(1, 9)

        # Check isolation conditions for all columns except carry_col
        valid = True
        for j in range(n_digits):
            if j == carry_col:
                continue
            col_sum = a_digs[j] + b_digs[j]
            if j == carry_col + 1:
                if col_sum > 8:  # must absorb the carry-in without propagating
                    valid = False
                    break
            else:
                if col_sum >= 10:  # no carry at this column
                    valid = False
                    break
        if not valid:
            continue

        # Determine carry and no-carry digit values for the varying operand
        fixed_dig = b_digs[carry_col] if vary_a else a_digs[carry_col]
        carry_opts = [d for d in range(10) if d + fixed_dig >= 10]
        no_carry_opts = [d for d in range(10) if d + fixed_dig < 10]
        if not carry_opts or not no_carry_opts:
            continue

        vary_carry = rng.choice(carry_opts)
        vary_no_carry = rng.choice(no_carry_opts)

        # Build pos/neg digit arrays
        a_pos_digs = a_digs[:]
        a_neg_digs = a_digs[:]
        b_pos_digs = b_digs[:]
        b_neg_digs = b_digs[:]
        if vary_a:
            a_pos_digs[carry_col] = vary_carry
            a_neg_digs[carry_col] = vary_no_carry
        else:
            b_pos_digs[carry_col] = vary_carry
            b_neg_digs[carry_col] = vary_no_carry

        a_pos = _from_digits(a_pos_digs)
        a_neg = _from_digits(a_neg_digs)
        b_pos = _from_digits(b_pos_digs)
        b_neg = _from_digits(b_neg_digs)

        if not (
            lo <= a_pos <= hi and lo <= b_pos <= hi and lo <= a_neg <= hi and lo <= b_neg <= hi
        ):
            continue

        key = (a_pos, b_pos, a_neg, b_neg)
        if key in seen:
            continue
        seen.add(key)

        for t in templates:
            if counts[t] >= n_per_template:
                continue
            fmt_pos, fmt_neg = TEMPLATES[t]
            pairs.append(
                ConceptPair(
                    prompt_pos=fmt_pos.format(a=a_pos, b=b_pos),
                    prompt_neg=fmt_neg.format(a=a_neg, b=b_neg),
                    label_pos="carry",
                    label_neg="no_carry",
                    predict_pos=str(a_pos + b_pos),
                    predict_neg=str(a_neg + b_neg),
                    template=t,
                    meta={
                        "a_pos": a_pos,
                        "b_pos": b_pos,
                        "a_neg": a_neg,
                        "b_neg": b_neg,
                        "carry_col": carry_col,
                        "n_digits": n_digits,
                        "vary_a": vary_a,
                    },
                )
            )
            counts[t] += 1

    return pairs
