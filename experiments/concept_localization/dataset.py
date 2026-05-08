"""Controlled carry-contrast dataset for concept localization.

Each pair is identical except for the units digit of B:
one variant causes a units-column carry, the other does not.
Double carries (carry cascading to tens) are excluded so the
only systematically varying computation is carry vs no-carry.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from experiments.addition.dataset_generation.generate_dataset_with_predictions import (
    TEMPLATES,
    TemplateID,
)


@dataclass
class CarryPair:
    a: int
    b_carry: int
    b_no_carry: int
    template: TemplateID
    prompt_carry: str
    prompt_no_carry: str


def _is_single_carry(a: int, b: int) -> bool:
    """True iff a+b has a carry in units that does NOT cascade to tens."""
    a_units = a % 10
    b_units = b % 10
    if a_units + b_units < 10:
        return False
    tens_a = (a // 10) % 10
    tens_b = (b // 10) % 10
    return tens_a + tens_b <= 8  # with carry-in of 1, still < 10


def generate_carry_pairs(
    n_per_template: int,
    templates: list[TemplateID],
    n_digits: int = 3,
    seed: int = 42,
) -> list[CarryPair]:
    """Generate paired prompts differing only in the units digit of B.

    For each sampled (a, b_prefix), one b has units that cause a carry,
    the other does not.  No double carries.  Returns up to n_per_template
    pairs per template.
    """
    rng = random.Random(seed)
    lo = 10 ** (n_digits - 1)
    hi = 10**n_digits - 1
    b_prefix_lo = 10 ** (n_digits - 2)
    b_prefix_hi = 10 ** (n_digits - 1) - 1

    counts: dict[str, int] = {str(t): 0 for t in templates}
    pairs: list[CarryPair] = []
    seen: set[tuple[int, int, str]] = set()
    attempts = 0
    max_attempts = n_per_template * len(templates) * 50

    while attempts < max_attempts and any(v < n_per_template for v in counts.values()):
        attempts += 1
        a = rng.randint(lo, hi)
        b_prefix = rng.randint(b_prefix_lo, b_prefix_hi)
        a_units = a % 10

        carry_units = [u for u in range(10) if a_units + u >= 10]
        no_carry_units = [u for u in range(10) if a_units + u < 10]
        if not carry_units or not no_carry_units:
            continue

        b_carry = b_prefix * 10 + rng.choice(carry_units)
        b_no_carry = b_prefix * 10 + rng.choice(no_carry_units)

        if not _is_single_carry(a, b_carry):
            continue

        for tmpl in templates:
            tkey = str(tmpl)
            if counts[tkey] >= n_per_template:
                continue
            key = (a, b_carry, tkey)
            if key in seen:
                continue
            seen.add(key)
            template_str = TEMPLATES[tmpl]
            pairs.append(
                CarryPair(
                    a=a,
                    b_carry=b_carry,
                    b_no_carry=b_no_carry,
                    template=tmpl,
                    prompt_carry=template_str.format(a=a, b=b_carry),
                    prompt_no_carry=template_str.format(a=a, b=b_no_carry),
                )
            )
            counts[tkey] += 1

    return pairs
