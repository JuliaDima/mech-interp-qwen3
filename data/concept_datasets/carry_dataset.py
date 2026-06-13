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

"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES: dict[str, tuple[str, str | None, str | None]] = {
    "T0": ("what is {a}+{b}? Answer: ", None, None),  # prediction computed per pair
    "T1": ("Does {a}+{b} have a carry? Answer yes or no: ", "yes", "no"),
    "T2": ("calc: {a}+{b}= ", None, None),       # prediction computed per pair
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
    """Generate contrastive carry pairs covering all reachable ones-digit cells.

    Enumerates all 99 reachable (ones_a, ones_b) combinations and generates
    n_per_cell = max(1, round(n_per_template / 99)) pairs per cell per template,
    keeping the total volume close to n_per_template.

    For carry cells (ones_a + ones_b >= 10) the target cell is the pos anchor;
    for no-carry cells it is the neg anchor. The partner digit is sampled
    uniformly from the full range of valid options so delta magnitudes are
    diverse. Balance is exactly 50/50 by construction. The only unreachable
    cell is (0, 0).
    """
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    n_per_cell = max(1, round(n_per_template / 99))
    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int, int, int, str]] = set()

    for da in range(10):
        for db in range(10):
            if da == 0 and db == 0:
                continue  # unreachable: no single-digit change can produce a carry

            is_carry = (da + db >= 10)

            # Valid partner digits for each variation direction
            if is_carry:
                vary_a_partners = list(range(0, 10 - db))               # d' + db < 10
                vary_b_partners = list(range(0, 10 - da))               # da + d' < 10
            else:
                vary_a_partners = list(range(max(0, 10 - db), 10)) if db > 0 else []
                vary_b_partners = list(range(max(0, 10 - da), 10)) if da > 0 else []

            valid_dirs = (
                (['a'] if vary_a_partners else []) +
                (['b'] if vary_b_partners else [])
            )
            if not valid_dirs:
                continue

            for t in templates:
                generated = 0
                attempts = 0
                while generated < n_per_cell and attempts < n_per_cell * 200:
                    attempts += 1

                    # Random higher digits; isolation: tens digits must sum to <= 8
                    ha = rng.randint(1, 9)
                    ma = rng.randint(0, 8)
                    hb = rng.randint(1, 9)
                    mb = rng.randint(0, 8 - ma)

                    vary_dir = rng.choice(valid_dirs)
                    if vary_dir == 'a':
                        partner = rng.choice(vary_a_partners)
                        a_anchor  = ha * 100 + ma * 10 + da
                        b_anchor  = hb * 100 + mb * 10 + db
                        a_partner = ha * 100 + ma * 10 + partner
                        b_partner = b_anchor
                    else:
                        partner = rng.choice(vary_b_partners)
                        a_anchor  = ha * 100 + ma * 10 + da
                        b_anchor  = hb * 100 + mb * 10 + db
                        a_partner = a_anchor
                        b_partner = hb * 100 + mb * 10 + partner

                    if is_carry:
                        a_pos, b_pos = a_anchor, b_anchor
                        a_neg, b_neg = a_partner, b_partner
                    else:
                        a_neg, b_neg = a_anchor, b_anchor
                        a_pos, b_pos = a_partner, b_partner

                    key = (a_pos, b_pos, a_neg, b_neg, t)
                    if key in seen:
                        continue
                    seen.add(key)

                    fmt, predict_pos_tmpl, predict_neg_tmpl = TEMPLATES[t]
                    pairs.append(
                        ConceptPair(
                            prompt_pos=fmt.format(a=a_pos, b=b_pos),
                            prompt_neg=fmt.format(a=a_neg, b=b_neg),
                            label_pos="carry",
                            label_neg="no_carry",
                            predict_pos=predict_pos_tmpl if predict_pos_tmpl is not None else str(a_pos + b_pos),
                            predict_neg=predict_neg_tmpl if predict_neg_tmpl is not None else str(a_neg + b_neg),
                            template=t,
                            meta={
                                "a_pos": a_pos,
                                "b_pos": b_pos,
                                "a_neg": a_neg,
                                "b_neg": b_neg,
                                "n_digits": 3,
                            },
                        )
                    )
                    generated += 1

    return pairs
