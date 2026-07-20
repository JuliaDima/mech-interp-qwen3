"""Prime number concept dataset (3-digit numbers only).

Pos: n is prime → "Yes"
Neg: n is not prime → "No"

All templates use "Yes or No: " prefix so the model reliably outputs "Yes" or
"No" rather than a number or other continuation.

n_pos, n_neg are both drawn from [100, 999] (3 digits). n_neg = n_pos + offset
for a random small offset in {-3,-2,-1,1,2,3} that lands on a composite,
staying within the 3-digit range. Offset is varied to avoid the fixed-gap
confound.
"""

from __future__ import annotations

import random

from experiments.concept_localization.concept_pair import ConceptPair

TEMPLATES: dict[str, tuple[str, str, str]] = {
    "T0": ("Is {n} a prime number? Answer yes or no: ", "yes", "no"),
    "T1": ("{n} is a prime number? Answer yes or no: ", "yes", "no"),
    "T2": ("is {n} prime? Answer yes or no: ", "yes", "no"),
}


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, int(n**0.5) + 1, 2):
        if n % d == 0:
            return False
    return True


def generate_prime_pairs(
    n_per_template: int = 100,
    templates: list[str] | None = None,
    seed: int = 42,
) -> list[ConceptPair]:
    """Pairs n_pos=prime vs n_neg=n_pos+offset (composite), both 3-digit."""
    if templates is None:
        templates = list(TEMPLATES)

    rng = random.Random(seed)
    primes = [n for n in range(100, 1000) if _is_prime(n)]

    pairs: list[ConceptPair] = []
    seen: set[tuple[int, int]] = set()
    counts = {t: 0 for t in templates}
    attempts = 0

    while attempts < n_per_template * len(templates) * 200 and any(
        v < n_per_template for v in counts.values()
    ):
        attempts += 1
        n_pos = rng.choice(primes)

        offsets = [-3, -2, -1, 1, 2, 3]
        rng.shuffle(offsets)
        n_neg = None
        for offset in offsets:
            candidate = n_pos + offset
            if 100 <= candidate <= 999 and not _is_prime(candidate):
                n_neg = candidate
                break
        if n_neg is None:
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


if __name__ == "__main__":
    pairs = generate_prime_pairs(n_per_template=20)
    assert len(pairs) == 60
    for p in pairs[:6]:
        assert 100 <= p.meta["n_pos"] <= 999 and 100 <= p.meta["n_neg"] <= 999
        assert _is_prime(p.meta["n_pos"]) and not _is_prime(p.meta["n_neg"])
    print(f"OK: {len(pairs)} pairs, balanced pos/neg, all 3-digit")
