"""Operand sampling and problem generation for the behavioural analysis sweep.

Each problem is a dataclass capturing operands, the symbolic expression, the
ground truth answer, and the metadata needed to reproduce it exactly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Problem dataclass
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """Single arithmetic problem with all metadata."""

    operation: str
    n_digits: int
    carry_type: str  # "carry_free" | "carry_heavy" | "none"
    seed: int
    expression: str
    ground_truth: str  # string; div has "quotient remainder"
    operands: tuple  # raw operand values

    # Populated during evaluation
    template: str = ""
    prompt: str = ""
    model_answer: str = ""
    correct: bool = False
    per_digit_correct: list[bool] = field(default_factory=list)
    per_digit_confidence: list[float] = field(default_factory=list)
    consistent: bool = True
    extraction_failed: bool = False


# ---------------------------------------------------------------------------
# Carry / borrow helpers
# ---------------------------------------------------------------------------


def _has_carry_free_addition(a: int, b: int) -> bool:
    """True iff adding a+b produces no carry at any digit position."""
    a_s, b_s = str(a), str(b)
    n = max(len(a_s), len(b_s))
    a_s, b_s = a_s.zfill(n), b_s.zfill(n)
    return all(int(a_s[i]) + int(b_s[i]) < 10 for i in range(n))


def _has_carry_heavy_addition(a: int, b: int) -> bool:
    """True iff adding a+b produces a carry at every digit position.

    Position 0 (units): a_0 + b_0 >= 10 (no carry-in).
    Position i > 0: a_i + b_i + 1 >= 10 (carry-in always 1 from previous).
    We use a simple carry simulation to verify.
    """
    a_s, b_s = str(a), str(b)
    n = max(len(a_s), len(b_s))
    a_s, b_s = a_s.zfill(n), b_s.zfill(n)
    carry = 0
    for i in range(n - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry
        if s < 10:
            return False  # no carry at this position
        carry = 1
    return True


def _has_borrow_free_subtraction(a: int, b: int) -> bool:
    """True iff a-b requires no borrow at any digit position (a >= b)."""
    if a < b:
        return False
    a_s, b_s = str(a), str(b)
    n = max(len(a_s), len(b_s))
    a_s, b_s = a_s.zfill(n), b_s.zfill(n)
    return all(int(a_s[i]) >= int(b_s[i]) for i in range(n))


def _has_borrow_heavy_subtraction(a: int, b: int) -> bool:
    """True iff a-b borrows at every digit position except the most significant.

    Construction: a_0 < b_0 starts the borrow chain, and borrow propagates
    through every subsequent position. The leading position must not borrow
    (so a > 0 after subtraction).  A simulation checks this exactly.
    """
    if a <= b:
        return False
    a_s, b_s = str(a), str(b)
    n = max(len(a_s), len(b_s))
    a_s, b_s = a_s.zfill(n), b_s.zfill(n)
    borrow = 0
    # Process right-to-left; track whether every non-leading position borrows.
    borrow_positions = []
    for i in range(n - 1, -1, -1):
        ai, bi = int(a_s[i]), int(b_s[i])
        if ai - borrow < bi:
            borrow_positions.append(True)
            borrow = 1
        else:
            borrow_positions.append(False)
            borrow = 0
    # borrow_positions is in reverse order (position n-1 first after reversal)
    borrow_positions.reverse()
    # Every position except the leading one must borrow
    return all(borrow_positions[1:]) and borrow_positions[0] is False


# ---------------------------------------------------------------------------
# Digit-by-digit samplers (efficient; avoid rejection for large n)
# ---------------------------------------------------------------------------


def _sample_carry_free_addition(n: int, rng: random.Random) -> tuple[int, int]:
    """Sample an n-digit pair (a, b) with no carry at any position."""
    digits_a, digits_b = [], []
    for i in range(n):
        is_leading = i == 0
        lo_a = 1 if is_leading else 0
        # For leading: a_{n-1} in [1,8] and b_{n-1} in [1, 9-a_{n-1}]
        # For others: a_i in [0,9] and b_i in [0, 9-a_i]
        if is_leading:
            ai = rng.randint(lo_a, 8)
            bi = rng.randint(1, 9 - ai)
        else:
            ai = rng.randint(0, 9)
            bi = rng.randint(0, 9 - ai)
        digits_a.append(ai)
        digits_b.append(bi)
    a = int("".join(str(d) for d in digits_a))
    b = int("".join(str(d) for d in digits_b))
    return a, b


def _sample_carry_heavy_addition(n: int, rng: random.Random) -> tuple[int, int]:
    """Sample an n-digit pair (a, b) where every digit position carries.

    Position 0 (units, rightmost in string = index n-1): a_i + b_i >= 10.
    Position i > 0 (carry-in = 1): a_i + b_i + 1 >= 10 → a_i + b_i >= 9.
    """
    digits_a, digits_b = [], []
    for i in range(n):
        is_leading = i == 0
        if is_leading:
            # Leading digit: a_i + b_i >= 10, both in [1, 9]
            ai = rng.randint(1, 9)
            lo_b = max(1, 10 - ai)
            if lo_b > 9:
                ai = 1
                lo_b = 9
            bi = rng.randint(lo_b, 9)
        elif i == n - 1:
            # Units digit (rightmost, no carry-in): a_i + b_i >= 10
            ai = rng.randint(1, 9)
            lo_b = max(0, 10 - ai)
            bi = rng.randint(lo_b, 9)
        else:
            # Middle digits (carry-in = 1): a_i + b_i >= 9
            ai = rng.randint(0, 9)
            lo_b = max(0, 9 - ai)
            bi = rng.randint(lo_b, 9)
        digits_a.append(ai)
        digits_b.append(bi)
    a = int("".join(str(d) for d in digits_a))
    b = int("".join(str(d) for d in digits_b))
    return a, b


def _sample_borrow_free_subtraction(n: int, rng: random.Random) -> tuple[int, int]:
    """Sample (a, b) with a > b and no borrow at any position (a_i >= b_i everywhere)."""
    digits_a, digits_b = [], []
    for i in range(n):
        is_leading = i == 0
        if is_leading:
            # a_0 > b_0 to guarantee a > b
            ai = rng.randint(2, 9)
            bi = rng.randint(1, ai - 1)
        else:
            ai = rng.randint(0, 9)
            bi = rng.randint(0, ai)
        digits_a.append(ai)
        digits_b.append(bi)
    a = int("".join(str(d) for d in digits_a))
    b = int("".join(str(d) for d in digits_b))
    return a, b


def _sample_borrow_heavy_subtraction(n: int, rng: random.Random) -> tuple[int, int]:
    """Sample (a, b) with a > b and borrow at every non-leading digit position.

    Construction:
      - Leading: a_{n-1} >= b_{n-1} + 2 (so the chain of borrows doesn't reach it).
      - Units: a_0 < b_0 (strict, starts the borrow chain).
      - Middle: a_i <= b_i (borrow propagates through borrow_in = 1).
      - b must be an n-digit number: b_{n-1} >= 1.
    """
    if n == 1:
        # Degenerate: can't have borrow-heavy with a single digit and a > b.
        # Fall back to borrow-free.
        return _sample_borrow_free_subtraction(1, rng)

    digits_a, digits_b = [], []
    # Leading digit: a_0 - b_0 - borrow >=0 after all borrows.
    # a_0 >= b_0 + 2 ensures the leading position never borrows.
    a_lead = rng.randint(3, 9)  # need a_lead >= b_lead + 2 and b_lead >= 1
    b_lead = rng.randint(1, a_lead - 2)
    digits_a.append(a_lead)
    digits_b.append(b_lead)

    # Middle positions (i in 1..n-2): a_i <= b_i (borrow propagates)
    for _ in range(n - 2):
        ai = rng.randint(0, 8)
        bi = rng.randint(ai, 9)
        digits_a.append(ai)
        digits_b.append(bi)

    # Units position (last): a_{n-1} < b_{n-1} (strict, starts the chain)
    ai_units = rng.randint(0, 8)
    bi_units = rng.randint(ai_units + 1, 9)
    digits_a.append(ai_units)
    digits_b.append(bi_units)

    a = int("".join(str(d) for d in digits_a))
    b = int("".join(str(d) for d in digits_b))
    return a, b


# ---------------------------------------------------------------------------
# Operand generators per operation
# ---------------------------------------------------------------------------


def _gen_addition(n: int, carry_type: str, rng: random.Random) -> tuple[int, int]:
    lo = 10 ** (n - 1)
    hi = 10**n - 1
    if carry_type == "carry_free":
        a, b = _sample_carry_free_addition(n, rng)
    elif carry_type == "carry_heavy":
        a, b = _sample_carry_heavy_addition(n, rng)
    else:
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
    return a, b


def _gen_subtraction(n: int, carry_type: str, rng: random.Random) -> tuple[int, int]:
    if carry_type == "carry_free":
        a, b = _sample_borrow_free_subtraction(n, rng)
    elif carry_type == "carry_heavy":
        a, b = _sample_borrow_heavy_subtraction(n, rng)
    else:
        lo = 10 ** (n - 1)
        hi = 10**n - 1
        a = rng.randint(lo, hi)
        b = rng.randint(lo, a - 1)
    return a, b


def _gen_multiplication(n: int, rng: random.Random) -> tuple[int, int]:
    lo = 10 ** (n - 1)
    hi = 10**n - 1
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    return a, b


def _gen_division(n: int, rng: random.Random) -> tuple[int, int]:
    """Dividend is n-digit; divisor is a small random 2-digit number."""
    lo = 10 ** (n - 1)
    hi = 10**n - 1
    a = rng.randint(lo, hi)
    b = rng.randint(10, 99)  # 2-digit divisor
    return a, b


def _gen_exponentiation(n: int, exp: int, rng: random.Random) -> tuple[int, int]:
    """Base is n-digit, exponent is fixed."""
    lo = 10 ** (n - 1)
    hi = 10**n - 1
    base = rng.randint(lo, hi)
    return base, exp


def _gen_gcd(n: int, rng: random.Random) -> tuple[int, int]:
    lo = 10 ** (n - 1)
    hi = 10**n - 1
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    return a, b


def _gen_modular(n: int, m: int, rng: random.Random) -> tuple[int, int]:
    lo = 10 ** (n - 1)
    hi = 10**n - 1
    a = rng.randint(lo, hi)
    return a, m


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------


def compute_answer(operation: str, operands: tuple) -> str:
    """Return the canonical string representation of the correct answer."""
    if operation == "addition":
        return str(operands[0] + operands[1])
    elif operation == "subtraction":
        return str(operands[0] - operands[1])
    elif operation == "multiplication":
        return str(operands[0] * operands[1])
    elif operation == "division":
        a, b = operands
        q, r = divmod(a, b)
        return f"{q} {r}"
    elif operation == "exponentiation":
        base, exp = operands
        return str(base**exp)
    elif operation == "gcd":
        return str(math.gcd(operands[0], operands[1]))
    elif operation == "modular":
        return str(operands[0] % operands[1])
    else:
        raise ValueError(f"Unknown operation: {operation}")


def make_expression(operation: str, operands: tuple) -> str:
    """Return the symbolic expression string for a problem."""
    if operation == "addition":
        return f"{operands[0]} + {operands[1]}"
    elif operation == "subtraction":
        return f"{operands[0]} - {operands[1]}"
    elif operation == "multiplication":
        return f"{operands[0]} * {operands[1]}"
    elif operation == "division":
        return f"{operands[0]} / {operands[1]}"
    elif operation == "exponentiation":
        return f"{operands[0]} ^ {operands[1]}"
    elif operation == "gcd":
        return f"gcd({operands[0]}, {operands[1]})"
    elif operation == "modular":
        return f"{operands[0]} mod {operands[1]}"
    else:
        raise ValueError(f"Unknown operation: {operation}")


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

# All carry types per operation
CARRY_TYPES: dict[str, list[str]] = {
    "addition": ["carry_free", "carry_heavy"],
    "subtraction": ["carry_free", "carry_heavy"],
    "multiplication": ["none"],
    "division": ["none"],
    "exponentiation": ["none"],
    "gcd": ["none"],
    "modular": ["none"],
}

# Sub-variants for parameterised operations
EXPONENTS = [2, 3, 4, 5]
MODULI = [7, 11, 13]


def generate_problems(
    operation: str,
    n_digits: int,
    carry_type: str,
    n_samples: int,
    base_seed: int,
    *,
    exp: int | None = None,
    modulus: int | None = None,
) -> list[Problem]:
    """Generate n_samples problems for one (operation, n_digits, carry_type) cell.

    The seed for the i-th sample is deterministically derived from base_seed so
    that cells are fully reproducible and independent of each other.
    """
    problems: list[Problem] = []
    for i in range(n_samples):
        seed = base_seed + i
        rng = random.Random(seed)

        if operation == "addition":
            operands = _gen_addition(n_digits, carry_type, rng)
        elif operation == "subtraction":
            operands = _gen_subtraction(n_digits, carry_type, rng)
        elif operation == "multiplication":
            operands = _gen_multiplication(n_digits, rng)
        elif operation == "division":
            operands = _gen_division(n_digits, rng)
        elif operation == "exponentiation":
            assert exp is not None
            operands = _gen_exponentiation(n_digits, exp, rng)
        elif operation == "gcd":
            operands = _gen_gcd(n_digits, rng)
        elif operation == "modular":
            assert modulus is not None
            operands = _gen_modular(n_digits, modulus, rng)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        expression = make_expression(operation, operands)
        ground_truth = compute_answer(operation, operands)

        problems.append(
            Problem(
                operation=operation,
                n_digits=n_digits,
                carry_type=carry_type,
                seed=seed,
                expression=expression,
                ground_truth=ground_truth,
                operands=operands,
            )
        )
    return problems


def build_all_problems(
    n_digits_range: range = range(1, 9),
    n_samples: int = 100,
    base_seed: int = 42,
) -> list[Problem]:
    """Generate all problems for the full behavioural sweep.

    Seed space is partitioned so each (operation, variant, n_digits, carry_type,
    sample_index) maps to a unique seed. This guarantees full reproducibility.
    """
    all_problems: list[Problem] = []
    op_idx = 0  # incremented for each (operation, variant) group

    def _next_base(group_size: int) -> int:
        nonlocal op_idx
        start = base_seed + op_idx * 10_000
        op_idx += 1
        return start

    for n in n_digits_range:
        # Addition / Subtraction (carry types)
        for op in ("addition", "subtraction"):
            for ct in CARRY_TYPES[op]:
                b = _next_base(n_samples)
                all_problems.extend(generate_problems(op, n, ct, n_samples, b))

        # Multiplication, Division, GCD
        for op in ("multiplication", "division", "gcd"):
            b = _next_base(n_samples)
            all_problems.extend(generate_problems(op, n, "none", n_samples, b))

        # Exponentiation (one variant per exponent)
        for exp in EXPONENTS:
            b = _next_base(n_samples)
            all_problems.extend(
                generate_problems("exponentiation", n, "none", n_samples, b, exp=exp)
            )

        # Modular arithmetic (one variant per modulus)
        for m in MODULI:
            b = _next_base(n_samples)
            all_problems.extend(generate_problems("modular", n, "none", n_samples, b, modulus=m))

    return all_problems
