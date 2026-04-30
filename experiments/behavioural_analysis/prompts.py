"""Prompt templates for the behavioural analysis sweep.

Four templates per operation:
  T1 — direct answer
  T2 — formal compute
  T3 — contextual word problem
  T4 — chain of thought

T3 is written once per operation with placeholders matching the operand names
used in make_expression().  T4 is a single generic CoT wrapper.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Template IDs
# ---------------------------------------------------------------------------

TEMPLATE_IDS = ["T1", "T2", "T3", "T4"]


# ---------------------------------------------------------------------------
# Single-answer template builders
# ---------------------------------------------------------------------------


def _t1(expr: str) -> str:
    return f"What is {expr}? Answer with just the number. "


def _t2(expr: str) -> str:
    return f"Compute the following: {expr} = "


def _t4(expr: str) -> str:
    return f"What is {expr}? Think step by step, then state the final answer clearly. "


# ---------------------------------------------------------------------------
# Per-operation T3 (contextual word problems)
# ---------------------------------------------------------------------------


def _t3_addition(a: int, b: int, **_) -> str:
    return (
        f"A warehouse has {a} boxes in stock. "
        f"A new shipment of {b} more boxes arrives. "
        f"How many boxes are there in total? Answer with just the number. "
    )


def _t3_subtraction(a: int, b: int, **_) -> str:
    return (
        f"A store begins the day with {a} items. "
        f"By closing time {b} items have been sold. "
        f"How many items remain? Answer with just the number. "
    )


def _t3_multiplication(a: int, b: int, **_) -> str:
    return (
        f"A rectangular grid has {a} rows and {b} columns. "
        f"How many cells does it contain in total? Answer with just the number. "
    )


def _t3_division(a: int, b: int, **_) -> str:
    return (
        f"{a} apples are divided equally among {b} people. "
        f"How many apples does each person receive and how many are left over? "
        f"Give the quotient and the remainder as two integers separated by a space. "
    )


def _t3_exponentiation(a: int, b: int, **_) -> str:
    # a = base, b = exponent
    return (
        f"A scientist measures a quantity of {a} units. "
        f"She needs to raise this quantity to the power of {b}. "
        f"What is the result? Answer with just the number. "
    )


def _t3_gcd(a: int, b: int, **_) -> str:
    return (
        f"Two ropes have lengths {a} cm and {b} cm respectively. "
        f"What is the length of the longest ruler that can measure both ropes "
        f"an exact whole number of times (the GCD)? Answer with just the number. "
    )


def _t3_modular(a: int, b: int, **_) -> str:
    # a = dividend, b = modulus
    return (
        f"{a} students are arranged in rows of {b}. "
        f"How many students are in the last row (which may be incomplete)? "
        f"Answer with just the number. "
    )


_T3_BUILDERS = {
    "addition": _t3_addition,
    "subtraction": _t3_subtraction,
    "multiplication": _t3_multiplication,
    "division": _t3_division,
    "exponentiation": _t3_exponentiation,
    "gcd": _t3_gcd,
    "modular": _t3_modular,
}

# ---------------------------------------------------------------------------
# T1/T2 per-operation expression formatters
# (Division and GCD need slightly different phrasing in T1/T2)
# ---------------------------------------------------------------------------


def _expr_t1_division(a: int, b: int) -> str:
    return (
        f"What is {a} divided by {b}? "
        f"Give the quotient and the remainder as two integers separated by a space."
    )


def _expr_t2_division(a: int, b: int) -> str:
    return (
        f"Compute {a} ÷ {b}. "
        f"Give the quotient and the remainder as two integers separated by a space. ="
    )


def _t4_division(a: int, b: int) -> str:
    return (
        f"What is {a} divided by {b}? "
        f"Think step by step, then clearly state the quotient and the remainder."
    )


# ---------------------------------------------------------------------------
# Public API: build_prompts
# ---------------------------------------------------------------------------


def build_prompts(operation: str, operands: tuple, expression: str) -> dict[str, str]:
    """Return a dict mapping template_id → prompt string for this problem.

    Args:
        operation: One of the canonical operation names.
        operands:  Raw operand tuple as stored in Problem.operands.
        expression: Pre-formatted expression string (from make_expression).

    Returns:
        Dict with keys "T1", "T2", "T3", "T4".
    """
    a, b = operands[0], operands[1]

    if operation == "division":
        return {
            "T1": _expr_t1_division(a, b),
            "T2": _expr_t2_division(a, b),
            "T3": _T3_BUILDERS["division"](a=a, b=b),
            "T4": _t4_division(a, b),
        }

    t3_fn = _T3_BUILDERS.get(operation)
    if t3_fn is None:
        raise ValueError(f"No T3 template for operation: {operation}")

    return {
        "T1": _t1(expression),
        "T2": _t2(expression),
        "T3": t3_fn(a=a, b=b),
        "T4": _t4(expression),
    }
