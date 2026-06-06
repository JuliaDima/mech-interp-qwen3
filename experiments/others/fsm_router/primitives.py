"""Primitive definitions and synthetic data generation for the FSM router.

Each PrimitiveDef specifies:
  - name, n_states: passed to PrimitiveRouter to build the SoftFSM
  - label_fn: given the predicate sequence of an expression, returns a
              per-token activation label (0.0 / 1.0, cumulative: once the
              primitive completes, the label stays 1 for all later tokens)

Shared predicates:
  NUMBER, PLUS, TIMES, MOD, DIV, EQUALS, OPEN_PAREN, CLOSE_PAREN, OTHER

Note on parity vs modular: at predicate level both look like NUM MOD NUM.
Parity is modular arithmetic mod 2; the FSM fires identically for both.
Distinguishing them requires knowing the actual modulus value, which is
beyond the predicate abstraction.  Downstream steering vectors can be
specialised separately once detection is established.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from experiments.fsm_router.predicates import P, tokenize_and_map


@dataclass(frozen=True)
class PrimitiveDef:
    name: str
    n_states: int
    label_fn: Callable[[list[P]], list[float]]


# ---------------------------------------------------------------------------
# Label functions — rule-based ground truth
# Each returns a cumulative binary sequence: 0 before completion, 1 after.
# ---------------------------------------------------------------------------


def _labels_addition(preds: list[P]) -> list[float]:
    seen_num = seen_op = done = False
    out = []
    for p in preds:
        if not done:
            if p == P.NUMBER and not seen_num:
                seen_num = True
            elif p == P.PLUS and seen_num:
                seen_op = True
            elif p == P.NUMBER and seen_op:
                done = True
        out.append(1.0 if done else 0.0)
    return out


def _labels_subtraction(preds: list[P]) -> list[float]:
    seen_num = seen_op = done = False
    out = []
    for p in preds:
        if not done:
            if p == P.NUMBER and not seen_num:
                seen_num = True
            elif p == P.MINUS and seen_num:
                seen_op = True
            elif p == P.NUMBER and seen_op:
                done = True
        out.append(1.0 if done else 0.0)
    return out


def _labels_multiplication(preds: list[P]) -> list[float]:
    seen_num = seen_op = done = False
    out = []
    for p in preds:
        if not done:
            if p == P.NUMBER and not seen_num:
                seen_num = True
            elif p == P.TIMES and seen_num:
                seen_op = True
            elif p == P.NUMBER and seen_op:
                done = True
        out.append(1.0 if done else 0.0)
    return out


def _labels_modular(preds: list[P]) -> list[float]:
    # Fires when: any expression followed by MOD and a NUMBER
    seen_expr = seen_mod = done = False
    out = []
    for p in preds:
        if not done:
            if p == P.NUMBER and not seen_expr:
                seen_expr = True
            elif p == P.MOD and seen_expr:
                seen_mod = True
            elif p == P.NUMBER and seen_mod:
                done = True
        out.append(1.0 if done else 0.0)
    return out


# ---------------------------------------------------------------------------
# Primitive registry
# ---------------------------------------------------------------------------

PRIMITIVE_DEFS: list[PrimitiveDef] = [
    PrimitiveDef("addition", n_states=4, label_fn=_labels_addition),
    PrimitiveDef("subtraction", n_states=4, label_fn=_labels_subtraction),
    PrimitiveDef("multiplication", n_states=4, label_fn=_labels_multiplication),
    PrimitiveDef("modular", n_states=4, label_fn=_labels_modular),
]

FSM_SPECS: list[tuple[str, int]] = [(p.name, p.n_states) for p in PRIMITIVE_DEFS]


# ---------------------------------------------------------------------------
# Synthetic expression templates
# ---------------------------------------------------------------------------

# Templates must produce token sequences that match the label_fn patterns.
# Each label_fn expects NUM OP NUM (in that order at the predicate level).
# "add X and Y" (PLUS before NUM) and "multiply X by Y" (TIMES before NUM)
# are excluded because the operator precedes the first number.
_T_ADD = [
    "{a} + {b}",
    "{a} plus {b}",
    "({a} + {b})",
    "compute {a} + {b}",
    "{a} + {b} =",
    "{a} + {b} + {c}",
]
_T_SUB = [
    "{a} - {b}",
    "{a} minus {b}",
    "({a} - {b})",
    "compute {a} - {b}",
    "{a} - {b} =",
]
_T_MUL = [
    "{a} * {b}",
    "{a} times {b}",
    "{a} × {b}",
    "({a} * {b})",
    "compute {a} * {b}",
    "{a} * {b} =",
]
_T_MOD = [
    "{a} mod {m}",
    "{a} % {m}",
    "({a} + {b}) mod {m}",
    "({a} * {b}) mod {m}",
    "compute {a} mod {m}",
    "{a} mod {m} = {r}",
    "{a} % 2",
]
_T_MIXED = [
    "({a} + {b}) * {c}",
    "({a} - {b}) mod {m}",
    "{a} * {b} + {c}",
    "({a} + {b}) mod {m} = {r}",
    "{a} + {b} * {c}",
]


def _rn(rng: random.Random, lo: int = 1, hi: int = 999) -> int:
    return rng.randint(lo, hi)


def generate_expressions(
    n_samples: int,
    seed: int = 42,
) -> list[tuple[str, list[list[float]]]]:
    """Generate synthetic math expressions with per-token primitive labels.

    Returns:
        list of (text, label_matrix) where label_matrix is a list of K lists,
        one per primitive, each of length T (number of tokens in text).
    """
    rng = random.Random(seed)
    K = len(PRIMITIVE_DEFS)

    pool = (
        [(t, "add") for t in _T_ADD]
        + [(t, "sub") for t in _T_SUB]
        + [(t, "mul") for t in _T_MUL]
        + [(t, "mod") for t in _T_MOD]
        + [(t, "mix") for t in _T_MIXED]
    )

    examples: list[tuple[str, list[list[float]]]] = []
    for _ in range(n_samples):
        tmpl, _ = rng.choice(pool)
        a, b, c = _rn(rng), _rn(rng), _rn(rng)
        m = rng.choice([2, 3, 5, 7, 10, 12])
        r = (a + b) % m
        text = tmpl.format(a=a, b=b, c=c, m=m, r=r)

        pairs = tokenize_and_map(text)
        preds = [p for _, p in pairs]
        label_matrix = [pdef.label_fn(preds) for pdef in PRIMITIVE_DEFS]
        examples.append((text, label_matrix))

    return examples
