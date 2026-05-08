"""Shared predicate vocabulary for the FSM router.

All primitives share the same predicate set. Predicates are rule-based
(not learned) so the mapping generalises to unseen numbers automatically:
"37", "1000", "99" all map to NUMBER regardless of value.

Primitives can extend the base set with task-specific predicates but
should not duplicate base predicates.

Tokenization modes
------------------
• tokenize_and_map(text)          — regex tokenizer, keeps whole numbers together
• from_token_strings(tok_strs)    — accepts pre-tokenized strings (e.g. Qwen tokens)
• from_token_strings(..., dedup=True) — collapse consecutive identical predicates,
  making the sequence invariant to digit-level vs whole-number tokenization.
  Use this whenever feeding Qwen token strings to the FSM router.
"""

from __future__ import annotations

import re
from enum import IntEnum


class P(IntEnum):
    NUMBER = 0
    PLUS = 1
    MINUS = 2
    TIMES = 3
    DIV = 4
    MOD = 5
    EQUALS = 6
    OPEN_PAREN = 7
    CLOSE_PAREN = 8
    COMMA = 9
    OTHER = 10


N_PREDICATES = len(P)

_KEYWORD_MAP: dict[str, P] = {
    # Addition
    "+": P.PLUS,
    "plus": P.PLUS,
    "add": P.PLUS,
    # Subtraction
    "-": P.MINUS,
    "minus": P.MINUS,
    "subtract": P.MINUS,
    # Multiplication
    "*": P.TIMES,
    "×": P.TIMES,
    "·": P.TIMES,
    "times": P.TIMES,
    "mul": P.TIMES,
    "multiply": P.TIMES,
    # Division
    "/": P.DIV,
    "÷": P.DIV,
    "div": P.DIV,
    "divide": P.DIV,
    # Modular
    "mod": P.MOD,
    "%": P.MOD,
    "modulo": P.MOD,
    # Equality / assignment
    "=": P.EQUALS,
    "==": P.EQUALS,
    "equals": P.EQUALS,
    "is": P.EQUALS,
    # Brackets
    "(": P.OPEN_PAREN,
    "[": P.OPEN_PAREN,
    "{": P.OPEN_PAREN,
    ")": P.CLOSE_PAREN,
    "]": P.CLOSE_PAREN,
    "}": P.CLOSE_PAREN,
    # Comma
    ",": P.COMMA,
}

# Tokeniser: keeps numbers together, splits on operators
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[+\-*/÷×%=(),\[\]{}]|\w+")


def token_to_predicate(tok: str) -> P:
    tok = tok.strip().lower()
    if tok in _KEYWORD_MAP:
        return _KEYWORD_MAP[tok]
    try:
        float(tok)
        return P.NUMBER
    except ValueError:
        return P.OTHER


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def tokenize_and_map(text: str) -> list[tuple[str, P]]:
    """Split text into tokens and map each to a predicate."""
    return [(tok, token_to_predicate(tok)) for tok in tokenize(text)]


def _dedup(pairs: list[tuple[str, P]]) -> list[tuple[str, P]]:
    """Collapse consecutive identical predicates, keeping the first token string."""
    if not pairs:
        return pairs
    out = [pairs[0]]
    for tok, pred in pairs[1:]:
        if pred != out[-1][1]:
            out.append((tok, pred))
    return out


def token_groups_from_strings(tok_strs: list[str]) -> list[tuple[P, int, int]]:
    """Map pre-tokenized strings to deduplicated predicate groups.

    Returns a list of (pred, first_tok_idx, last_tok_idx) — one entry per
    contiguous run of the same predicate.  Whitespace-only tokens (bare "Ġ")
    are skipped so indices refer to positions in the original tok_strs list.

    Used to map an FSM firing position in *predicate space* back to the
    corresponding *token position* (last_tok_idx of the firing group).

    Example:
        ["3", "7", "Ġ+", "Ġ", "1", "1", "5"]
        → [(NUMBER, 0, 1), (PLUS, 2, 2), (NUMBER, 4, 6)]
        (index 3 is the standalone space, skipped)
    """
    groups: list[list] = []  # [pred, first_idx, last_idx] — mutable for extend
    for orig_idx, tok in enumerate(tok_strs):
        cleaned = tok.lstrip("Ġ▁ ")
        if not cleaned.strip():
            continue
        pred = token_to_predicate(cleaned)
        if groups and groups[-1][0] == pred:
            groups[-1][2] = orig_idx
        else:
            groups.append([pred, orig_idx, orig_idx])
    return [(P(g[0]), g[1], g[2]) for g in groups]


def from_token_strings(
    tok_strs: list[str],
    *,
    dedup: bool = True,
) -> list[tuple[str, P]]:
    """Map a pre-tokenized list of strings (e.g. from Qwen tokenizer) to predicates.

    Args:
        tok_strs: token strings with optional leading whitespace (Ġ-prefix stripped).
        dedup:    collapse consecutive identical predicates.  Must be True when the
                  tokenizer is digit-level (Qwen) so that ["3","7","+","1","1","5"]
                  produces [NUMBER, PLUS, NUMBER] rather than [NUMBER,NUMBER,PLUS,
                  NUMBER,NUMBER,NUMBER], matching the FSM's training distribution.

    Returns:
        list of (token_string, predicate) pairs, length ≤ len(tok_strs).
    """
    # Strip leading whitespace markers (Ġ, ▁, spaces); skip pure-whitespace tokens
    cleaned = [t.lstrip("Ġ▁ ") for t in tok_strs]
    pairs = [(tok, token_to_predicate(tok)) for tok in cleaned if tok.strip()]
    return _dedup(pairs) if dedup else pairs
