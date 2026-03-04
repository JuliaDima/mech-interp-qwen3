"""Target token helpers for the addition reproduction experiment.

Maps each prompt to its expected numeric answer and provides a robust
helper that selects the *first* token of the correct answer string for use
as the attribution / logit target.

Design choice — "first token":
  Anthropic attributtes from the single most-constrained logit position,
  which for "95" on the Qwen3 tokenizer is the first digit token produced
  after the equals sign.  We select the first token of the expected answer
  string and document that choice here.  If the answer tokenises to a single
  token (e.g. "9", "42") there is no ambiguity.  For multi-token answers
  (e.g. "100" → ["1", "00"] or ["100"]) we return the *full* span so callers
  can pick what they need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .prompts import CALC_GRID, FOCUS_PROMPT, GENERALIZATION_PROMPTS, NL_VARIANT

if TYPE_CHECKING:
    from mechinterp_qwen3.attribution_model import AttributionModel

# ---------------------------------------------------------------------------
# Static expected-answer map (all prompts used in the experiment)
# ---------------------------------------------------------------------------

EXPECTED_ANSWERS: dict[str, str] = {
    FOCUS_PROMPT: "95",
    NL_VARIANT: "95",
}

# Build from CALC_GRID
for _entry in CALC_GRID:
    EXPECTED_ANSWERS[_entry["prompt"]] = _entry["answer"]

# Add generalization prompts
for _g in GENERALIZATION_PROMPTS:
    EXPECTED_ANSWERS[_g["prompt"]] = _g["answer"]


# ---------------------------------------------------------------------------
# Token-selection helper
# ---------------------------------------------------------------------------


def get_target_tokens(
    prompt: str,
    model: AttributionModel,
    *,
    span: str | None = None,
) -> list[int]:
    """Return the token IDs of the expected answer for *prompt*.

    Args:
        prompt: The input prompt string (must be in EXPECTED_ANSWERS).
        model:  The loaded AttributionModel (used for its tokenizer).
        span:   Optional override — tokenise this string instead of looking
                up from EXPECTED_ANSWERS.  Useful for ad-hoc targets.

    Returns:
        List of token IDs representing the expected answer string.
        Callers that want only the *first* token should take ``[0]``.

    Raises:
        KeyError: if *prompt* is not found in EXPECTED_ANSWERS and no *span*
            override is given.
        ValueError: if the answer tokenises to zero tokens.

    Selection rationale:
        We tokenise the expected answer string with ``add_special_tokens=False``
        and return the full token span.  For the primary analysis (operand plots
        and attribution) we use only the **first** token — this mirrors
        Anthropic's approach of attributing from the first constrained logit
        position (where the model "commits" to the tens digit of the sum).
    """
    if span is None:
        span = EXPECTED_ANSWERS[prompt]

    tokenizer = model.tokenizer  # type: ignore[attr-defined]
    token_ids: list[int] = tokenizer(
        span,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]

    if len(token_ids) == 0:
        raise ValueError(
            f"Answer string {span!r} tokenised to zero tokens. "
            "Check the tokenizer and answer string."
        )

    return token_ids


def get_first_target_token(
    prompt: str,
    model: AttributionModel,
    *,
    span: str | None = None,
) -> int:
    """Convenience wrapper — returns only the *first* target token ID.

    This is the token used for logit attribution in all operand plots and
    graph computations, matching Anthropic's primary analysis position.
    """
    return get_target_tokens(prompt, model, span=span)[0]


def describe_tokenisation(
    prompt: str,
    model: AttributionModel,
) -> str:
    """Human-readable description of how the expected answer tokenises.

    Useful for debugging and for the README's tokenisation section.
    """
    tokenizer = model.tokenizer  # type: ignore[attr-defined]
    answer = EXPECTED_ANSWERS[prompt]
    token_ids = get_target_tokens(prompt, model)
    token_strs = [tokenizer.decode([t]) for t in token_ids]

    lines = [
        f"Prompt:  {prompt!r}",
        f"Answer:  {answer!r}",
        f"Tokens:  {token_strs}  (ids: {token_ids})",
        f"Using:   token_ids[0] = {token_strs[0]!r} (id {token_ids[0]}) as attribution target",
    ]
    return "\n".join(lines)
